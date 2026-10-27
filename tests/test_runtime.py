"""The runtime, tested against the real exported model where there is one."""

import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from wakemeup_runtime import Resampler, WakeWord

#: A real exported model: the one named in $WAKEMEUP_TEST_MODEL, or, inside the
#: training repository, its models/Kosmo. Most of the suite runs without it;
#: the tests that need it skip.
MODEL = Path(
    os.environ.get("WAKEMEUP_TEST_MODEL")
    or Path(__file__).resolve().parents[2] / "models" / "Kosmo" / "Kosmo.onnx"
)
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="no exported model to test against")


# -- resampling -------------------------------------------------------------


@pytest.mark.parametrize("source", [48_000, 44_100, 32_000, 22_050, 16_000])
def test_resampler_holds_the_rate_and_the_tone(source):
    """A 440 Hz tone must stay 440 Hz, at the right length, block by block."""
    target, seconds = 16_000, 2.0
    resampler = Resampler(source, target)
    t = np.arange(int(source * seconds)) / source
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    block = source // 40  # 25 ms, as a sound card would deliver it
    out = np.concatenate([resampler.process(tone[i : i + block]) for i in range(0, tone.size, block)])

    assert abs(out.size / seconds - target) < target * 0.01  # no drift
    settled = out[target // 4 : -target // 4]
    spectrum = np.abs(np.fft.rfft(settled * np.hanning(settled.size)))
    peak = np.fft.rfftfreq(settled.size, 1 / target)[spectrum.argmax()]
    assert abs(peak - 440) < 5
    assert 0.4 < np.abs(settled).max() < 0.6  # amplitude preserved


def test_resampler_suppresses_what_would_alias():
    """A 12 kHz tone cannot survive a move to 16 kHz -- it must be filtered,
    not folded down to 4 kHz, which is what a plain decimation would do."""
    source, target = 48_000, 16_000
    resampler = Resampler(source, target)
    t = np.arange(source) / source
    tone = (0.5 * np.sin(2 * np.pi * 12_000 * t)).astype(np.float32)
    out = np.concatenate([resampler.process(tone[i : i + 1200]) for i in range(0, tone.size, 1200)])

    settled = out[2000:-2000]
    assert np.abs(settled).max() < 0.05, "the tone aliased instead of being filtered out"


def test_resampler_is_a_passthrough_at_the_same_rate():
    resampler = Resampler(16_000, 16_000)
    assert not resampler.active
    block = np.random.default_rng(0).standard_normal(1600).astype(np.float32)
    assert np.array_equal(resampler.process(block), block)


def test_resampler_takes_any_block_size():
    resampler = Resampler(48_000, 16_000)
    rng = np.random.default_rng(1)
    total = 0
    for _ in range(60):
        size = int(rng.integers(1, 2000))
        total += resampler.process(rng.standard_normal(size).astype(np.float32)).size
    assert total > 0


# -- loading ----------------------------------------------------------------


def test_missing_model_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no wakeword model"):
        WakeWord(tmp_path / "nothing.onnx")


def test_missing_sidecar_says_they_are_a_pair(tmp_path):
    model = tmp_path / "Word.onnx"
    model.write_bytes(b"not really a model")
    with pytest.raises(FileNotFoundError, match="beside it"):
        WakeWord(model)


def test_foreign_sidecar_is_rejected(tmp_path):
    model = tmp_path / "Word.onnx"
    model.write_bytes(b"not really a model")
    (tmp_path / "Word.json").write_text(json.dumps({"format": "something.else"}))
    with pytest.raises(ValueError, match="not a wakemeup sidecar"):
        WakeWord(model)


def test_the_gate_is_the_one_the_threshold_was_calibrated_behind():
    """The threshold only means its false-wake rate behind the gate it was
    measured with, so that gate travels in the sidecar and is the default."""
    meta = {"detect": {"min_hits": 3, "refractory_seconds": 1.5, "step_seconds": 0.08}}
    assert WakeWord._gate(meta, None, None, None) == (3, 1.5, 80)
    # asked-for values win; a sidecar from before the gate was recorded falls back
    assert WakeWord._gate(meta, 2, None, 100) == (2, 1.5, 100)
    assert WakeWord._gate({}, None, None, None) == (2, 2.0, 100)


# -- the real model ---------------------------------------------------------


@needs_model
def test_reads_its_own_parameters_from_the_sidecar():
    ear = WakeWord(MODEL)
    assert ear.word
    assert ear.sample_rate == 16_000
    assert ear.window == int(ear.clip_seconds * ear.sample_rate)
    assert 0.0 < ear.threshold < 1.0
    assert ear.step == ear.sample_rate * ear.step_ms // 1000


@needs_model
def test_threshold_can_be_overridden():
    assert WakeWord(MODEL, threshold=0.99).threshold == 0.99


@needs_model
def test_silence_is_never_the_wakeword():
    ear = WakeWord(MODEL)
    quiet = np.zeros(ear.sample_rate * 5, dtype=np.float32)
    assert not ear.feed(quiet)
    assert ear.probability < ear.threshold


@needs_model
def test_accepts_int16_bytes_and_floats_alike():
    ear = WakeWord(MODEL)
    n = ear.sample_rate
    floats = (np.random.default_rng(0).standard_normal(n) * 0.01).astype(np.float32)
    ints = (floats * 32767).astype(np.int16)
    assert ear.feed(floats) is False
    ear.reset()
    assert ear.feed(ints) is False
    ear.reset()
    assert ear.feed(ints.tobytes()) is False


@needs_model
def test_block_size_does_not_change_the_score():
    """The same audio must score the same whether it arrives in one piece or
    a hundred -- otherwise the detector depends on the sound card's buffer."""
    ear = WakeWord(MODEL)
    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(ear.sample_rate * 3) * 0.05).astype(np.float32)

    ear.feed(audio)
    whole = ear.probability
    ear.reset()
    for i in range(0, audio.size, 137):  # deliberately not a round number
        ear.feed(audio[i : i + 137])
    assert abs(ear.probability - whole) < 1e-4


@needs_model
def test_reset_clears_the_refractory_countdown():
    """After a detection the detector goes quiet for a while. reset() must
    lift that, or a caller that stops listening during a conversation comes
    back deaf."""
    ear = WakeWord(MODEL)
    ear._muted = ear.sample_rate * 10  # as if it had just fired
    ear.reset()
    assert ear._muted == 0


@needs_model
def test_hears_the_wakeword_and_ignores_other_speech():
    """The one test that says the thing works. Needs the training corpus."""
    raw = Path("H:/wakeword/raw_data")
    samples_dir = Path("H:/wakeword/training_data/positive")
    if not raw.is_dir() or not samples_dir.is_dir():
        pytest.skip("training audio not mounted")

    def read(path):
        with wave.open(str(path), "rb") as handle:
            data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            return data.astype(np.float32) / 32768.0, handle.getframerate()

    ear = WakeWord(MODEL)
    # the bare word in two voices the current models are validated on, not trained on
    positives = [
        path
        for voice in ("sage", "ballad")
        for path in sorted(samples_dir.glob(f"kosmo_plain_openai_{voice}_*.wav"))[:6]
    ]
    assert positives, "no wakeword clips found"

    heard = 0
    for path in positives:
        samples, rate = read(path)
        padded = np.concatenate([np.zeros(rate // 2, np.float32), samples])
        heard += bool(ear.scan(padded, rate))
    assert heard >= len(positives) - 1, f"only heard {heard}/{len(positives)}"

    speech = sorted((raw / "negative" / "librivox").glob("*.wav"))[:1]
    samples, rate = read(speech[0])
    false_alarms = ear.scan(samples[: 60 * rate], rate)
    assert len(false_alarms) <= 1, f"{len(false_alarms)} false wakes in a minute of speech"


# -- openWakeWord models ----------------------------------------------------

#: Where the feature models are cached (the training fetches them there too); the
#: tests that need them skip rather than download.
FEATURES = Path(os.environ.get("WAKEMEUP_CACHE_DIR") or Path.home() / ".cache" / "wakemeup") / "oww"
needs_features = pytest.mark.skipif(
    not (FEATURES / "embedding_model.onnx").is_file(), reason="no embedding_model.onnx in the cache"
)
#: The real model's head alone, in openWakeWord's layout (`wakemeup export --oww-head`)
OWW_HEAD = MODEL.with_name(MODEL.stem + ".oww-head.onnx")


def tiny_oww_model(path: Path, n: int = 16) -> Path:
    """An openWakeWord-shaped model: ``[1, n, 96]`` embeddings in, ``[1, 1]`` probability out."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nodes = [
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1, 2], keepdims=1),
        helper.make_node("Flatten", ["mean"], ["flat"], axis=1),
        helper.make_node("Sigmoid", ["flat"], ["p"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, n, 96])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, [1, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_the_mel_spectrogram_is_openwakewords():
    """The numpy front end against openWakeWord's own melspectrogram.onnx."""
    from wakemeup_runtime._oww import melspectrogram

    reference = FEATURES / "melspectrogram.onnx"
    if not reference.is_file():
        pytest.skip("no melspectrogram.onnx in the cache")
    import onnxruntime as ort

    pcm = np.random.default_rng(1).standard_normal(16_000).astype(np.float32) * 3000
    session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    theirs = np.squeeze(session.run(None, {session.get_inputs()[0].name: pcm[None]})[0]) / 10 + 2
    ours = melspectrogram(pcm)
    assert ours.shape == (97, 32)  # one frame per 10 ms that a whole 512-sample frame fits
    np.testing.assert_allclose(ours, theirs[: ours.shape[0]], atol=1e-4)


@needs_features
def test_an_openwakeword_model_needs_no_sidecar(tmp_path):
    ear = WakeWord(tiny_oww_model(tmp_path / "hey_tiny.onnx"), features=FEATURES)
    assert ear.kind == "openwakeword" and ear.word == "hey_tiny" and ear.streaming
    # openWakeWord's window, step and rule: 16 embeddings of 80 ms, one step over 0.5 fires
    assert (ear.window, ear.step, ear.threshold, ear.min_hits) == (31712, 1280, 0.5, 1)
    ear.threshold = 2.0
    steps = []
    for start in range(0, 16_000 * 2, ear.step):
        ear.feed(np.full(ear.step, 0.01, dtype=np.float32))
        steps.append(ear.probability)
    assert steps[:5] == [0.0] * 5 and 0.0 < steps[5] < 1.0  # openWakeWord's first five steps say nothing


@needs_features
def test_an_openwakeword_model_scores_the_same_streamed_or_in_windows(tmp_path):
    path = tiny_oww_model(tmp_path / "tiny.onnx")
    stream, window = WakeWord(path, features=FEATURES), WakeWord(path, features=FEATURES, streaming=False)
    assert stream.streaming and not window.streaming
    audio = (np.random.default_rng(5).standard_normal(16_000 * 3) * 0.05).astype(np.float32)
    for ear in (stream, window):
        ear.threshold = 2.0
        ear.feed(audio)
    assert stream.probability == pytest.approx(window.probability, abs=1e-5)


def test_the_embedding_model_is_found_where_it_is_said_to_be(tmp_path):
    from wakemeup_runtime._oww import embedding_model

    (tmp_path / "embedding_model.onnx").write_bytes(b"model")
    assert embedding_model(tmp_path) == embedding_model(tmp_path / "embedding_model.onnx")
    with pytest.raises(FileNotFoundError, match="no embedding_model.onnx"):
        embedding_model(tmp_path / "elsewhere")


def test_tflite_models_point_to_the_onnx(tmp_path):
    model = tmp_path / "hey_jarvis.tflite"
    model.write_bytes(b"tflite")
    with pytest.raises(ValueError, match="use that"):
        WakeWord(model)


@needs_features
@pytest.mark.skipif(not OWW_HEAD.is_file(), reason="no openWakeWord head of the exported model")
def test_the_models_head_in_openwakewords_layout_hears_what_the_model_hears():
    """Features computed here, in front of the bare head, equal the ones inside the whole model."""
    positives = sorted(Path("H:/wakeword/training_data/positive").glob("kosmo_plain_openai_sage_*.wav"))[:2]
    if not positives:
        pytest.skip("training audio not mounted")
    from wakemeup_runtime import Resampler

    rng = np.random.default_rng(0)
    parts = []
    for path in positives:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            clip = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        parts += [(rng.standard_normal(32_000) * 0.003).astype(np.float32), Resampler(rate, 16_000).process(clip)]
    audio = np.concatenate([*parts, np.zeros(32_000, dtype=np.float32)])

    native, oww = WakeWord(MODEL), WakeWord(OWW_HEAD, features=FEATURES)
    assert native.kind == "wakemeup" and oww.kind == "openwakeword"
    scores = {"native": [], "oww": []}
    for ear in (native, oww):
        ear.threshold = 2.0
    for start in range(0, audio.size - 1280 + 1, 1280):
        for name, ear in (("native", native), ("oww", oww)):
            ear.feed(audio[start : start + 1280])
            scores[name].append(ear.probability)
    native_scores, oww_scores = (np.array(scores[k][30:]) for k in ("native", "oww"))  # both warmed up
    assert native_scores.max() > 0.9  # the word is in there
    np.testing.assert_allclose(oww_scores, native_scores, atol=1e-4)


# -- the meter --------------------------------------------------------------


def test_amplitude_db_scale():
    from wakemeup_runtime import amplitude_db
    from wakemeup_runtime.meter import DB_FLOOR

    assert amplitude_db(np.zeros(100)) == DB_FLOOR
    assert amplitude_db(np.array([], dtype=np.float32)) == DB_FLOOR
    assert abs(amplitude_db(np.full(100, 1.0)) - 0.0) < 1e-6
    assert abs(amplitude_db(np.full(100, 0.5)) + 6.02) < 0.05


def test_meter_bars_track_level_and_score():
    from wakemeup_runtime import LevelMeter
    from wakemeup_runtime.meter import DB_FLOOR

    meter = LevelMeter(threshold=0.9, width=10, color=False)
    quiet = meter.render(DB_FLOOR, 0.0)
    loud = meter.render(0.0, 1.0)
    assert quiet.count(meter._full) == 0
    assert loud.count(meter._full) == 20  # both bars full
    assert "wake 1.00" in loud


def test_meter_warns_about_a_dead_microphone(monkeypatch):
    from wakemeup_runtime import LevelMeter
    from wakemeup_runtime.meter import DB_FLOOR

    clock = [1000.0]
    monkeypatch.setattr("wakemeup_runtime.meter.time.time", lambda: clock[0])
    meter = LevelMeter(threshold=0.5, color=False)
    meter._silent_since = clock[0]
    assert "no signal" not in meter.render(DB_FLOOR, 0.0)
    clock[0] += 10
    assert "no signal" in meter.render(DB_FLOOR, 0.0)
    assert "no signal" not in meter.render(-20.0, 0.0)  # signal clears it


def test_meter_counts_detections(capsys):
    from wakemeup_runtime import LevelMeter

    meter = LevelMeter(threshold=0.5, color=False)
    meter.log_detection("kosmo", 0.97)
    out = capsys.readouterr().out
    assert "KOSMO" in out and "0.97" in out
    assert "1 hit(s)" in meter.render(-20.0, 0.1)


# -- opening a microphone ---------------------------------------------------


class FakeStream:
    def __init__(self, samplerate, channels, blocksize):
        self.samplerate, self.channels, self.blocksize = samplerate, channels, blocksize


class FakeSounddevice:
    """A card that accepts only the rates and channel counts it was given."""

    def __init__(self, rates, channels=(1,), default=44_100, max_channels=2):
        self.rates, self.channels = set(rates), set(channels)
        self.info = {
            "name": "fake mic",
            "default_samplerate": default,
            "max_input_channels": max_channels,
        }
        self.attempts = []

    def query_devices(self, device=None, kind=None):
        return self.info

    def InputStream(self, samplerate, channels, dtype, blocksize, device, callback):
        self.attempts.append((samplerate, channels))
        if samplerate not in self.rates or channels not in self.channels:
            raise RuntimeError("Error opening InputStream: Invalid sample rate")
        return FakeStream(samplerate, channels, blocksize)


@pytest.fixture
def fake_sd(monkeypatch):
    def install(card):
        monkeypatch.setitem(__import__("sys").modules, "sounddevice", card)
        return card

    return install


def test_prefers_the_rate_it_was_asked_for(fake_sd):
    from wakemeup_runtime import open_input_stream

    card = fake_sd(FakeSounddevice([16_000, 48_000]))
    stream, rate = open_input_stream(preferred=16_000)
    assert rate == 16_000 and stream.blocksize == 1600
    assert card.attempts == [(16_000, 1)]


def test_falls_back_when_the_card_refuses_16k(fake_sd):
    """The whole reason this function exists: most cards do refuse it."""
    from wakemeup_runtime import open_input_stream

    card = fake_sd(FakeSounddevice([48_000]))
    stream, rate = open_input_stream(preferred=16_000)
    assert rate == 48_000
    assert stream.blocksize == 4800  # 100 ms at the rate actually granted
    assert (16_000, 1) in card.attempts  # it did try


def test_falls_back_to_more_channels(fake_sd):
    from wakemeup_runtime import open_input_stream

    fake_sd(FakeSounddevice([44_100], channels=(2,)))
    stream, rate = open_input_stream(preferred=16_000)
    assert rate == 44_100 and stream.channels == 2


def test_an_unopenable_device_lists_what_was_tried(fake_sd):
    from wakemeup_runtime import open_input_stream

    card = fake_sd(FakeSounddevice([]))
    with pytest.raises(RuntimeError, match="could not open"):
        open_input_stream(preferred=16_000)
    assert len(card.attempts) >= 6
