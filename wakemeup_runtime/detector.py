"""Load a trained wakeword and listen for it.

One class, :class:`WakeWord`. Point it at the ``.onnx`` file you downloaded
and feed it microphone audio::

    from wakemeup_runtime import WakeWord

    ear = WakeWord("Kosmo.onnx")
    while True:
        block = microphone.read()               # any block size, any rate
        if ear.feed(block, rate=48_000):
            print(f"heard {ear.word}")

Everything the model needs to be used correctly -- its sample rate, how long
a window it expects, the threshold the training run settled on -- travels in
the ``.json`` file exported beside it, so the two are downloaded and kept
together and there is nothing to configure by hand.

A model trained with openWakeWord works the same way: point ``WakeWord`` at
its ``.onnx``. It has no ``.json``; the features in front of it are computed
here (see :mod:`._oww`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from . import _oww
from ._resample import Resampler
from .meter import amplitude_db

logger = logging.getLogger(__name__)

#: What ``wakemeup export`` writes into the sidecar. Checked so that a JSON
#: file that happens to sit next to the model is not silently taken for one.
SIDECAR_FORMAT = "wakemeup.onnx.v1"

#: Rates to try on a microphone, after the one that was asked for and the
#: device's own default. Nearly every card offers one of these.
FALLBACK_RATES = (48_000, 44_100, 32_000, 22_050, 16_000)


@dataclass(frozen=True)
class Heard:
    """One block's worth of what the microphone gave us.

    ``level_db`` is the input level in dBFS, which is what tells a muted or
    badly gained microphone apart from a model that is not working.
    """

    level_db: float
    probability: float
    detected: bool
    rate: int


def _sounddevice():
    """Import sounddevice, and say what to install when it will not load.

    Two different failures, and on Linux the second is the usual one:
    ``sounddevice`` is a binding, and the PortAudio library it binds to is a
    system package rather than a wheel. Without it the import raises an
    ``OSError`` from deep inside the module, which says nothing about how to
    fix it.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            'the microphone needs sounddevice: pip install "wakemeup-runtime[mic]"'
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"sounddevice is installed but PortAudio is not ({exc}). It is a system "
            f"library, not a wheel:\n"
            f"    Debian/Ubuntu/Raspberry Pi OS:  sudo apt install libportaudio2\n"
            f"    Fedora:                         sudo dnf install portaudio\n"
            f"    macOS:                          brew install portaudio\n"
            f"Reading audio in yourself and calling feed() needs none of this."
        ) from exc
    return sd


def open_input_stream(
    *,
    device: int | str | None = None,
    preferred: int = 16_000,
    blocksize: int | None = None,
    step_ms: int = 100,
    callback=None,
):
    """Open a microphone at a rate it accepts. Returns ``(stream, rate)``.

    **Sound cards very commonly refuse 16 kHz.** WASAPI devices usually offer
    only their native 48 kHz, and the failure is a bare "Invalid sample rate"
    from PortAudio at the moment of opening -- so a caller that simply asks
    for the model's rate works on some machines and not others.

    Each candidate is tried by *actually opening* the stream, because probing
    with ``check_input_settings`` is not reliable: MME devices report support
    for 16 kHz and then fail to open at it.

    Whatever rate comes back is the caller's to hand to :meth:`WakeWord.feed`,
    which converts it.
    """
    sd = _sounddevice()
    info = sd.query_devices(device if device is not None else None, "input")
    rates: list[int] = []
    for rate in (preferred, int(info.get("default_samplerate") or 0), *FALLBACK_RATES):
        if rate > 0 and rate not in rates:
            rates.append(rate)

    channel_options = [1]
    if int(info.get("max_input_channels") or 1) > 1:
        channel_options.append(int(info["max_input_channels"]))

    errors: list[str] = []
    for channels in channel_options:
        for rate in rates:
            block = int(blocksize or rate * step_ms / 1000)
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=block,
                    device=device,
                    callback=callback,
                )
            except Exception as exc:
                errors.append(f"{rate} Hz / {channels} ch: {exc}")
                continue
            if rate != preferred:
                logger.info("%s opened at %d Hz (asked for %d)", info["name"], rate, preferred)
            return stream, rate

    detail = "\n  ".join(errors)
    raise RuntimeError(
        f"could not open {info.get('name')!r} for recording. Tried:\n  {detail}"
    )


def _onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("wakemeup-runtime needs onnxruntime: pip install onnxruntime") from exc
    return ort


def input_devices() -> list[tuple[int, str, str, int]]:
    """Every microphone the system offers: ``(index, name, api, channels)``."""
    sd = _sounddevice()
    return [
        (i, d["name"], sd.query_hostapis(d["hostapi"])["name"], d["max_input_channels"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


class WakeWord:
    """A trained wakeword, and the sliding window that listens for it.

    ``model`` is the path to the ``.onnx`` file; the ``.json`` of the same
    name must sit beside it.

    The detector holds the last :attr:`clip_seconds` of audio and scores it
    every :attr:`step_ms`. A hit needs ``min_hits`` consecutive windows over
    the threshold -- a spoken word covers several, where a door slam covers
    one -- and is followed by ``refractory`` seconds of not listening, since
    the word stays in the window for as long as the window is wide and would
    otherwise be counted once per step on its way out.

    Stateful, and not thread-safe: one instance per audio stream. Cheap to
    run -- a few tens of thousands of parameters, once per step -- but not
    free, so feed it from a worker rather than from an audio callback.

    Models over openWakeWord's features ship a second graph beside the
    ``.onnx``, named in the sidecar under ``"streaming"``: the same model one
    step at a time, carrying its state, at about a sixteenth of the work of
    scoring the whole window. It is used whenever the step is the one it was
    built for (``streaming=None``); ``True`` insists on it, ``False`` scores
    whole windows. Both give the same scores.

    ``model`` may also be an openWakeWord model (the ``.onnx`` its training
    writes, recognised by its input of ``[batch, n, 96]`` embeddings). It
    needs no sidecar: the mel spectrogram and the speech embedding in front of
    it are computed here, the embedding model (``embedding_model.onnx``,
    Apache 2.0) fetched once into ``~/.cache/wakemeup/oww`` or read from
    ``features`` (the file or its folder). It streams in steps of 80 ms and
    fires, as openWakeWord does, when one step scores ``threshold`` (0.5)
    or more; :attr:`kind` says which sort of model was loaded.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        threshold: float | None = None,
        min_hits: int | None = None,
        refractory: float | None = None,
        step_ms: int | None = None,
        threads: int = 1,
        providers: Sequence[str] | None = None,
        streaming: bool | None = None,
        features: str | Path | None = None,
    ) -> None:
        self.path = Path(model).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"no wakeword model at {self.path}")

        self._threads, self._providers = threads, providers
        self._oww: _oww.OpenWakeWordModel | None = None  # an openWakeWord model with its feature stages
        meta = self._load_metadata(features)
        #: ``"wakemeup"`` or ``"openwakeword"``: which sort of model this is.
        self.kind: str = "openwakeword" if self._oww is not None else "wakemeup"
        audio = meta["audio"]

        #: The word this model was trained on. For logs and for the user --
        #: nothing in here matches on it, the network is the only thing that
        #: knows what it sounds like.
        self.word: str = meta.get("wakeword") or self.path.stem
        self.sample_rate: int = int(audio["sample_rate"])
        self.clip_seconds: float = float(audio["clip_seconds"])
        #: Samples in one window, at :attr:`sample_rate`.
        self.window: int = int(round(self.clip_seconds * self.sample_rate))
        #: Probability above which a window counts. Defaults to the value the
        #: training run calibrated for its target rate of false wake-ups.
        self.threshold: float = float(
            threshold if threshold is not None else meta.get("threshold", 0.5)
        )
        self.min_hits, self.refractory, self.step_ms = self._gate(
            meta, min_hits, refractory, step_ms
        )
        self.step = max(1, int(self.sample_rate * self.step_ms / 1000))
        self.metadata: dict[str, Any] = meta

        self._session = None  # the window graph, opened when first needed
        if self._oww is not None:
            self._stream = self._oww_stream(streaming)
        else:
            self._stream = self._open_stream(meta, streaming)
        #: True when the model runs one step at a time on its streaming graph.
        self.streaming: bool = self._stream is not None
        if not self.streaming and self._oww is None:
            self._window_session()

        self._resamplers: dict[int, Resampler] = {}
        self._buffer = np.zeros(self.window, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._hits = 0
        self._muted = 0  # samples of refractory left to run down
        self._position = 0  # samples consumed, at the model's rate
        self._fired_at = 0  # where the last wake-up fired, at the model's rate

        #: The score of the most recent window, 0..1. Useful for a level meter
        #: while tuning :attr:`threshold`.
        self.probability = 0.0

        logger.info(
            "wakeword %r: %.2fs window @ %d Hz, threshold %.2f%s",
            self.word,
            self.clip_seconds,
            self.sample_rate,
            self.threshold,
            ", streaming" if self.streaming else "",
        )

    # -- construction helpers ---------------------------------------------

    def _load_metadata(self, features: str | Path | None) -> dict:
        """The sidecar of a WakeMeUp model; for an openWakeWord model, what its graph says.

        A ``.json`` beside the model makes it a WakeMeUp model. Without one,
        the graph decides: an input of ``[batch, n, 96]`` embeddings is
        openWakeWord's layout; anything else is a WakeMeUp model whose sidecar
        is missing, and is reported as that.
        """
        if self.path.suffix.lower() == ".tflite":
            raise ValueError(
                f"{self.path.name}: .tflite models are not supported. openWakeWord's training "
                f"writes an .onnx beside it; use that."
            )
        if not self.path.with_suffix(".json").is_file():
            _onnxruntime()
            try:
                session = self._open_session(self.path)
            except Exception:  # noqa: BLE001 - not a graph onnxruntime can read
                session = None
            if session is not None and _oww.is_openwakeword(session):
                self._oww = _oww.OpenWakeWordModel(session, self._open_session, features)
                return {
                    "format": "openwakeword",
                    "wakeword": self.path.stem,
                    "threshold": 0.5,
                    "audio": {"sample_rate": _oww.SAMPLE_RATE, "clip_seconds": self._oww.window / _oww.SAMPLE_RATE},
                    # openWakeWord's own rule: one step over the threshold is a detection
                    "detect": {"step_seconds": _oww.STEP / _oww.SAMPLE_RATE, "min_hits": 1, "refractory_seconds": 2.0},
                    "model": {"embeddings": self._oww.n_embeddings},
                    "features": str(self._oww.features_path),
                }
        return self._load_sidecar(self.path)

    def _oww_stream(self, streaming: bool | None):
        """The openWakeWord model step by step, when the step is its 80 ms; else None (whole windows)."""
        if streaming is False:
            return None
        if self.step != _oww.STEP:
            if streaming:
                raise ValueError(f"openWakeWord models step {_oww.STEP} samples (80 ms); leave step_ms at its default")
            logger.info("step %d ms is not openWakeWord's 80 ms; scoring whole windows", self.step_ms)
            return None
        return self._oww

    @staticmethod
    def _load_sidecar(model: Path) -> dict:
        sidecar = model.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"{model.name} has no {sidecar.name} beside it. The two are a pair: "
                f"the sidecar carries the sample rate, the window length and the "
                f"threshold the model was trained with. Download both."
            )
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("format") != SIDECAR_FORMAT:
            raise ValueError(
                f"{sidecar.name} is not a wakemeup sidecar (format={meta.get('format')!r})"
            )
        return meta

    @staticmethod
    def _gate(
        meta: dict, min_hits: int | None, refractory: float | None, step_ms: int | None
    ) -> tuple[int, float, int]:
        """``(min_hits, refractory, step_ms)``: what was asked for, else the sidecar's.

        The threshold was calibrated behind one particular gate, and the same
        threshold behind a different one gives a different rate of false
        wake-ups -- so by default the gate is the one it was measured with.
        Sidecars from before that was recorded fall back to the old defaults.
        """
        gate = meta.get("detect") or {}
        if min_hits is None:
            min_hits = gate.get("min_hits", 2)
        if refractory is None:
            refractory = gate.get("refractory_seconds", 2.0)
        if step_ms is None:
            step_ms = round(1000 * gate.get("step_seconds", 0.1))
        return max(1, int(min_hits)), float(refractory), int(step_ms)

    def _open_session(self, path: Path):
        ort = _onnxruntime()

        # One thread by default. This runs ten times a second beside a live
        # audio stream, and onnxruntime otherwise fans out across every core
        # -- which on a small board is what turns a missed deadline into
        # dropped microphone audio. The graph is tiny; it does not need help.
        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = self._threads
        return ort.InferenceSession(
            str(path), options, providers=list(self._providers or ["CPUExecutionProvider"])
        )

    def _window_session(self):
        """The graph that scores one whole window, opened on first use."""
        if self._session is None:
            self._session = self._open_session(self.path)
            self._input = self._session.get_inputs()[0].name
        return self._session

    def _open_stream(self, meta: dict, streaming: bool | None):
        """The step graph, when the model has one and the step fits it; else None."""
        spec = meta.get("streaming")
        if streaming is False:
            return None
        if not spec:
            if streaming:
                raise ValueError(f"{self.path.name} has no streaming graph; export it again")
            return None
        if self.step != int(spec["step_samples"]):
            if streaming:
                raise ValueError(
                    f"the streaming graph steps {spec['step_samples']} samples, "
                    f"not {self.step}; leave step_ms at its default"
                )
            logger.info("step %d ms is not the streaming graph's; scoring whole windows", self.step_ms)
            return None
        path = self.path.with_name(spec["model"])
        if not path.is_file():
            if streaming:
                raise FileNotFoundError(f"the streaming graph {path} is missing")
            logger.warning(
                "%s is missing beside %s; scoring whole windows instead, which costs "
                "about 16 times the CPU -- too much for a Raspberry Pi",
                path.name,
                self.path.name,
            )
            return None
        session = self._open_session(path)
        # every state input "x" comes back as output "x_out"; the first output is the logit
        outputs = [o.name for o in session.get_outputs()]
        self._state_order = [n[: -len("_out")] for n in outputs[1:]]
        state = {
            i.name: np.zeros([1, *i.shape[1:]], dtype=np.float32)
            for i in session.get_inputs()
            if i.name != "audio"
        }
        zeros = np.zeros((1, self.step), dtype=np.float32)
        for _ in range(int(spec.get("warmup_steps", 0))):
            state = dict(zip(self._state_order, session.run(None, {"audio": zeros, **state})[1:]))
        # what a stretch of silence leaves, as a fresh zero-filled window holds
        self._silence = state
        self._state = {k: v.copy() for k, v in state.items()}
        return session

    def _stream_step(self, block: np.ndarray) -> float:
        """Advance the streaming graph by one step; the probability of the window ending here."""
        if self._oww is not None:
            return self._oww.step(block)
        out = self._stream.run(None, {"audio": block[None, :].astype(np.float32), **self._state})
        self._state = dict(zip(self._state_order, out[1:]))
        return float(1.0 / (1.0 + np.exp(-float(np.reshape(out[0], -1)[0]))))

    # -- streaming ---------------------------------------------------------

    def feed(self, audio: np.ndarray | bytes, rate: int | None = None) -> bool:
        """Offer a block of mono audio. True if the wakeword just finished.

        ``audio`` may be ``float32`` in -1..1, ``int16``, or raw PCM16 bytes,
        shaped ``(n,)`` or ``(n, 1)``. ``rate`` is what it was recorded at and
        defaults to the model's own; anything else is resampled, statefully,
        so blocks join cleanly.

        Blocks may be any length: what does not fill a step is kept for the
        next call.
        """
        samples = self._to_mono_float(audio)
        source_rate = int(rate or self.sample_rate)
        if source_rate != self.sample_rate:
            resampler = self._resamplers.get(source_rate)
            if resampler is None:
                resampler = self._resamplers[source_rate] = Resampler(
                    source_rate, self.sample_rate
                )
                logger.debug("resampling %d Hz -> %d Hz", source_rate, self.sample_rate)
            samples = resampler.process(samples)

        if samples.size:
            self._pending = np.concatenate([self._pending, samples])

        heard = False
        while self._pending.size >= self.step:
            block, self._pending = self._pending[: self.step], self._pending[self.step :]
            self._position += self.step
            if self._stream is not None:
                # the streaming graph has to see every step to keep its state
                probability = self._stream_step(block)
            else:
                self._buffer = np.concatenate([self._buffer[self.step :], block])

            if self._muted > 0:
                # Inside the refractory period. The window is still kept up to
                # date -- it has to be, or it would be stale when listening
                # resumes -- but nothing is inferred from it.
                self._muted -= self.step
                continue

            self.probability = probability if self._stream is not None else self.score(self._buffer)
            if self.probability >= self.threshold:
                self._hits += 1
                if self._hits >= self.min_hits:
                    self._hits = 0
                    self._muted = int(self.refractory * self.sample_rate)
                    self._fired_at = self._position
                    heard = True
            else:
                self._hits = 0
        return heard

    def score(self, window: np.ndarray) -> float:
        """Probability that one window of exactly :attr:`window` samples is the word."""
        if self._oww is not None:
            return self._oww.window_score(window)
        logit = self._window_session().run(None, {self._input: window[None, :].astype(np.float32)})[0]
        return float(1.0 / (1.0 + np.exp(-float(np.reshape(logit, -1)[0]))))

    def reset(self) -> None:
        """Forget the window, the partial hit and the refractory countdown.

        Worth doing whenever the microphone has been hearing something that
        must not become a detection -- your own speaker, most of all -- and
        after any gap in capture, where the window would otherwise splice two
        unrelated moments together.
        """
        self._buffer = np.zeros(self.window, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._hits = 0
        self._muted = 0
        self._position = 0
        self._fired_at = 0
        self.probability = 0.0
        if self._oww is not None:
            self._oww.reset()
        elif self._stream is not None:
            self._state = {k: v.copy() for k, v in self._silence.items()}
        for resampler in self._resamplers.values():
            resampler.reset()

    # -- offline -----------------------------------------------------------

    def scan(self, audio: np.ndarray | bytes, rate: int | None = None) -> list[float]:
        """Run over a whole recording. Returns the time of each detection.

        Times are in seconds from the start, measured at the end of the
        window that fired -- which is roughly when the word finished being
        said. Resets first, so a scan is independent of anything fed before,
        and ends with a window of silence, so a word at the very end of the
        recording is heard in full, as `wakemeup detect` does.
        """
        self.reset()
        times: list[float] = []
        samples = self._to_mono_float(audio)
        source_rate = int(rate or self.sample_rate)
        chunk = max(self.step, source_rate // 10)
        for start in range(0, samples.size, chunk):
            if self.feed(samples[start : start + chunk], source_rate):
                times.append(self._fired_at / self.sample_rate)
        if self.feed(np.zeros(self.window, dtype=np.float32)):
            times.append(self._fired_at / self.sample_rate)
        return times

    # -- microphone --------------------------------------------------------

    def listen(
        self,
        on_detect: Callable[[float], None] | None = None,
        *,
        device: int | str | None = None,
        rate: int | None = None,
        blocksize: int | None = None,
    ) -> Iterator[Heard]:
        """Yield one :class:`Heard` per block from the microphone, forever.

        Needs the ``mic`` extra (``pip install "wakemeup-runtime[mic]"``).
        Convenience for a demo, a meter or a small script; an application
        that already owns a microphone should call :meth:`feed` instead.

        ``rate`` is only a preference. Sound cards very commonly refuse
        16 kHz, so whatever they do accept is opened and converted -- see
        :func:`open_input_stream`.
        """
        import queue

        blocks: queue.Queue = queue.Queue()
        stream, device_rate = open_input_stream(
            device=device,
            preferred=int(rate or self.sample_rate),
            blocksize=blocksize,
            step_ms=self.step_ms,
            callback=lambda indata, frames, t, s: blocks.put(indata[:, 0].copy()),
        )

        with stream:
            while True:
                block = blocks.get()
                detected = self.feed(block, device_rate)
                if detected and on_detect is not None:
                    on_detect(self.probability)
                yield Heard(
                    level_db=amplitude_db(block),
                    probability=self.probability,
                    detected=detected,
                    rate=device_rate,
                )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_mono_float(audio: np.ndarray | bytes) -> np.ndarray:
        if isinstance(audio, (bytes, bytearray, memoryview)):
            return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        array = np.asarray(audio)
        if array.ndim > 1:
            array = array[:, 0]
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float32) / 32768.0
        return array.astype(np.float32, copy=False).reshape(-1)

    def __repr__(self) -> str:
        return (
            f"WakeWord({self.word!r}, {self.clip_seconds}s @ {self.sample_rate}Hz, "
            f"threshold={self.threshold:.2f}{', streaming' if self.streaming else ''}"
            f"{', openWakeWord' if self._oww is not None else ''})"
        )
