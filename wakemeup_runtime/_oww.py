"""openWakeWord models: their features are computed here, the model scores them.

A model from openWakeWord's training (its ``.onnx``) is only the last stage:
it takes the 96-dimensional speech embeddings of the last ``n`` steps of
80 ms, ``[batch, n, 96]``, and returns a probability.  The two stages in front
of it are the ones every WakeMeUp model carries inside its own graph; for an
openWakeWord model they run here:

* the mel spectrogram, in numpy, as openWakeWord computes it: 32 bands from
  60 to 3800 Hz, 25 ms frames every 10 ms, in dB, scaled by ``x / 10 + 2``.
  The 80 dB floor is taken per frame, as WakeMeUp does, so a frame's features
  do not depend on how the audio was cut into blocks; openWakeWord takes it
  over each call, which differs only where a frame's bands span more than
  80 dB, as digital silence does and real audio does not;
* Google's speech embedding, ``embedding_model.onnx`` of openWakeWord's
  v0.5.1 release (Apache 2.0), fetched once and checked against its SHA-256,
  or read from where the caller says it is.

It streams the way openWakeWord streams: every 1280 samples, 8 new mel frames,
one embedding of the last 76 frames, and the model over the last ``n``
embeddings.  The 8 frames are the ones that end with the step (computed from
its last 1632 samples), so a streamed score is exactly the score of the
window ending there; openWakeWord reaches back 1760 samples, which puts its
frames 128 samples earlier.  It starts the way openWakeWord starts: a mel
history of ones, the embeddings of 4 s of quiet noise, and the first 5 scores 0.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16_000
N_FFT, WIN_LENGTH, HOP = 512, 400, 160
N_MELS, F_MIN, F_MAX = 32, 60.0, 3800.0
AMIN, TOP_DB = 1e-10, 80.0
INT16_SCALE = 32768.0
WINDOW_FRAMES = 76  # mel frames per embedding: 0.775 s
FRAME_STRIDE = 8  # mel frames between embeddings
STEP = FRAME_STRIDE * HOP  # samples between embeddings: 1280, 80 ms
TAIL = N_FFT - HOP  # samples a step's first frame reaches back into the step before (352)
EMBEDDING_DIM = 96
WARMUP_SECONDS = 4.0  # openWakeWord fills its embedding history with this much quiet noise
SILENT_STEPS = 5  # ... and reports 0 for this many steps

EMBEDDING_FILE = "embedding_model.onnx"
EMBEDDING_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/" + EMBEDDING_FILE
EMBEDDING_SHA256 = "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"


def window_samples(n_embeddings: int) -> int:
    """Samples of audio ``n_embeddings`` consecutive embeddings are computed from (31712 for 16)."""
    return (WINDOW_FRAMES + FRAME_STRIDE * (n_embeddings - 1) - 1) * HOP + N_FFT


# -- the mel spectrogram -----------------------------------------------------


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=np.float64)
    mel = hz / (200.0 / 3)
    log = hz >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(hz, 1e-12) / 1000.0) / (np.log(6.4) / 27.0), mel)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel - 15.0)), mel * (200.0 / 3))


def mel_filterbank() -> np.ndarray:
    """``(257, 32)``: triangles evenly spaced on the Slaney scale, each normalised to unit area."""
    freqs = np.linspace(0, SAMPLE_RATE // 2, N_FFT // 2 + 1)
    points = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    widths = np.diff(points)
    slopes = points[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    bank = np.maximum(0.0, np.minimum(down, up))
    return bank * (2.0 / (points[2 : N_MELS + 2] - points[:N_MELS]))[None, :]


def _window() -> np.ndarray:
    """A periodic 400-point Hann window, centred in the 512-point frame."""
    window = np.zeros(N_FFT)
    pad = (N_FFT - WIN_LENGTH) // 2
    window[pad : pad + WIN_LENGTH] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN_LENGTH) / WIN_LENGTH)
    return window


_WINDOW = _window()
_BANK = mel_filterbank()


def melspectrogram(pcm: np.ndarray) -> np.ndarray:
    """``(frames, 32)`` of audio in the int16 range, one frame per 160 samples that a whole 512 fit."""
    pcm = np.asarray(pcm, dtype=np.float64).reshape(-1)
    if pcm.size < N_FFT:
        return np.zeros((0, N_MELS), dtype=np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(pcm, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _BANK, AMIN))
    db = np.maximum(db, db.max(axis=1, keepdims=True) - TOP_DB)
    return (db / 10.0 + 2.0).astype(np.float32)


# -- the speech embedding ----------------------------------------------------


def embedding_model(path: str | Path | None = None, download: bool = True) -> Path:
    """``embedding_model.onnx``: ``path`` (the file, or a folder holding it), else the cache, fetched if missing.

    The cache is ``$WAKEMEUP_CACHE_DIR/oww`` or ``~/.cache/wakemeup/oww``, the
    folder the WakeMeUp training keeps it in.
    """
    if path is not None:
        path = Path(path).expanduser()
        found = path / EMBEDDING_FILE if path.is_dir() else path
        if not found.is_file():
            raise FileNotFoundError(f"no {EMBEDDING_FILE} at {path}")
        return found
    root = os.environ.get("WAKEMEUP_CACHE_DIR")
    found = (Path(root) if root else Path.home() / ".cache" / "wakemeup") / "oww" / EMBEDDING_FILE
    if not found.is_file():
        if not download:
            raise FileNotFoundError(f"no {EMBEDDING_FILE} in {found.parent}")
        found.parent.mkdir(parents=True, exist_ok=True)
        partial = found.with_suffix(".part")
        try:
            urllib.request.urlretrieve(EMBEDDING_URL, partial)
        except OSError as exc:
            raise RuntimeError(
                f"could not fetch the speech embedding openWakeWord models need ({exc}). "
                f"Download {EMBEDDING_URL} and pass its path as features=..."
            ) from exc
        if hashlib.sha256(partial.read_bytes()).hexdigest() != EMBEDDING_SHA256:
            partial.unlink()
            raise RuntimeError(f"{EMBEDDING_URL} did not have the expected SHA-256")
        partial.replace(found)
    return found


# -- the model ---------------------------------------------------------------


def is_openwakeword(session) -> bool:
    """An openWakeWord model reads ``[batch, n, 96]`` embeddings; a WakeMeUp graph reads audio."""
    shape = session.get_inputs()[0].shape
    return len(shape) == 3 and shape[-1] == EMBEDDING_DIM


class OpenWakeWordModel:
    """An openWakeWord model with its feature stages: one probability per 80 ms step."""

    def __init__(self, session, open_session: Callable, features: str | Path | None = None):
        self.session = session
        self._input = session.get_inputs()[0].name
        n = session.get_inputs()[0].shape[1]
        if not isinstance(n, int):
            raise ValueError("the model does not say how many embeddings it reads (its input dim 1 is not fixed)")
        outputs = session.get_outputs()[0].shape
        if len(outputs) > 1 and isinstance(outputs[-1], int) and outputs[-1] != 1:
            raise ValueError(f"a model with {outputs[-1]} classes; only one wakeword per model is supported")
        #: Embeddings the model reads: one per 80 ms step.
        self.n_embeddings = n
        #: Samples those embeddings are computed from: the window the model sees.
        self.window = window_samples(n)
        self.features_path = embedding_model(features)
        self._embedding = open_session(self.features_path)
        self._embedding_input = self._embedding.get_inputs()[0].name
        rng = np.random.default_rng(0)
        noise = rng.integers(-1000, 1000, int(WARMUP_SECONDS * SAMPLE_RATE)).astype(np.float32) / INT16_SCALE
        self._start = self.features(noise)[-n:]
        self.reset()

    def embed(self, mel_windows: np.ndarray) -> np.ndarray:
        """``(k, 76, 32)`` mel windows -> ``(k, 96)``."""
        out = self._embedding.run(None, {self._embedding_input: mel_windows[..., None].astype(np.float32)})[0]
        return out.reshape(-1, EMBEDDING_DIM)

    def features(self, samples: np.ndarray) -> np.ndarray:
        """``(k, 96)``: an embedding every 80 ms of a recording in -1..1, oldest first."""
        mel = melspectrogram(np.asarray(samples, dtype=np.float32) * INT16_SCALE)
        if mel.shape[0] < WINDOW_FRAMES:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        starts = range(0, mel.shape[0] - WINDOW_FRAMES + 1, FRAME_STRIDE)
        return self.embed(np.stack([mel[s : s + WINDOW_FRAMES] for s in starts]))

    def predict(self, embeddings: np.ndarray) -> float:
        """The model's probability for the last :attr:`n_embeddings` embeddings."""
        x = embeddings[-self.n_embeddings :][None].astype(np.float32)
        return float(np.reshape(self.session.run(None, {self._input: x})[0], -1)[0])

    def window_score(self, window: np.ndarray) -> float:
        """The probability for one window of exactly :attr:`window` samples."""
        return self.predict(self.features(window))

    def reset(self) -> None:
        self._tail = np.zeros(TAIL, dtype=np.float64)
        self._mel = np.ones((WINDOW_FRAMES, N_MELS), dtype=np.float32)
        self._embeddings = self._start.copy()
        self._steps = 0

    def step(self, block: np.ndarray) -> float:
        """Advance by one block of :data:`STEP` samples in -1..1; the probability now."""
        pcm = np.concatenate([self._tail, np.asarray(block, dtype=np.float64) * INT16_SCALE])
        self._tail = pcm[-TAIL:]
        self._mel = np.concatenate([self._mel, melspectrogram(pcm)])[-WINDOW_FRAMES:]
        self._embeddings = np.concatenate([self._embeddings, self.embed(self._mel[None])])[-self.n_embeddings :]
        self._steps += 1
        probability = self.predict(self._embeddings)
        return 0.0 if self._steps <= SILENT_STEPS else probability
