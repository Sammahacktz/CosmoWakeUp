"""Rate conversion in numpy, so the runtime needs no signal-processing library.

A sound card gives you 44.1 or 48 kHz; the model was trained at 16 kHz and
will quietly fail on anything else -- not with an error, but by never firing,
because a voice arriving three times too fast does not look like speech.

Downsampling is the direction that needs a filter. Throwing away two samples
in three folds everything above 8 kHz back down into the speech band, and
what folds back is mostly consonants: an "s" reappears as a rumble. So this
is a proper polyphase resampler (windowed-sinc lowpass, upsample by ``L``,
decimate by ``M``) rather than an interpolation, and it keeps its filter
state between blocks so a stream of them joins without clicks.

``scipy.signal.resample_poly`` and ``soxr`` both do this better and faster.
Neither is here on purpose: this file is about a hundred lines of numpy, and
the point of the runtime is that a Raspberry Pi can ``pip install`` it in a
few seconds.
"""

from __future__ import annotations

from math import gcd

import numpy as np


def _lowpass(taps: int, cutoff: float, gain: float) -> np.ndarray:
    """A windowed-sinc lowpass, ``cutoff`` in cycles per sample."""
    n = np.arange(taps) - (taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    return (kernel * np.blackman(taps) * gain).astype(np.float32)


class Resampler:
    """Stateful rate conversion for a continuous stream.

    Feed it whatever blocks the device produces; it returns the same audio at
    ``target_rate``. Block lengths need not divide anything -- what does not
    make a whole output sample is carried to the next call.

    One instance per stream. Rebuilding it per block would restart the filter
    cold every time, which is the clicking this exists to avoid.
    """

    def __init__(self, source_rate: int, target_rate: int, quality: int = 12) -> None:
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self.active = self.source_rate != self.target_rate

        if not self.active:
            return

        divisor = gcd(self.source_rate, self.target_rate)
        #: Upsample by L, then keep one sample in M. For 48k -> 16k that is
        #: 1 and 3; for 44.1k -> 16k, 160 and 441.
        self.up = self.target_rate // divisor
        self.down = self.source_rate // divisor

        # Cutoff is set by whichever side is narrower: the new Nyquist when
        # downsampling, the old one when upsampling. Measured against the
        # upsampled rate, which is where the filter actually runs.
        half_band = 0.5 / max(self.up, self.down)
        taps = 2 * quality * max(self.up, self.down) + 1
        # Upsampling by inserting zeros divides the amplitude by L; the
        # filter puts it back.
        self._taps = _lowpass(taps, half_band * 0.95, gain=float(self.up))

        # The filter reaches this far back into the input.
        self._reach = int(np.ceil(len(self._taps) / self.up))
        self._history = np.zeros(self._reach, dtype=np.float32)

        # Where the next output sits on the upsampled timeline, and how much
        # input has been consumed -- together they are what stops rounding
        # from accumulating into a drift of a few samples per second.
        self._next = 0
        self._consumed = 0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self.process(block)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Convert one block. Returns float32, possibly empty."""
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        if not self.active:
            return samples
        if not samples.size:
            return np.zeros(0, dtype=np.float32)

        padded = np.concatenate([self._history, samples])
        # Absolute index of padded[0] in the whole stream of inputs.
        base = self._consumed - self._reach

        out = []
        taps, up, down = self._taps, self.up, self.down
        while True:
            # Which input sample this output sits on, and which phase of the
            # filter that lands in.
            source_index, phase = divmod(self._next, up)
            # The newest input this output needs. Stop when it is not here yet.
            if source_index >= base + len(padded):
                break
            phase_taps = taps[phase::up]
            start = source_index - len(phase_taps) + 1
            local = start - base
            if local < 0:
                break  # would reach behind the history we kept
            window = padded[local : local + len(phase_taps)]
            if window.size < len(phase_taps):
                break
            out.append(float(np.dot(window, phase_taps[::-1])))
            self._next += down

        self._consumed += samples.size
        self._history = padded[-self._reach :] if padded.size >= self._reach else padded
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def reset(self) -> None:
        """Forget the stream. For a gap in capture, where carrying the old
        filter state across would splice two unrelated moments together."""
        if self.active:
            self._history = np.zeros(self._reach, dtype=np.float32)
            self._next = 0
            self._consumed = 0
