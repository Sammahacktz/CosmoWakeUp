"""A terminal meter, so you can see what the microphone is giving the model.

Almost everything that goes wrong with a wakeword goes wrong before the
model: the wrong input device, a muted one, gain so low the word is in the
noise, or a rate that was never converted. None of those raise -- they just
mean the word is never heard, which is indistinguishable from a bad model
until you look at the audio.

So this draws one self-updating line with the input level in dBFS beside the
current score::

    mic  -22.0 dB [################......]  wake 0.03 [#.....................]

Pure stdlib and numpy, like the rest of the runtime. Used by
``python -m wakemeup_runtime <model> --mic``; usable on its own if you are
building something similar.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time

import numpy as np

#: Quieter than this is drawn as an empty bar.
DB_FLOOR = -80.0
#: Below this the input counts as no signal at all.
SILENCE_DB = -70.0
#: Seconds of that before the meter says so.
SILENCE_WARN_AFTER = 3.0

BLOCK_FULL = "█"
BLOCK_EMPTY = "░"


def amplitude_db(samples: np.ndarray) -> float:
    """RMS of a block in dBFS, floored at :data:`DB_FLOOR`."""
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not array.size:
        return DB_FLOOR
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    return max(DB_FLOOR, 20.0 * math.log10(rms)) if rms > 0 else DB_FLOOR


def _enable_ansi() -> bool:
    """Switch on VT sequences on a Windows console. True if colour will work."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _unicode_ok() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        (BLOCK_FULL + BLOCK_EMPTY).encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class LevelMeter:
    """One line of terminal output, redrawn in place.

    ``update()`` per block; ``log_detection()`` when the word is heard, which
    scrolls a permanent line above the meter so a session leaves a record.
    """

    def __init__(self, threshold: float, width: int = 22, color: bool | None = None) -> None:
        self.threshold = threshold
        self.width = width
        self.detections = 0
        self.last_detection: str | None = None
        self._silent_since: float | None = time.time()
        self.color = _enable_ansi() if color is None else color
        self._full, self._empty = (
            (BLOCK_FULL, BLOCK_EMPTY) if _unicode_ok() else ("#", ".")
        )
        self._last_len = 0

    # -- drawing ----------------------------------------------------------

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _level_color(self, db: float) -> str:
        if db > -3.0:
            return "31"  # red: close to clipping
        if db > -18.0:
            return "32"  # green: a healthy level
        return "33"  # yellow: quiet

    def _bar(self, fraction: float) -> str:
        filled = max(0, min(self.width, int(round(fraction * self.width))))
        return self._full * filled + self._empty * (self.width - filled)

    def render(self, level_db: float, probability: float) -> str:
        now = time.time()
        if level_db > SILENCE_DB:
            self._silent_since = None
        elif self._silent_since is None:
            self._silent_since = now

        level = self._paint(self._bar((level_db - DB_FLOOR) / -DB_FLOOR), self._level_color(level_db))
        score = self._paint(
            self._bar(probability), "32" if probability >= self.threshold else "36"
        )
        parts = [f"mic {level_db:6.1f} dB [{level}]", f"wake {probability:4.2f} [{score}]"]
        if self.detections:
            parts.append(
                self._paint(f"{self.detections} hit(s) last {self.last_detection}", "32")
            )
        if self._silent_since is not None and now - self._silent_since > SILENCE_WARN_AFTER:
            parts.append(
                self._paint("no signal -- check the input device", "33")
            )
        return "  ".join(parts)

    def update(self, level_db: float, probability: float) -> None:
        line = self.render(level_db, probability)
        columns = shutil.get_terminal_size((120, 24)).columns
        pad = max(0, self._last_len - len(line))
        self._last_len = len(line)
        sys.stdout.write("\r" + line[: columns - 1] + " " * pad)
        sys.stdout.flush()

    def log_detection(self, word: str, probability: float) -> None:
        """Scroll a permanent line above the meter."""
        self.detections += 1
        self.last_detection = time.strftime("%H:%M:%S")
        sys.stdout.write("\r" + " " * self._last_len + "\r")
        message = f"[{self.last_detection}] {word.upper()}  p={probability:.2f}"
        sys.stdout.write(self._paint(message, "1;32") + "\n")
        sys.stdout.flush()
        self._last_len = 0

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
