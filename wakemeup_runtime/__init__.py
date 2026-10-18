"""Run a WakeMeUp wakeword model. Two dependencies, one class.

    pip install "git+https://github.com/Sammahacktz/CosmoWakeUp.git"

    from wakemeup_runtime import WakeWord

    ear = WakeWord("Kosmo.onnx")          # Kosmo.json must sit beside it
    if ear.feed(block, rate=48_000):
        print("heard", ear.word)

Training happens elsewhere; nothing of it is needed here. What you download
is three files -- the ONNX graph, its streaming twin and the JSON that says
how to feed them -- and this reads them.
"""

from ._resample import Resampler
from .detector import SIDECAR_FORMAT, Heard, WakeWord, input_devices, open_input_stream
from .meter import LevelMeter, amplitude_db

__all__ = [
    "Heard",
    "LevelMeter",
    "Resampler",
    "SIDECAR_FORMAT",
    "WakeWord",
    "amplitude_db",
    "input_devices",
    "open_input_stream",
    "__version__",
]
__version__ = "0.1.0"
