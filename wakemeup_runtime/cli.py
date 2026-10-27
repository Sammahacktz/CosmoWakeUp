"""Check a downloaded model without writing any code.

    python -m wakemeup_runtime Kosmo.onnx --info            # what is in it
    python -m wakemeup_runtime Kosmo.onnx recording.wav     # scan a file
    python -m wakemeup_runtime Kosmo.onnx --mic             # listen, with a meter
    python -m wakemeup_runtime Kosmo.onnx --bench           # CPU cost on this machine
    python -m wakemeup_runtime --list-devices               # which microphone?

The model may also be one trained with openWakeWord (its .onnx, no .json).
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .detector import WakeWord, input_devices
from .meter import LevelMeter


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float32 from a PCM wav, without pulling in an audio library."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav files are read here")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0
        channels = handle.getnchannels()
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, handle.getframerate()


def list_devices() -> int:
    try:
        devices = input_devices()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    for index, name, api, channels in devices:
        print(f"{index:3d}  {name[:45]:45s} {api:20s} ch={channels}")
    print("\nPass one with --device, by index or by part of its name.")
    return 0


def show_info(ear: WakeWord) -> int:
    meta = ear.metadata
    if ear.kind == "openwakeword":
        print(f"model      : openWakeWord, {meta['model']['embeddings']} embeddings in, "
              f"features computed here ({meta['features']})")
    print(f"word       : {ear.word}")
    print(f"audio      : {ear.clip_seconds}s window @ {ear.sample_rate} Hz")
    print(f"threshold  : {ear.threshold:.2f}  (min_hits {ear.min_hits}, step {ear.step_ms} ms)")
    print(f"inference  : {'streaming, one step at a time' if ear.streaming else 'the whole window every step'}")
    print(f"trained    : {meta.get('created', 'unknown')}")
    metrics = meta.get("metrics") or {}
    if metrics.get("best_f1") is not None:
        print(f"validation : F1 {metrics['best_f1']:.3f}")
    # "calibration" since the threshold is chosen for a false-wake rate;
    # older sidecars only carry the rate measured at the best-F1 threshold
    stream = meta.get("calibration") or meta.get("stream_false_alarms") or {}
    rate = stream.get("false_wakes_per_hour", stream.get("events_per_hour"))
    if rate is not None:
        print(f"false wakes: {rate:.2f}/hour on held-out audio when it was measured")
    if stream.get("recall") is not None:
        print(f"recall     : {stream['recall']:.1%} of held-out wakeword clips")
    return 0


def bench(ear: WakeWord, seconds: float = 20.0) -> int:
    """Time the model on noise, as the microphone would feed it: the CPU it takes here."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(seconds * ear.sample_rate)) * 0.05).astype(np.float32)
    ear.threshold = 2.0  # never fires, so every step is scored
    ear.feed(audio[: ear.sample_rate])  # warm up the session
    ear.reset()
    started = time.perf_counter()
    for start in range(0, audio.size, ear.step):
        ear.feed(audio[start : start + ear.step])
    elapsed = time.perf_counter() - started
    steps = audio.size // ear.step
    mode = "streaming" if ear.streaming else "whole window"
    print(f"{mode}: {1000 * elapsed / steps:.2f} ms per {ear.step_ms} ms step, "
          f"{100 * elapsed / seconds:.1f} % of one core")
    return 0


def listen(ear: WakeWord, device, rate: int | None) -> int:
    """Live meter: the input level beside the score, until Ctrl-C."""
    meter = LevelMeter(ear.threshold)
    print(f"listening for '{ear.word}' at threshold {ear.threshold:.2f} -- Ctrl-C to stop")
    try:
        for heard in ear.listen(device=device, rate=rate):
            if heard.detected:
                meter.log_detection(ear.word, heard.probability)
            meter.update(heard.level_db, heard.probability)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        meter.close()
        print(exc, file=sys.stderr)
        print("\nRun --list-devices and pick one with --device.", file=sys.stderr)
        return 1
    meter.close()
    print(f"stopped after {meter.detections} detection(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wakemeup_runtime",
        description="Try a wakeword model on a file or a microphone.",
    )
    parser.add_argument(
        "model", nargs="?", help="path to the .onnx: a WakeMeUp model (its .json beside it) or an openWakeWord model"
    )
    parser.add_argument("audio", nargs="*", help="wav files to scan")
    parser.add_argument("--mic", action="store_true", help="listen on the microphone")
    parser.add_argument("--info", action="store_true", help="print what the model expects, and exit")
    parser.add_argument("--bench", action="store_true", help="time the model on this machine, and exit")
    parser.add_argument("--no-streaming", action="store_true", help="score the whole window every step")
    parser.add_argument("--list-devices", action="store_true", help="print the microphones, and exit")
    parser.add_argument("--threshold", type=float, help="override the model's own threshold")
    parser.add_argument("--min-hits", type=int, help="consecutive windows needed (default: the model's own)")
    parser.add_argument("--rate", type=int, help="preferred microphone rate")
    parser.add_argument("--device", help="microphone index or part of its name")
    parser.add_argument(
        "--features", help="openWakeWord models: embedding_model.onnx or its folder (default: fetched once)"
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        return list_devices()
    if not args.model:
        parser.error("give a model, or --list-devices")

    ear = WakeWord(
        args.model,
        threshold=args.threshold,
        min_hits=args.min_hits,
        streaming=False if args.no_streaming else None,
        features=args.features,
    )

    if args.info:
        return show_info(ear)
    if args.bench:
        return bench(ear)

    if args.mic:
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        return listen(ear, device, args.rate)

    if not args.audio:
        parser.error("give some wav files, or --mic, or --info")

    for path in args.audio:
        samples, rate = read_wav(path)
        hits = ear.scan(samples, rate)
        print(f"{path}: {len(hits)} detection(s)" + "".join(f"\n   {t:8.2f}s" for t in hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
