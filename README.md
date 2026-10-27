# CosmoWakeUp

Run a WakeMeUp wakeword on a device: the Python package `wakemeup-runtime`.
Two dependencies, one class, no training stack. A wakeword you trained
yourself with [openWakeWord](https://github.com/dscripka/openWakeWord) runs
here too, through the same class (see [openWakeWord models](#openwakeword-models)).

```bash
pip install "git+https://github.com/Sammahacktz/CosmoWakeUp.git"
```

```python
from wakemeup_runtime import WakeWord

ear = WakeWord("Kosmo.onnx")            # Kosmo.json must sit beside it

while True:
    block = microphone.read()           # any block size, any rate
    if ear.feed(block, rate=48_000):
        print(f"heard {ear.word}")
```

That is the whole interface. `feed()` returns `True` on the block where the
word finished, and `False` the rest of the time.

## What you download

A wakeword is trained for you on the WakeMeUp server; what you get back is
**three files** with the same name:

| file | what it is |
|---|---|
| `Kosmo.onnx` | the network: raw 16 kHz audio in, one score out |
| `Kosmo.json` | its sample rate, window length and recommended threshold |
| `Kosmo.stream.onnx` | the same network one 80 ms step at a time, which streaming uses (exports from before it do without) |

Keep them together in one folder. The runtime reads the `.json` to find out
how to feed the networks, so there is nothing to configure by hand and no way
for them to drift apart. Retraining for a different word is a new set of files
and no code change: nothing here knows what the word *is*.

## Installing

`numpy` and `onnxruntime`, and that is deliberate — this is the half that runs
on a Raspberry Pi or inside a Home Assistant add-on, so torch and the training
stack stay upstream.

```bash
pip install "git+https://github.com/Sammahacktz/CosmoWakeUp.git"
pip install "wakemeup-runtime[mic] @ git+https://github.com/Sammahacktz/CosmoWakeUp.git"
```

The `mic` extra adds `sounddevice`, and is only needed for `listen()` and the
`--mic` command line. An application that already has a microphone feeds
`feed()` itself and needs nothing extra.

On Linux that extra also needs PortAudio, which is a system package rather
than a wheel:

```bash
sudo apt install libportaudio2      # Debian, Ubuntu, Raspberry Pi OS
```

**Sound cards very commonly refuse 16 kHz.** `listen()` and `open_input_stream()`
therefore try the model's rate, then the device's default, then 48/44.1/32/22.05
kHz, opening each in turn until one works, and convert whatever they get. Asking
for a rate the card rejects fails with a bare "Invalid sample rate" from
PortAudio, which is why this is done by opening rather than by asking.

## Trying a model out

```bash
python -m wakemeup_runtime Kosmo.onnx --info              # what it expects
python -m wakemeup_runtime Kosmo.onnx recording.wav       # scan a file
python -m wakemeup_runtime Kosmo.onnx --mic               # listen live
python -m wakemeup_runtime Kosmo.onnx --bench             # CPU it takes on this machine
python -m wakemeup_runtime --list-devices                 # which microphone?
```

`--mic` draws the input level beside the score, and redraws it in place:

```
mic  -22.0 dB [████████████████░░░░░░]  wake 0.03 [█░░░░░░░░░░░░░░░░░░░░░]
[18:42:07] KOSMO  p=0.99
```

**Start here when a model "does not work".** Almost everything that goes
wrong goes wrong before the model — the wrong input device, a muted one, gain
so low the word is in the noise — and none of it raises. It just means the
word is never heard, which looks exactly like a bad model until you watch the
level. If the bar stays at the floor the meter says `no signal` and you want
`--list-devices` and `--device`.

The same numbers are available to your own code:

```python
for heard in ear.listen():
    print(heard.level_db, heard.probability, heard.detected)
```

## Audio

Anything mono: `float32` in -1..1, `int16`, or raw PCM16 bytes, shaped `(n,)`
or `(n, 1)`. Blocks may be any length and need not be consistent — what does
not fill a step is carried to the next call.

**Pass the rate you actually record at.** Sound cards rarely offer 16 kHz, so
`feed(block, rate=48_000)` is the normal case; conversion to the model's rate
happens inside, with a proper anti-aliasing filter and state kept across
blocks. Getting this wrong does not raise, it just means the word is never
heard, so it is the first thing to check.

## Tuning

```python
ear = WakeWord("Kosmo.onnx", threshold=0.9, min_hits=2, refractory=2.0)
```

| | what it does | when to change it |
|---|---|---|
| `threshold` | score a window must beat | raise if it wakes on its own, lower if you have to say the word twice |
| `min_hits` | consecutive windows needed | 2 filters most single-frame noise for one step of latency |
| `refractory` | quiet time after a detection | stops one word being counted as it leaves the window |
| `step_ms` | how often it scores | latency against CPU; 100 ms is a tenth of a second and a fraction of a core |

All four default to what the model's `.json` says. The threshold is the one
the training run calibrated for one false wake-up per hour of held-out speech,
measured behind exactly this `min_hits` / `refractory` / `step_ms` gate —
change the gate and that rate no longer holds. If a false wake is expensive,
because it opens a paid connection or wakes a household, raise the threshold,
or recalibrate for a stricter target with `wakemeup eval --target-false-wakes
0.2 --write` and export again. `ear.probability` is the last score, which is
what to watch while deciding.

Call `ear.reset()` whenever the microphone has been hearing something that must
not become a detection — your own speaker above all — and after any gap in
capture.

## openWakeWord models

A model trained with openWakeWord (the `.onnx` its training notebook writes)
is loaded the same way, from its `.onnx` alone:

```python
ear = WakeWord("hey_jarvis.onnx")      # no .json: recognised by its input
if ear.feed(block, rate=48_000):
    print(f"heard {ear.word}")
```

```bash
python -m wakemeup_runtime hey_jarvis.onnx --info
python -m wakemeup_runtime hey_jarvis.onnx --mic
```

So you can use the wakeword you got from WakeMeUp, or train another one with
openWakeWord's own tools and run it here without changing your code.

- **What is computed here.** An openWakeWord model is only the last stage: it
  scores the speech embeddings of the last 16 (or so) steps of 80 ms. The
  stages in front of it are computed by this package: the mel spectrogram in
  numpy, and Google's speech embedding with `embedding_model.onnx`, which is
  fetched once from openWakeWord's v0.5.1 release (1.3 MB, checked against its
  SHA-256) into `~/.cache/wakemeup/oww` (or `$WAKEMEUP_CACHE_DIR/oww`). Offline,
  download it yourself and pass `WakeWord(..., features="path/to/embedding_model.onnx")`
  or `--features`. The features equal those inside a WakeMeUp model, to about
  1e-6.
- **Detection.** As in openWakeWord: one step scoring `threshold` (default
  0.5) or more is a detection (`min_hits=1`), then `refractory` (2 s) of not
  listening; the first 5 steps score 0. All three can be set as for any model.
- **`.onnx` only.** openWakeWord also exports `.tflite`; this package runs
  onnxruntime, so use the `.onnx` written beside it. Models with more than one
  wakeword in one file (several outputs) are not supported: load one per model.
- **Licences.** The speech embedding is Google's, under the Apache License
  2.0. openWakeWord's own pre-trained wakewords ("hey jarvis", "alexa", ...)
  are under CC BY-NC-SA 4.0, so not for commercial use; a model you train
  yourself is yours to license.
- **Cost.** About 1.6 ms per 80 ms step on one desktop core, like a WakeMeUp
  model's streaming graph (`--bench` measures it on your device).

## Notes

- **Streaming.** A model over openWakeWord's features scores a 1.98 s window
  every 80 ms. Recomputing that window each time costs about 22 ms on one
  desktop core, too much for a Raspberry Pi; its `.stream.onnx` computes only
  the newest 80 ms and keeps the rest as state, about 1.9 ms on the same core,
  with the same scores. `WakeWord` uses it whenever it sits beside the model
  and `step_ms` is left at 80 (`streaming=False` or `--no-streaming` turns it
  off; without the file it falls back to whole windows, with a warning).
  `--info` says which it runs, `--bench` how long a step takes.
- **One instance per audio stream.** It is stateful and not thread-safe.
- **Feed it from a worker, not from an audio callback.** One inference per
  `step_ms` is cheap but not free, and a callback that overruns its deadline
  drops microphone audio.
- `scan(samples, rate)` runs over a whole recording and returns the time of
  each detection, for testing against files.
