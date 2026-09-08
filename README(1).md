# UE Crowd Gender Counter (Unreal Engine + Python/OpenCV 5 ONNX)

Capture any crowd scene in Unreal Engine and automatically detect every
visible face, classify its visual gender presentation (Male/Female), and
report the counts back to Blueprint — with **zero manual model downloads**
and **zero model files checked into your project**. The entire pipeline is
OpenCV 5.x-native ONNX, running in-editor via Unreal's Python Editor
Script Plugin.

| Raw capture | Labeled output (3 faces, all classified) |
|---|---|
| ![raw](docs/screenshots/capture-crowd-raw.png) | ![labeled](docs/screenshots/capture-crowd-labeled.png) |

## Why this exists: OpenCV 5 killed the old approach

The classic recipe for this (Levi & Hassner `gender_net.caffemodel` + the
`opencv_face_detector_uint8.pb` ResNet-SSD) is **dead on OpenCV 5.0** —
both the Caffe *and* TensorFlow DNN importers were removed, so
`cv2.dnn.readNet` fails on those files with
`Caffe importer has been removed`. This project is the drop-in
replacement: two ONNX models, loaded with `cv2.dnn.readNetFromONNX` /
`cv2.FaceDetectorYN`, with the same stdout contract your Blueprints
already expect.

## What's novel

- **No `models/` folder.** Both ONNX weights live on GitHub; on first run
  `gender.py` downloads them from pinned URLs into a local cache
  (`Content/Python/_model_cache/`) and reuses them forever. First run
  needs internet; every run after is fully offline. Swap two URL
  constants to point at your own mirrored weights.
- **OpenCV 5.x-correct YuNet.** Uses the dynamic-input-shape
  `face_detection_yunet_2026may.onnx` build — the fixed-shape 2023mar
  file misbehaves with the OpenCV 5 DNN engine.
- **Verified gender polarity.** The genderage model emits
  `[female_score, male_score, age]`; index **1 = Male, 0 = Female** (per
  the upstream author's documented convention), so counts can't silently
  invert.
- **Blueprint-native I/O contract.** `run_from_path()` prints exactly one
  line — `M,F` (e.g. `0,3`) or `-1,-1` on failure — nothing else on
  stdout. Details go to `unreal.log_error()` (Output Log), not the pipe
  your Blueprint parses.
- **Non-blocking screenshot readiness.** `Take High Res Screenshot`
  returns before the PNG finishes writing. `check_screenshot_ready()` is
  an instant, stateful size-stability check (safe to call from a
  Blueprint Delay-poll loop) — the script never sleeps, never freezes
  the editor.

## How it works

The pipeline is one Blueprint flow handing off to one Python script:

1. **Capture** — `Take High Res Screenshot` saves a timestamped PNG. The
   Blueprint polls `check_screenshot_ready(path)` (via `Execute Python
   Command` + a Delay node) until the file size stops changing.

2. **Count** — a final `Execute Python Command (Advanced)` runs, in
   **Execute Statement** mode:

   ```python
   import importlib, gender
   importlib.reload(gender)
   gender.run_from_path(r"<CapturedPhotoPath>")
   ```

   Blueprint reads the `Log Output` string, `Split String` on `",`, and
   `String → Int` each half.

3. **Detect faces** — YuNet (`cv2.FaceDetectorYN`) with score threshold
   `0.6`, NMS `0.3`, and a 20 px minimum-face filter to reject noise.

4. **Classify gender** — each face box is center-crop aligned
   (scale = `224 / (max(w,h)·1.5)`, per the upstream reference impl),
   fed to the ~1 MB `genderage.onnx` at 224×224 (BGR→RGB, zero mean),
   and labeled by `argmax` of the two gender scores.

5. **Report** — counts printed as `M,F`; a labeled copy
   (`<name>_labeled.png`) and optional JSON of every box are written next
   to the capture.

## Setup

1. Enable **Python Editor Script Plugin** (Edit → Plugins).
2. Install dependencies into Unreal's *own* bundled Python interpreter
   (find it with `import sys; print(sys.executable)` in the Python
   console), then:
   ```
   "<path from above>" -m pip install "opencv-python>=5" numpy
   ```
3. Drop `gender.py` into `<YourProject>/Content/Python/`. That's it — no
   model files, no `models/` folder. The first `run_from_path` call
   downloads ~1.2 MB of ONNX weights from GitHub automatically.
   (Air-gapped machine? Download the two URLs at the top of `gender.py`
   manually into `Content/Python/_model_cache/`.)
4. Build the Blueprint: Delay-poll `check_screenshot_ready()`, then call
   the one-liner above via `Execute Python Command (Advanced)`.

## Known limitations

- **Estimate, not ground truth.** This classifies *visual facial
  presentation* with a small CNN trained on real photos. MetaHuman /
  stylized crowd characters shift the distribution — expect lower
  confidence (the example above: 22–47%) and spot-check before trusting
  counts for anything beyond rough QA.
- **Faces under ~20 px are skipped** (tunable via `MIN_FACE_PX`), and
  profile/strongly turned heads may be missed by the detector.
- **Editor-only.** The Python Editor Script Plugin doesn't ship in
  packaged builds; a shipped game would need a C++ port or a bundled
  Python runtime.
- **First run needs internet** for the one-time model download.

## Repo layout

```
Content/Python/gender.py            # detection + classification + Blueprint entry point
docs/screenshots/
  capture-crowd-raw.png             # unannotated crowd capture
  capture-crowd-labeled.png         # boxed + classified output ("Male: 0  Female: 3  Total: 3")
```
