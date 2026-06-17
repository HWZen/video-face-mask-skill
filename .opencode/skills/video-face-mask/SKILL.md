---
name: video-face-mask
description: Use when the user wants to mosaic or sticker over faces in a video with InsightFace and OpenCV, including Python environment setup, GPU or CPU dependency selection, input validation, processing, and final output generation.
---

# Video Face Mask

Use this skill for end-to-end video face masking with `InsightFace` + `OpenCV`.

This skill is opinionated:

- use `InsightFace`, not YOLO
- prefer the smallest correct setup
- prefer GPU inference and encoding when the machine supports it
- keep the original audio in the final output unless the user asks otherwise

## Workflow

### 1. Confirm Python Environment Strategy

Before installing anything, inspect what Python workflow the user already uses.

Check:

- `python --version`
- `py --version` on Windows if needed
- `conda info --envs` if `conda` exists
- whether the current project already has `pyproject.toml`, `requirements.txt`, `environment.yml`, `.python-version`, `.venv/`, or similar

Decide like this:

- if the user clearly already uses `conda`, prefer a new dedicated `conda` environment
- if the user clearly uses `venv`, prefer a new dedicated `venv`
- if the user explicitly wants to reuse an existing environment, follow that
- if the user is inexperienced or gives no clear preference, choose a fresh dedicated environment rather than polluting an existing one
- if it is ambiguous and the user seems experienced, ask one short question before installing

Do not assume that installing into the base environment is acceptable.

### 2. Install Only The Needed Stack

Do not install YOLO or Ultralytics for this skill.

Required runtime stack:

- `insightface`
- `opencv-python`
- `tqdm`
- ONNX Runtime:
  - use `onnxruntime-gpu==1.20.1` when a suitable NVIDIA GPU is present
  - otherwise use `onnxruntime==1.20.1`

GPU check:

- inspect `nvidia-smi`
- if the machine has a supported NVIDIA GPU, prefer GPU mode
- on Windows, if ONNX Runtime GPU DLL loading is likely to be a problem, install `torch` for CUDA runtime DLL availability only

The helper script already knows how to add `torch\lib` to the DLL search path when `torch` is installed.

### 3. Validate Required User Inputs

Before processing, confirm these inputs exist and are sufficient:

- input video path
- masking mode: `mosaic` or `sticker`
- sticker image path when mode is `sticker`
- output path and file name

If any of these are missing, ask only for the missing pieces.

Also verify:

- input video exists
- sticker exists when needed
- output parent directory exists or can be created safely
- `ffmpeg` and `ffprobe` are available in `PATH`

### 4. Execute With The Helper Script

Use `scripts/face_mask_video.py`.

Default technical behavior should match the current working configuration unless the user gives new parameters:

- detector: `InsightFace buffalo_l`
- `det_size=960`
- `det_thresh=0.28`
- two-pass retry upscale: `1.5`
- face box expansion: `1.2`
- missed-frame hold: `4`
- mosaic minimum block size: `20`
- mosaic max cells on long edge: `18`
- preserve original audio: enabled

Example commands:

```bash
python scripts/face_mask_video.py --input input.mp4 --output output.mp4 --mode mosaic
```

```bash
python scripts/face_mask_video.py --input input.mp4 --output output.mp4 --mode sticker --sticker ./ChatGPT_doro.png
```

### 5. Tell The User What They Can Tune

After or before execution, explain the main knobs briefly when relevant:

- `--det-thresh`: lower catches more faces, but increases false positives
- `--det-size`: larger improves hard detections, but costs speed and memory
- `--upscale`: larger helps tiny faces in the retry pass, but slows processing
- `--expand-scale`: larger covers more of the head, but may cover too much background
- `--hold-frames`: larger reduces one-frame misses, but can create trailing overlays
- `--mosaic-min-block`: larger means chunkier mosaic
- `--mosaic-max-cells`: smaller means chunkier mosaic on large faces

If the user gives explicit parameter values, use them instead of defaults.

## Execution Notes

- Prefer `hevc_nvenc` when ffmpeg supports it and the machine has NVIDIA NVENC.
- Otherwise fall back to `libx265`.
- Keep progress visible with `tqdm`.
- Do not silently drop audio from the final file unless the user asked for a silent output.
- When sticker mode is used, do not stretch the sticker. Keep aspect ratio.
- The current sticker behavior should be no-crop over the face box: scale large enough to cover the face dimensions, then place it centered on the face, allowing it to extend beyond the face box.
