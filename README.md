# video-face-mask-skill

Reusable AI skill + helper script for masking faces in videos with `InsightFace` + `OpenCV`.

This repo focuses on one practical workflow:

- detect faces with `InsightFace`
- cover them with either mosaic or a sticker image
- keep the original audio track by default
- prefer GPU inference/encoding when available, otherwise fall back cleanly

## Core Skill File

If another AI agent or automation tool needs the actual skill instructions, the core skill file is:

- `D:/Github/video-face-mask-skill/.opencode/skills/video-face-mask/SKILL.md`

Repo-relative path:

- `.opencode/skills/video-face-mask/SKILL.md`

That file contains the workflow rules, environment-selection logic, dependency-selection logic, required user-input checks, execution defaults, and parameter-tuning guidance.

## Repo Layout

- `.opencode/skills/video-face-mask/SKILL.md`: the reusable OpenCode skill
- `scripts/face_mask_video.py`: helper script the skill runs
- `requirements-cpu.txt`: CPU dependencies
- `requirements-gpu.txt`: NVIDIA GPU dependencies

## Use In OpenCode

Point OpenCode at this repo's skill directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "skills": {
    "paths": [
      "D:/Github/video-face-mask-skill/.opencode/skills"
    ]
  }
}
```

Then restart OpenCode.

## What The Script Does

- `InsightFace buffalo_l` face detection
- two-pass detection: original frame, then 1.5x upscale retry
- short hold for missed detections: default `4` frames
- box expansion: default `1.2`
- mosaic mode:
  - minimum block size `20`
  - no more than `18` cells on the long edge
- sticker mode:
  - no stretching
  - no crop-to-face-box
  - scale by aspect ratio and place centered over the face
- output video:
  - prefers `hevc_nvenc` on supported NVIDIA systems
  - otherwise falls back to `libx265`
- audio:
  - remuxes the original audio track back into the output by default

## Typical Usage

```bash
python scripts/face_mask_video.py --input input.mp4 --output output.mp4 --mode mosaic
```

```bash
python scripts/face_mask_video.py --input input.mp4 --output output.mp4 --mode sticker --sticker ./ChatGPT_doro.png
```

## Notes

- `ffmpeg` and `ffprobe` are part of the required environment. If they are missing, the operator or AI agent should install them and ensure they are in `PATH` before processing.
- On Windows + NVIDIA, `onnxruntime-gpu` may need CUDA/cuDNN DLLs. This repo's workflow can use `torch` only as a runtime DLL provider for ONNX Runtime. It is not used for YOLO.
- The skill itself contains the environment-selection and installation rules, including when to ask the user and when to decide for a beginner.
