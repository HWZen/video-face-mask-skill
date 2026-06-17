import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import insightface
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def apply_mosaic(img, x1, y1, x2, y2, min_block_size=20, max_cells=18):
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return img

    block = max(min_block_size, int(np.ceil(max(w, h) / max_cells)))
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(
        roi,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = mosaic
    return img


def apply_sticker(img, x1, y1, x2, y2, sticker):
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return img

    sh, sw = sticker.shape[:2]
    if sh <= 0 or sw <= 0:
        return img

    scale = max(w / sw, h / sh)
    rw = max(1, int(round(sw * scale)))
    rh = max(1, int(round(sh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(sticker, (rw, rh), interpolation=interp)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx1 = cx - rw // 2
    sy1 = cy - rh // 2
    sx2 = sx1 + rw
    sy2 = sy1 + rh

    ix1 = max(0, sx1)
    iy1 = max(0, sy1)
    ix2 = min(img.shape[1], sx2)
    iy2 = min(img.shape[0], sy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return img

    px1 = ix1 - sx1
    py1 = iy1 - sy1
    px2 = px1 + (ix2 - ix1)
    py2 = py1 + (iy2 - iy1)
    patch = resized[py1:py2, px1:px2]

    if patch.shape[2] == 4:
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        src = patch[:, :, :3].astype(np.float32)
        dst = img[iy1:iy2, ix1:ix2].astype(np.float32)
        out = src * alpha + dst * (1.0 - alpha)
        img[iy1:iy2, ix1:ix2] = out.astype(np.uint8)
    else:
        img[iy1:iy2, ix1:ix2] = patch[:, :, :3]
    return img


def expand_box(box, width, height, scale=1.2):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(cx - bw * 0.5))
    ny1 = max(0, int(cy - bh * 0.5))
    nx2 = min(width, int(cx + bw * 0.5))
    ny2 = min(height, int(cy + bh * 0.5))
    return nx1, ny1, nx2, ny2


def enable_ort_cuda_from_torch():
    try:
        import torch
    except ImportError:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        os.add_dll_directory(str(torch_lib))
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def detect_faces_two_pass(face_app, frame, upscale=1.5):
    faces = face_app.get(frame)
    if faces:
        return [f.bbox.astype(int) for f in faces]

    h, w = frame.shape[:2]
    up_w = int(w * upscale)
    up_h = int(h * upscale)
    up = cv2.resize(frame, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    faces_up = face_app.get(up)
    if not faces_up:
        return []

    boxes = []
    for face in faces_up:
        x1, y1, x2, y2 = face.bbox.astype(float)
        boxes.append([
            int(x1 / upscale),
            int(y1 / upscale),
            int(x2 / upscale),
            int(y2 / upscale),
        ])
    return boxes


def run_command(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def has_nvidia_gpu():
    result = run_command(["nvidia-smi", "-L"])
    return result.returncode == 0 and bool(result.stdout.strip())


def ffmpeg_has_encoder(name):
    result = run_command(["ffmpeg", "-hide_banner", "-encoders"])
    return result.returncode == 0 and name in result.stdout


def choose_video_encoder(preferred="auto"):
    if preferred != "auto":
        return preferred
    if has_nvidia_gpu() and ffmpeg_has_encoder("hevc_nvenc"):
        return "hevc_nvenc"
    return "libx265"


def build_ffmpeg_encode_command(width, height, fps, output_path, encoder, quality):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
    ]

    if encoder == "hevc_nvenc":
        cmd += [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
        ]
    elif encoder == "libx265":
        cmd += [
            "-c:v",
            "libx265",
            "-crf",
            str(quality),
            "-preset",
            "medium",
        ]
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    cmd += ["-pix_fmt", "yuv420p", str(output_path)]
    return cmd


def mux_audio(video_only_path, source_video_path, final_output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(final_output_path),
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg audio mux failed")


def ensure_tool_exists(name):
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required tool not found in PATH: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Mask faces in a video with InsightFace.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--mode", choices=["mosaic", "sticker"], required=True)
    parser.add_argument("--sticker", help="Sticker image path for sticker mode")
    parser.add_argument("--det-size", type=int, default=960)
    parser.add_argument("--det-thresh", type=float, default=0.28)
    parser.add_argument("--upscale", type=float, default=1.5)
    parser.add_argument("--expand-scale", type=float, default=1.2)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--mosaic-min-block", type=int, default=20)
    parser.add_argument("--mosaic-max-cells", type=int, default=18)
    parser.add_argument("--encoder", choices=["auto", "hevc_nvenc", "libx265"], default="auto")
    parser.add_argument("--quality", type=int, default=28)
    parser.add_argument("--no-audio", action="store_true", help="Do not copy original audio into the final output")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "sticker":
        if not args.sticker:
            raise ValueError("--sticker is required when --mode sticker")
        sticker_path = Path(args.sticker).resolve()
        sticker = read_image_unicode(sticker_path)
        if sticker is None:
            raise FileNotFoundError(f"Sticker image could not be loaded: {sticker_path}")
    else:
        sticker = None

    ensure_tool_exists("ffmpeg")
    ensure_tool_exists("ffprobe")

    enable_ort_cuda_from_torch()
    available = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    ctx_id = 0 if use_cuda else -1

    face_app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    print(
        f"Detector: InsightFace buffalo_l "
        f"(det_thresh={args.det_thresh}, det_size={args.det_size}, providers={providers})"
    )

    cap = cv2.VideoCapture(str(input_path), cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {width}x{height} @ {fps:.2f}fps, {total} frames")

    encoder = choose_video_encoder(args.encoder)
    print(f"Encoder: {encoder}")

    video_only_path = output_path.with_name(output_path.stem + ".video_only" + output_path.suffix)
    ffmpeg_cmd = build_ffmpeg_encode_command(width, height, fps, video_only_path, encoder, args.quality)
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    pbar = tqdm(total=total if total > 0 else None, unit="frame", desc="Processing", ncols=80)

    last_detected_boxes = []
    miss_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected_boxes = detect_faces_two_pass(face_app, frame, upscale=args.upscale)
        if detected_boxes:
            boxes = detected_boxes
            last_detected_boxes = [b[:] for b in detected_boxes]
            miss_count = 0
        elif last_detected_boxes and miss_count < args.hold_frames:
            boxes = last_detected_boxes
            miss_count += 1
        else:
            boxes = []
            last_detected_boxes = []
            miss_count = 0

        for box in boxes:
            x1, y1, x2, y2 = expand_box(box, width, height, scale=args.expand_scale)
            if args.mode == "mosaic":
                apply_mosaic(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    min_block_size=args.mosaic_min_block,
                    max_cells=args.mosaic_max_cells,
                )
            else:
                apply_sticker(frame, x1, y1, x2, y2, sticker)

        pipe.stdin.write(frame.tobytes())
        pbar.update(1)

    pbar.close()
    cap.release()
    pipe.stdin.close()
    pipe.wait()

    if pipe.returncode != 0:
        raise RuntimeError("Video encoding failed")

    if args.no_audio:
        if output_path.exists():
            output_path.unlink()
        video_only_path.replace(output_path)
    else:
        mux_audio(video_only_path, input_path, output_path)
        video_only_path.unlink(missing_ok=True)

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
