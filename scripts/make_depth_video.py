#!/usr/bin/env python3
"""Generate an H.264 grayscale relative-depth video with Depth Anything V2 Small."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


UPSTREAM_CODE = "https://github.com/DepthAnything/Depth-Anything-V2.git"
MODEL_REPO = "depth-anything/Depth-Anything-V2-Small"
MODEL_FILE = "depth_anything_v2_vits.pth"
MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}


def multiple_of_14(value: float) -> int:
    return max(14, int(round(value / 14.0) * 14))


def model_dimensions(width: int, height: int, long_side: int) -> tuple[int, int]:
    if height >= width:
        return multiple_of_14(width * long_side / height), long_side
    return long_side, multiple_of_14(height * long_side / width)


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def fit_dimensions(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    scale = min(out_w / src_w, out_h / src_h)
    return even(round(src_w * scale)), even(round(src_h * scale))


def ensure_model_code(path: Path) -> Path:
    source_file = path / "depth_anything_v2" / "dpt.py"
    if source_file.is_file():
        return path
    if path.exists():
        raise RuntimeError(
            f"Model-code directory exists but is not a Depth Anything V2 checkout: {path}"
        )
    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "Git is required for the first run. Install Git or pass --model-code-dir "
            "pointing to an existing Depth-Anything-V2 checkout."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading official Depth Anything V2 source...", flush=True)
    subprocess.run([git, "clone", "--depth", "1", UPSTREAM_CODE, str(path)], check=True)
    if not source_file.is_file():
        raise RuntimeError(f"The cloned repository does not contain {source_file}")
    return path


def load_model(cache_root: Path, model_code_dir: Path | None, weights_path: Path | None):
    if model_code_dir is None:
        model_code_dir = ensure_model_code(cache_root / "Depth-Anything-V2")
    elif not (model_code_dir / "depth_anything_v2" / "dpt.py").is_file():
        raise RuntimeError(f"Not a Depth-Anything-V2 checkout: {model_code_dir}")
    sys.path.insert(0, str(model_code_dir.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    if weights_path is not None:
        weights = weights_path.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {weights}")
    else:
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model",
            cache_dir=str(cache_root / "huggingface"),
        )
    model = DepthAnythingV2(**MODEL_CONFIG)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device).eval()
    print(f"Depth model ready on {device}.", flush=True)
    return model


def resolve_output_size(args: argparse.Namespace, src_w: int, src_h: int) -> tuple[int, int]:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    width = even(args.width if args.width is not None else src_w)
    height = even(args.height if args.height is not None else src_h)
    if width < 2 or height < 2:
        raise ValueError("Output dimensions must be at least 2 pixels")
    return width, height


def build_ffmpeg_command(
    ffmpeg: str,
    output: Path,
    width: int,
    height: int,
    fps: float,
    source: Path,
    keep_audio: bool,
    overwrite: bool,
) -> list[str]:
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "pipe:0",
    ]
    if keep_audio:
        command.extend(["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        command.append("-an")
    command.extend([
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
    ])
    if keep_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    command.extend(["-movflags", "+faststart", str(output)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a video into a grayscale relative-depth H.264 MP4."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source video")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4")
    parser.add_argument("--width", type=int, help="Output canvas width; use with --height")
    parser.add_argument("--height", type=int, help="Output canvas height; use with --width")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Depth model long-side input size (default: 518)")
    parser.add_argument("--keep-audio", action="store_true",
                        help="Include the source audio track (off by default)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output MP4")
    parser.add_argument("--temp-dir", type=Path,
                        help="Directory for temporary float depth cache files")
    parser.add_argument("--model-code-dir", type=Path,
                        help="Existing official Depth-Anything-V2 source checkout")
    parser.add_argument("--weights", type=Path,
                        help="Existing Depth Anything V2 Small checkpoint")
    parser.add_argument("--cache-root", type=Path,
                        help="Cache directory for model source and downloaded weights")
    parser.add_argument("--limit", type=int, default=0,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Output path must be different from the source video")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose another path or pass --overwrite."
        )
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg and retry.")
    if args.input_size < 224:
        raise ValueError("--input-size must be at least 224")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_w <= 0 or src_h <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions or frame count")
    if args.limit > 0:
        frame_count = min(frame_count, args.limit)

    out_w, out_h = resolve_output_size(args, src_w, src_h)
    content_w, content_h = fit_dimensions(src_w, src_h, out_w, out_h)
    offset_x, offset_y = (out_w - content_w) // 2, (out_h - content_h) // 2
    model_w, model_h = model_dimensions(src_w, src_h, args.input_size)
    cache_root = args.cache_root or Path.home() / ".cache" / "depth-video"
    cache_root = cache_root.expanduser().resolve()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = load_model(cache_root, args.model_code_dir, args.weights)

    temp_parent = args.temp_dir.expanduser().resolve() if args.temp_dir else None
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = frame_count * model_w * model_h * np.dtype(np.float32).itemsize
    free_bytes = shutil.disk_usage(temp_parent or Path(tempfile.gettempdir())).free
    if free_bytes < estimated_bytes + 64 * 1024 * 1024:
        cap.release()
        raise OSError(
            f"Not enough free space for the temporary depth cache (~{estimated_bytes / 1e9:.2f} GB). "
            "Choose another --temp-dir."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="depth-video-", dir=temp_parent) as temp_name:
        cache_path = Path(temp_name) / "depth-cache.npy"
        cache = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32,
            shape=(frame_count, model_h, model_w),
        )
        samples = []
        print(
            f"Input: {src_w}x{src_h} at {fps:.3f} fps; processing {frame_count} frames.",
            flush=True,
        )
        with torch.inference_mode():
            for index in range(frame_count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended before frame {index + 1}/{frame_count}")
                prediction = model.infer_image(frame, args.input_size)
                reduced = cv2.resize(
                    prediction, (model_w, model_h), interpolation=cv2.INTER_AREA
                )
                cache[index] = reduced
                if index % 6 == 0:
                    samples.append(reduced[::4, ::4].reshape(-1))
                if (index + 1) % 50 == 0 or index + 1 == frame_count:
                    print(f"Depth inference {index + 1}/{frame_count}", flush=True)
        cap.release()
        cache.flush()

        sample_values = np.concatenate(samples)
        low, high = np.percentile(sample_values, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(cache)), float(np.max(cache))
        if high <= low:
            high = low + 1.0
        print(f"Shared grayscale range: {low:.4f} to {high:.4f}", flush=True)

        try:
            encoder = subprocess.Popen(
                build_ffmpeg_command(
                    ffmpeg, output, out_w, out_h, fps, source,
                    args.keep_audio, args.overwrite,
                ),
                stdin=subprocess.PIPE,
            )
            try:
                for index in range(frame_count):
                    normalized = np.clip((cache[index] - low) / (high - low), 0.0, 1.0)
                    gray = (normalized * 255.0).astype(np.uint8)
                    gray = cv2.resize(gray, (content_w, content_h), interpolation=cv2.INTER_CUBIC)
                    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
                    canvas[offset_y:offset_y + content_h, offset_x:offset_x + content_w] = gray
                    encoder.stdin.write(canvas.tobytes())
                    if (index + 1) % 100 == 0 or index + 1 == frame_count:
                        print(f"Encoding {index + 1}/{frame_count}", flush=True)
            except Exception:
                encoder.stdin.close()
                encoder.wait()
                raise
            else:
                encoder.stdin.close()
            return_code = encoder.wait()
        finally:
            cache.flush()
            cache._mmap.close()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}")

    print(f"Created grayscale depth video: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
