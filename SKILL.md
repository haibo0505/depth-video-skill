---
name: depth-video
description: 将普通视频转换为灰度相对深度 MP4，使用 Depth Anything V2。用户要求制作深度图视频或参考 Wenxi-learning/DepthVideo 时使用；不要用于立体 3D 视频或人物二值遮罩。
metadata:
  short-description: Generate grayscale depth-map videos
---

# 深度视频复刻

Create a grayscale relative-depth video from the user's source. Nearer regions should appear brighter and farther regions darker. This is a depth estimate, not a stereo pair or a person-only mask.

Use `scripts/make_depth_video.py` for the conversion. It downloads the official Depth Anything V2 Small code and checkpoint on first run, estimates depth frame by frame, normalizes the clip using one shared 1st-to-99th-percentile range, and encodes an H.264 grayscale MP4. Output defaults to the source dimensions and frame rate and has no audio, matching the referenced DepthVideo workflow.

Before running:

- Confirm the source file exists and identify a separate output path. Preserve the original video.
- Check for FFmpeg, Python dependencies, network access for the first model download, and enough temporary disk space for the float depth cache.
- If the output path already exists, choose a new versioned path or get the user's direction; do not overwrite it silently.
- Keep the input aspect ratio. Use another output size only when requested.

Run the helper with the input and output paths. Pass `--keep-audio` only if the user wants the original audio retained. The script can also accept `--width` and `--height` together to fit the depth image inside a requested canvas, and `--input-size` to trade speed for detail.

After rendering, confirm that the MP4 decodes, has the expected dimensions, frame rate, and duration, and contains a grayscale stream. Sample a few frames across the clip to check that the foreground/background depth ordering is sensible. Report that estimated depth can be uncertain in dark, transparent, reflective, or textureless regions.

For method and model provenance, see [README.md](README.md#method-and-model).
