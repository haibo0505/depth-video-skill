# 深度视频复刻

A Codex skill and command-line helper for turning ordinary video into a grayscale relative-depth MP4. Nearer areas are brighter; farther areas are darker.

The workflow follows the output format of [Wenxi-learning/DepthVideo](https://github.com/Wenxi-learning/DepthVideo): estimate depth frame by frame, normalize the clip with a shared robust range, and encode H.264 grayscale. This repository contains an independent runner; it does not copy or bundle the upstream tool, its model source, or model weights.

## Install as a Codex skill

Copy this folder to the Codex skills directory:

```text
~/.codex/skills/depth-video
```

On Windows, the default location is:

```text
%USERPROFILE%\.codex\skills\depth-video
```

Restart or refresh Codex's skill discovery if the skill does not appear immediately.

## Requirements

- Python 3.10 or newer
- FFmpeg with `ffmpeg` available on `PATH`
- Git, internet access on first run, and enough temporary disk space for a float depth cache
- Python dependencies from `requirements.txt`

Create an isolated Python environment and install the dependencies:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The helper downloads the official Depth Anything V2 Small source and checkpoint into the user's cache on first use. The source code is obtained from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and the checkpoint from [Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Small). The small model is released under Apache-2.0.

## Use

```powershell
.venv\Scripts\python.exe scripts\make_depth_video.py `
  --input "D:\素材\dance.mp4" `
  --output "D:\素材\dance_depth.mp4"
```

The output uses the source dimensions and frame rate by default and drops audio. To retain the source audio, add `--keep-audio`.

Optional controls:

```text
--width WIDTH --height HEIGHT   Fit into an output canvas while preserving aspect ratio
--input-size PIXELS             Model long-side size; default 518
--keep-audio                    Mux the source audio into the output
--overwrite                     Replace an existing output path
--temp-dir PATH                 Store temporary depth cache on a chosen drive
--model-code-dir PATH           Reuse an existing Depth-Anything-V2 checkout
--weights PATH                  Reuse a downloaded Small-model checkpoint
```

The helper never overwrites the source. It refuses to replace an existing output unless `--overwrite` is supplied.

## Method and model

Depth Anything V2 Small predicts relative monocular depth, not measured distances. The script stores reduced-resolution float predictions for the clip, samples every sixth frame, computes a shared 1st-to-99th-percentile range, then maps the range to 0–255 grayscale. This keeps contrast stable across frames and makes closer regions brighter in the output.

The output is H.264 MP4 in grayscale-compatible `yuv420p`; audio is omitted unless requested. Model predictions can be uncertain for transparent, reflective, textureless, or very dark content.

## License

The skill and helper are released under the MIT License. The model source and checkpoint remain subject to their own license terms; see the upstream [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) and [model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small).
