"""Create tiny synthetic clips for local infrastructure and UI smoke checks.

These clips are intentionally not caption-quality fixtures. The tone clip contains
an audible sine wave, not speech.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("examples/demo-videos"))
    parser.add_argument("--force", action="store_true", help="replace existing generated clips")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg is required to create demo videos")
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = {
        "moving-shape-silent.mp4": [
            "-f", "lavfi", "-i", "color=c=0x101826:s=640x360:r=24",
            "-vf", "drawbox=x='20+120*t':y=130:w=90:h=90:color=0xff6b3d:t=fill",
            "-t", "4", "-an",
        ],
        "steady-color-silent.mp4": [
            "-f", "lavfi", "-i", "color=c=0x273a54:s=640x360:r=24",
            "-vf", "drawgrid=w=80:h=80:t=2:c=0x72a7ff@0.65",
            "-t", "4", "-an",
        ],
        "generated-tone-no-speech.mp4": [
            "-f", "lavfi", "-i", "color=c=0x18251d:s=640x360:r=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-vf", "drawbox=x=250:y=120:w=140:h=120:color=0x54d39a:t=fill",
            "-map", "0:v:0", "-map", "1:a:0", "-t", "4",
        ],
    }
    for filename, inputs in jobs.items():
        destination = args.output / filename
        if destination.exists() and not args.force:
            continue
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(destination),
        ]
        subprocess.run(command, check=True, timeout=30, capture_output=True)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
