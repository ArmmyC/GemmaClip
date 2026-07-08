from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gemmaclip.captioner import build_fallback_captions, generate_captions
from gemmaclip.download import download_video
from gemmaclip.frames import DEFAULT_FRAME_STRATEGY, VALID_FRAME_STRATEGIES, export_debug_artifacts, extract_frames
from gemmaclip.io import Task, make_frame_manifest_entry, read_tasks, write_frame_manifest, write_results
from gemmaclip.validate import validate_results
from gemmaclip.video import probe_video

DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"
DEFAULT_WORKDIR = "/tmp/gemmaclip"

LOGGER = logging.getLogger("gemmaclip")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        tasks = read_tasks(args.input)
        workdir = Path(args.workdir).resolve()
        debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None
        frame_manifest: list[dict[str, Any]] = []
        results = []
        for task in tasks:
            result, manifest_entry = process_task(
                task,
                workdir=workdir,
                debug_dir=debug_dir,
                dry_run=args.dry_run,
                frame_strategy=args.frame_strategy,
            )
            results.append(result)
            if manifest_entry is not None:
                frame_manifest.append(manifest_entry)

        write_frame_manifest(frame_manifest, workdir / "frame_manifest.json")
        validate_results(tasks, results)
        write_results(results, args.output)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GemmaClip Track 2 baseline pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Path to the input tasks JSON file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to the output results JSON file.")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Working directory for downloaded videos and extracted frames.")
    parser.add_argument(
        "--debug-dir",
        default="",
        help="Optional directory for copied frame artifacts and per-task contact sheets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use placeholder captions and skip all Gemma API calls.",
    )
    parser.add_argument(
        "--frame-strategy",
        choices=sorted(VALID_FRAME_STRATEGIES),
        default=DEFAULT_FRAME_STRATEGY,
        help="Frame extraction strategy.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def process_task(
    task: Task,
    workdir: Path,
    debug_dir: Path | None = None,
    dry_run: bool = False,
    frame_strategy: str = DEFAULT_FRAME_STRATEGY,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        video_path = download_video(task, destination_dir=workdir / "videos")
        metadata = probe_video(video_path)
        extracted_frames = extract_frames(
            task.task_id,
            video_path,
            metadata,
            strategy=frame_strategy,
            destination_root=workdir / "frames",
        )
        manifest_entry = make_frame_manifest_entry(task.task_id, video_path, extracted_frames, metadata)
        if debug_dir is not None:
            export_debug_artifacts(task.task_id, extracted_frames, debug_dir)
        LOGGER.info(
            "Processed task %s: duration=%.2fs frames=%s extracted=%s",
            task.task_id,
            metadata.duration_seconds,
            metadata.frame_count if metadata.frame_count is not None else "unknown",
            len(extracted_frames),
        )
        captions = generate_captions(
            task,
            extracted_frames,
            dry_run=dry_run,
            debug_dir=debug_dir,
            logger=LOGGER,
        )
    except Exception as exc:
        LOGGER.warning("Task %s failed, writing fallback captions: %s", task.task_id, exc)
        captions = build_fallback_captions(task.styles)
        manifest_entry = None

    return (
        {
            "task_id": task.task_id,
            "captions": captions,
        },
        manifest_entry,
    )


if __name__ == "__main__":
    raise SystemExit(main())
