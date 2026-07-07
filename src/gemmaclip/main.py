from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

from gemmaclip.download import download_video
from gemmaclip.frames import extract_uniform_frames
from gemmaclip.io import Task, read_tasks, write_results
from gemmaclip.validate import validate_results
from gemmaclip.video import probe_video

DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"

LOGGER = logging.getLogger("gemmaclip")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        tasks = read_tasks(args.input)
        results = [process_task(task) for task in tasks]
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
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def process_task(task: Task) -> dict[str, Any]:
    try:
        video_path = download_video(task)
        metadata = probe_video(video_path)
        extracted_frames = extract_uniform_frames(task.task_id, video_path, metadata)
        LOGGER.info(
            "Processed task %s: duration=%.2fs frames=%s extracted=%s",
            task.task_id,
            metadata.duration_seconds,
            metadata.frame_count if metadata.frame_count is not None else "unknown",
            len(extracted_frames),
        )
        captions = build_placeholder_captions(task.styles)
    except Exception as exc:
        LOGGER.warning("Task %s failed, writing fallback captions: %s", task.task_id, exc)
        captions = build_fallback_captions(task.styles)

    return {
        "task_id": task.task_id,
        "captions": captions,
    }


def build_placeholder_captions(styles: Sequence[str]) -> dict[str, str]:
    templates = {
        "formal": (
            "A short video clip is available, and a fuller caption will be generated after the visual analysis stage is enabled."
        ),
        "sarcastic": (
            "A short video clip arrives, politely expecting interpretation before the interesting captioning logic has actually shown up."
        ),
        "humorous_tech": (
            "A short video clip is standing by while the pipeline waits for its real captioning upgrade instead of placeholder mode."
        ),
        "humorous_non_tech": (
            "A short video clip shows up, and the caption is still warming up like a comedian before the first joke."
        ),
    }
    return {style: templates[style] for style in styles}


def build_fallback_captions(styles: Sequence[str]) -> dict[str, str]:
    templates = {
        "formal": (
            "The video could not be fully processed, so this placeholder caption notes a short scene with visible activity."
        ),
        "sarcastic": (
            "The video resisted a full analysis, which is a very efficient way to remain slightly mysterious."
        ),
        "humorous_tech": (
            "The clip hit a processing snag, so the captioning stack returned a graceful fallback instead of a dramatic crash."
        ),
        "humorous_non_tech": (
            "The video kept some secrets, so this caption politely fills in while the details stay offstage."
        ),
    }
    return {style: templates[style] for style in styles}


if __name__ == "__main__":
    raise SystemExit(main())
