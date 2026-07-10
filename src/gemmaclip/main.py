from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gemmaclip.captioner import build_fallback_captions, generate_captions
from gemmaclip.download import download_video
from gemmaclip.frames import DEFAULT_FRAME_STRATEGY, VALID_FRAME_STRATEGIES, export_debug_artifacts, extract_frames
from gemmaclip.gemma_client import (
    DEFAULT_PROVIDER_FIREWORKS_JUDGE,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_OPENROUTER,
    load_gemma_config,
)
from gemmaclip.io import Task, make_frame_manifest_entry, read_tasks, write_frame_manifest, write_results
from gemmaclip.validate import validate_results
from gemmaclip.video import probe_video

DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"
DEFAULT_WORKDIR = "/tmp/gemmaclip"
DEFAULT_MAX_RUNTIME_SECONDS = 570.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS = 15.0
MIN_NEXT_TASK_BUDGET_SECONDS = 45.0
MAX_NEXT_TASK_BUDGET_SECONDS = 90.0
FINAL_WRITE_BUFFER_SECONDS = 20.0

LOGGER = logging.getLogger("gemmaclip")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    start_time = time.monotonic()
    args = parse_args(argv)
    try:
        tasks = read_tasks(args.input)
        workdir = Path(args.workdir).resolve()
        debug_dir = Path(args.debug_dir).resolve() if args.debug_dir else None
        # Preserve a schema-valid result if a remote operation or the container watchdog stops the batch.
        write_results(build_progress_results(tasks, {}), args.output)
        process_tasks(
            tasks,
            workdir=workdir,
            output_path=args.output,
            debug_dir=debug_dir,
            dry_run=args.dry_run,
            frame_strategy=args.frame_strategy,
            max_runtime_seconds=args.max_runtime_seconds,
            start_time=start_time,
        )
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
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=DEFAULT_MAX_RUNTIME_SECONDS,
        help="Soft runtime budget used to switch remaining tasks to fallback captions before timeout.",
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
    env: Mapping[str, str] | None = None,
    remaining_seconds: float | None = None,
    remaining_time_fn: Callable[[], float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        values = env if env is not None else os.environ
        config = None if dry_run else load_gemma_config(values)
        use_google_fast_frames = bool(
            config is not None and config.provider in {DEFAULT_PROVIDER_GOOGLE, DEFAULT_PROVIDER_OPENROUTER}
        )
        use_fireworks_judge_frames = bool(
            config is not None and config.provider == DEFAULT_PROVIDER_FIREWORKS_JUDGE
        )

        video_path = download_video(
            task,
            destination_dir=workdir / "videos",
            max_duration_seconds=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        )
        metadata = probe_video(video_path, timeout_seconds=DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS)
        extracted_frames = extract_frames(
            task.task_id,
            video_path,
            metadata,
            strategy=frame_strategy,
            destination_root=workdir / "frames",
            google_fast=use_google_fast_frames,
            fireworks_judge=use_fireworks_judge_frames,
            command_timeout_seconds=DEFAULT_MEDIA_COMMAND_TIMEOUT_SECONDS,
            env=values,
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
            env=values,
            logger=LOGGER,
            remaining_seconds=remaining_seconds,
            remaining_time_fn=remaining_time_fn,
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


def process_tasks(
    tasks: Sequence[Task],
    *,
    workdir: Path,
    output_path: str | Path,
    debug_dir: Path | None = None,
    dry_run: bool = False,
    frame_strategy: str = DEFAULT_FRAME_STRATEGY,
    max_runtime_seconds: float = DEFAULT_MAX_RUNTIME_SECONDS,
    start_time: float | None = None,
    now_fn=time.monotonic,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    active_start_time = now_fn() if start_time is None else start_time
    runtime_deadline = active_start_time + max_runtime_seconds
    frame_manifest: list[dict[str, Any]] = []
    completed_results: dict[str, dict[str, Any]] = {}

    for task in tasks:
        remaining_task_count = len(tasks) - len(completed_results)
        if should_fill_remaining_with_fallbacks(
            active_start_time,
            completed_count=len(completed_results),
            remaining_task_count=remaining_task_count,
            max_runtime_seconds=max_runtime_seconds,
            now_fn=now_fn,
            logger=LOGGER,
        ):
            LOGGER.warning(
                "Runtime budget is low; filling remaining %s task(s) with fallback captions.",
                remaining_task_count,
            )
            progress_results = build_progress_results(tasks, completed_results)
            write_frame_manifest(frame_manifest, workdir / "frame_manifest.json")
            validate_results(tasks, progress_results)
            write_results(progress_results, output_path)
            return progress_results

        remaining_seconds = max(0.0, max_runtime_seconds - max(0.0, now_fn() - active_start_time))
        result, manifest_entry = process_task(
            task,
            workdir=workdir,
            debug_dir=debug_dir,
            dry_run=dry_run,
            frame_strategy=frame_strategy,
            env=env,
            remaining_seconds=remaining_seconds,
            remaining_time_fn=lambda: max(0.0, runtime_deadline - now_fn()),
        )
        completed_results[task.task_id] = result
        if manifest_entry is not None:
            frame_manifest.append(manifest_entry)

        progress_results = build_progress_results(tasks, completed_results)
        write_frame_manifest(frame_manifest, workdir / "frame_manifest.json")
        validate_results(tasks, progress_results)
        write_results(progress_results, output_path)

    final_results = build_progress_results(tasks, completed_results)
    write_frame_manifest(frame_manifest, workdir / "frame_manifest.json")
    validate_results(tasks, final_results)
    write_results(final_results, output_path)
    return final_results


def build_progress_results(
    tasks: Sequence[Task],
    completed_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        if task.task_id in completed_results:
            results.append(completed_results[task.task_id])
            continue
        results.append(
            {
                "task_id": task.task_id,
                "captions": build_fallback_captions(task.styles),
            }
        )
    return results


def should_fill_remaining_with_fallbacks(
    start_time: float,
    *,
    completed_count: int,
    remaining_task_count: int,
    max_runtime_seconds: float,
    now_fn=time.monotonic,
    logger: logging.Logger | None = None,
) -> bool:
    active_logger = logger or LOGGER
    if remaining_task_count <= 0:
        return False

    elapsed_seconds = max(0.0, now_fn() - start_time)
    remaining_seconds = max_runtime_seconds - elapsed_seconds
    next_task_budget = _estimate_next_task_budget(elapsed_seconds, completed_count)
    should_stop = remaining_seconds <= next_task_budget + FINAL_WRITE_BUFFER_SECONDS
    active_logger.info(
        "Runtime guard: elapsed=%.1fs remaining=%.1fs completed=%s remaining_tasks=%s next_task_budget=%.1fs fill_remaining=%s",
        elapsed_seconds,
        remaining_seconds,
        completed_count,
        remaining_task_count,
        next_task_budget,
        should_stop,
    )
    return should_stop


def _estimate_next_task_budget(elapsed_seconds: float, completed_count: int) -> float:
    if completed_count <= 0:
        return MIN_NEXT_TASK_BUDGET_SECONDS

    average_seconds = elapsed_seconds / completed_count
    return min(
        MAX_NEXT_TASK_BUDGET_SECONDS,
        max(MIN_NEXT_TASK_BUDGET_SECONDS, average_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
