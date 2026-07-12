from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gemmaclip.frames import ExtractedFrame
from gemmaclip.io import Task, safe_task_id
from gemmaclip.leaderboard.config import FireworksLeaderboardConfig, load_fireworks_leaderboard_config
from gemmaclip.leaderboard.fireworks import (
    FireworksLeaderboardClient,
    FireworksLeaderboardRequestError,
    FireworksLeaderboardRuntimeError,
)
from gemmaclip.leaderboard.prompts import (
    build_generation_messages,
    build_repair_messages,
    build_review_messages,
)
from gemmaclip.leaderboard.validation import (
    CaptionValidationError,
    validate_caption_payload,
    validate_review_payload,
)

LOGGER = logging.getLogger("gemmaclip.leaderboard")


def generate_fireworks_leaderboard_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    config: FireworksLeaderboardConfig | None = None,
    env: Mapping[str, str] | None = None,
    remaining_seconds: float | None = None,
    remaining_time_fn: Callable[[], float] | None = None,
    debug_dir: str | Path | None = None,
    client_factory: Callable[[FireworksLeaderboardConfig], Any] | None = None,
    allow_remote: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    active_logger = logger or LOGGER
    active_config = config if config is not None else load_fireworks_leaderboard_config(env)
    if active_config is None or not active_config.is_configured or not allow_remote:
        active_logger.info(
            "Task %s Fireworks leaderboard provider is unconfigured; using deterministic fallback.",
            task.task_id,
        )
        return build_leaderboard_fallback_captions(task.styles)

    selected_frames = select_leaderboard_frames(frames)
    if len(selected_frames) != 6:
        active_logger.warning("Task %s did not provide exactly six frames; using deterministic fallback.", task.task_id)
        return build_leaderboard_fallback_captions(task.styles)

    live_remaining = _make_remaining_time_fn(remaining_seconds, remaining_time_fn)
    client = client_factory(active_config) if client_factory is not None else FireworksLeaderboardClient(active_config)
    debug_events: list[dict[str, Any]] = []
    valid_captions: dict[str, str] = {}
    initial_fallback_eligible = False
    initial_safe_to_repair = True

    for index, model in enumerate(active_config.model_order):
        if index > 0 and not initial_fallback_eligible:
            break
        if not _has_time(live_remaining, active_config.min_generation_remaining_seconds):
            initial_safe_to_repair = False
            break
        try:
            result = client.complete_json(
                build_generation_messages(task.task_id, task.styles, selected_frames),
                model=model,
                temperature=active_config.generation_temperature,
                validator=lambda payload: validate_caption_payload(payload, task.styles),
                remaining_time_fn=live_remaining,
                minimum_remaining_seconds=active_config.min_generation_remaining_seconds,
                operation="generation",
                styles=task.styles,
                debug_callback=debug_events.append,
            )
            valid_captions = _merge_missing(valid_captions, result, task.styles)
            if _has_all_styles(valid_captions, task.styles):
                break
            initial_fallback_eligible = True
        except FireworksLeaderboardRuntimeError:
            initial_safe_to_repair = False
            break
        except FireworksLeaderboardRequestError as exc:
            valid_captions = _merge_missing(valid_captions, exc.partial_captions, task.styles)
            initial_fallback_eligible = exc.fallback_eligible
            initial_safe_to_repair = exc.fallback_eligible
            if not exc.fallback_eligible:
                break
        except Exception as exc:
            # Test doubles and alternate clients can surface a plain exception;
            # treat it as a retryable network/provider failure without logging it.
            valid_captions = _merge_missing(valid_captions, {}, task.styles)
            initial_fallback_eligible = _looks_retryable(exc)
            initial_safe_to_repair = initial_fallback_eligible
            if not initial_fallback_eligible:
                break

    if initial_safe_to_repair:
        missing = [style for style in task.styles if style not in valid_captions]
        if missing:
            valid_captions = _repair_missing_styles(
                task,
                selected_frames,
                valid_captions,
                missing,
                client,
                active_config,
                live_remaining,
                debug_events,
            )

    # Fill only styles that remain missing.  Already valid generated captions
    # are retained byte-for-byte and are never replaced by this fallback.
    review_eligible = _has_all_styles(valid_captions, task.styles)
    fallback = build_leaderboard_fallback_captions(task.styles)
    result = _merge_missing(valid_captions, fallback, task.styles)
    result = _ensure_final_validity(result, task.styles)

    if active_config.enable_review and review_eligible and _has_all_styles(result, task.styles):
        if _has_time(live_remaining, active_config.min_review_remaining_seconds):
            reviewed = _review_captions(
                task,
                selected_frames,
                result,
                client,
                active_config,
                live_remaining,
                debug_events,
            )
            if reviewed is not None:
                result = _ensure_final_validity(reviewed, task.styles, preserve=result)
        else:
            active_logger.info("Task %s skipping independent review because the live budget is below %.1fs.", task.task_id, active_config.min_review_remaining_seconds)

    if debug_dir is not None:
        _write_debug_summary(task, selected_frames, debug_events, debug_dir, live_remaining)
    return {style: result[style] for style in task.styles}


def select_leaderboard_frames(frames: Sequence[ExtractedFrame]) -> list[ExtractedFrame]:
    """Return six separate frames in chronological order.

    The main CLI's Fireworks extractor already returns six frames.  The
    deterministic down-selection makes this helper safe for callers and test
    doubles that provide more frames, while refusing to synthesize frames.
    """

    ordered = sorted(frames, key=lambda frame: (frame.timestamp_seconds, str(frame.path)))
    if len(ordered) < 6:
        return []
    if len(ordered) == 6:
        return ordered
    last_index = len(ordered) - 1
    indices = [round(last_index * ratio) for ratio in (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)]
    return [ordered[index] for index in indices]


def build_leaderboard_fallback_captions(styles: Sequence[str]) -> dict[str, str]:
    templates = {
        "formal": "The video could not be fully processed, so this fallback caption records a short scene with visible activity and no additional assumptions.",
        "sarcastic": "The video declined to reveal its full story, delivering a remarkably efficient demonstration of how mystery can remain professionally unhelpful.",
        "humorous_tech": "The captioning pipeline encountered a processing snag, so this fallback keeps the visible scene grounded while avoiding unsupported details.",
        "humorous_non_tech": "The video kept some details private, so this fallback politely describes a visible moment while letting the missing context stay offstage.",
    }
    return {style: templates[style] for style in styles}


def _repair_missing_styles(
    task: Task,
    frames: Sequence[ExtractedFrame],
    valid_captions: dict[str, str],
    missing_styles: Sequence[str],
    client: Any,
    config: FireworksLeaderboardConfig,
    remaining_time_fn: Callable[[], float],
    debug_events: list[dict[str, Any]],
) -> dict[str, str]:
    merged = dict(valid_captions)
    outstanding = list(missing_styles)
    for model in config.model_order:
        if not outstanding or not _has_time(remaining_time_fn, config.min_generation_remaining_seconds):
            break
        try:
            repaired = client.complete_json(
                build_repair_messages(task.task_id, outstanding, merged, frames),
                model=model,
                temperature=config.repair_temperature,
                validator=lambda payload, requested=tuple(outstanding): validate_caption_payload(
                    payload, requested, reject_unrequested_styles=True
                ),
                remaining_time_fn=remaining_time_fn,
                minimum_remaining_seconds=config.min_generation_remaining_seconds,
                operation="repair",
                styles=tuple(outstanding),
                debug_callback=debug_events.append,
            )
            merged = _merge_missing(merged, repaired, task.styles)
            outstanding = [style for style in outstanding if style not in merged]
        except FireworksLeaderboardRuntimeError:
            break
        except FireworksLeaderboardRequestError as exc:
            merged = _merge_missing(merged, exc.partial_captions, task.styles)
            outstanding = [style for style in outstanding if style not in merged]
            if not exc.fallback_eligible:
                break
        except Exception as exc:
            if not _looks_retryable(exc):
                break
    return merged


def _review_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    captions: dict[str, str],
    client: Any,
    config: FireworksLeaderboardConfig,
    remaining_time_fn: Callable[[], float],
    debug_events: list[dict[str, Any]],
) -> dict[str, str] | None:
    for index, model in enumerate(config.review_model_order):
        if not _has_time(remaining_time_fn, config.min_review_remaining_seconds):
            return None
        try:
            return client.complete_json(
                build_review_messages(task.task_id, task.styles, frames, captions),
                model=model,
                temperature=config.review_temperature,
                validator=lambda payload: validate_review_payload(payload, task.styles),
                remaining_time_fn=remaining_time_fn,
                minimum_remaining_seconds=config.min_review_remaining_seconds,
                operation="review",
                styles=task.styles,
                debug_callback=debug_events.append,
            )
        except FireworksLeaderboardRuntimeError:
            return None
        except FireworksLeaderboardRequestError as exc:
            if not exc.fallback_eligible:
                return None
            if index == 1:
                return None
        except Exception as exc:
            if not _looks_retryable(exc) or index == 1:
                return None
    return None


def _merge_missing(
    current: Mapping[str, str],
    additions: Mapping[str, str] | object,
    styles: Sequence[str],
) -> dict[str, str]:
    result = dict(current)
    if not isinstance(additions, Mapping):
        return result
    for style in styles:
        value = additions.get(style)
        if style in result or not isinstance(value, str):
            continue
        try:
            result[style] = validate_caption_text_for_merge(value, result)
        except CaptionValidationError:
            continue
    return result


def validate_caption_text_for_merge(value: str, existing: Mapping[str, str]) -> str:
    from gemmaclip.leaderboard.validation import validate_caption_text

    cleaned = validate_caption_text(value)
    if any(cleaned.casefold() == other.casefold() for other in existing.values()):
        raise CaptionValidationError("duplicate caption", category="duplicate_caption")
    return cleaned


def _ensure_final_validity(
    captions: Mapping[str, str],
    styles: Sequence[str],
    *,
    preserve: Mapping[str, str] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for style in styles:
        candidate = captions.get(style)
        if isinstance(candidate, str):
            try:
                cleaned = validate_caption_text_for_merge(candidate, result)
                result[style] = cleaned
                continue
            except CaptionValidationError:
                pass
        if preserve and isinstance(preserve.get(style), str):
            try:
                result[style] = validate_caption_text_for_merge(preserve[style], result)
                continue
            except CaptionValidationError:
                pass
        fallback = build_leaderboard_fallback_captions((style,))[style]
        if any(fallback.casefold() == existing.casefold() for existing in result.values()):
            fallback = f"{fallback} This wording remains deliberately non-specific."
        result[style] = validate_caption_text_for_merge(fallback, result)
    return result


def _has_all_styles(captions: Mapping[str, str], styles: Sequence[str]) -> bool:
    return all(style in captions for style in styles)


def _has_time(remaining_time_fn: Callable[[], float], threshold: float) -> bool:
    return max(0.0, float(remaining_time_fn())) >= threshold


def _looks_retryable(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, ValueError, TypeError, KeyError)) or exc.__class__.__module__.startswith("httpx")


def _make_remaining_time_fn(
    remaining_seconds: float | None,
    remaining_time_fn: Callable[[], float] | None,
) -> Callable[[], float]:
    if remaining_time_fn is not None:
        return remaining_time_fn
    if remaining_seconds is None:
        return lambda: float("inf")
    started_at = time.monotonic()
    return lambda: max(0.0, remaining_seconds - (time.monotonic() - started_at))


def _write_debug_summary(
    task: Task,
    frames: Sequence[ExtractedFrame],
    events: Sequence[Mapping[str, Any]],
    debug_dir: str | Path,
    remaining_time_fn: Callable[[], float],
) -> None:
    path = Path(debug_dir) / f"{safe_task_id(task.task_id)}_fireworks_leaderboard_debug.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_events = [
        {
            key: value
            for key, value in event.items()
            if key in {"operation", "model", "status", "status_code", "category", "elapsed_seconds", "word_counts"}
        }
        for event in events
    ]
    path.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "requested_styles": list(task.styles),
                "selected_frame_timestamps": [frame.timestamp_seconds for frame in frames],
                "remaining_seconds_at_summary": round(max(0.0, remaining_time_fn()), 3),
                "events": safe_events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
