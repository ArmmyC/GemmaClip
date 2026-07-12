from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

MIN_CAPTION_WORDS = 18
MAX_CAPTION_WORDS = 35

_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_AUDIO_ASSERTION_PATTERN = re.compile(
    r"\b(?:audio|sound|noise|music|song|speech|dialogue|conversation|talk(?:s|ed|ing)?|"
    r"speak(?:s|ing)?|say(?:s|ing)?|said|hear(?:s|d|ing)?|listen(?:s|ed|ing)?|"
    r"sing(?:s|ing)?|voice|voices)\b",
    re.IGNORECASE,
)
_REFUSAL_PATTERN = re.compile(
    r"\b(?:as an ai|i (?:cannot|can't|can not)|unable to|can't provide|cannot provide|"
    r"no caption|not enough information|i don't know)\b",
    re.IGNORECASE,
)
_MARKDOWN_PATTERN = re.compile(r"```|^\s*[-*+]\s+|^\s*#{1,6}\s+|\*\*|__|\[[^\]]+\]\([^)]+\)")
_PLACEHOLDER_PATTERN = re.compile(r"^\s*(?:<[^>]+>|caption|n/?a|none|null)\s*$", re.IGNORECASE)


class CaptionValidationError(ValueError):
    """Deterministic caption validation failure with a safe category."""

    def __init__(self, message: str, *, category: str = "invalid_caption") -> None:
        super().__init__(message)
        self.category = category


class DuplicateJSONKeyError(ValueError):
    pass


def normalize_caption(value: str) -> str:
    """Apply only formatting normalizations that cannot invent content."""

    cleaned = re.sub(r"\s+", " ", value.strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", cleaned)


def caption_word_count(value: str) -> int:
    return len(_WORD_PATTERN.findall(value))


def validate_caption_text(value: object) -> str:
    if not isinstance(value, str):
        raise CaptionValidationError("caption value is not a string", category="not_string")
    cleaned = normalize_caption(value)
    if not cleaned:
        raise CaptionValidationError("caption is empty", category="empty")
    if _PLACEHOLDER_PATTERN.fullmatch(cleaned):
        raise CaptionValidationError("caption is a placeholder", category="placeholder")
    if _MARKDOWN_PATTERN.search(cleaned):
        raise CaptionValidationError("caption contains markdown", category="markdown")
    if _REFUSAL_PATTERN.search(cleaned):
        raise CaptionValidationError("caption contains refusal boilerplate", category="refusal")
    if _AUDIO_ASSERTION_PATTERN.search(cleaned):
        raise CaptionValidationError("caption contains an unsupported audio assertion", category="audio_assertion")
    if len(_WORD_PATTERN.findall(cleaned)) < 3:
        raise CaptionValidationError("caption contains too little alphabetic content", category="alphabetic_content")
    word_count = caption_word_count(cleaned)
    if word_count < MIN_CAPTION_WORDS:
        raise CaptionValidationError("caption is below the minimum word count", category="too_short")
    if word_count > MAX_CAPTION_WORDS:
        raise CaptionValidationError("caption is above the maximum word count", category="too_long")
    return cleaned


def validate_caption_payload(
    payload: object,
    styles: Sequence[str],
    *,
    reject_unrequested_styles: bool = True,
) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise CaptionValidationError("response is not a JSON object", category="not_object")

    requested = tuple(styles)
    if not requested:
        raise CaptionValidationError("no requested styles", category="no_styles")

    known_style_keys = {"formal", "sarcastic", "humorous_tech", "humorous_non_tech"}
    missing = [style for style in requested if style not in payload]
    if missing:
        raise CaptionValidationError(f"missing requested style: {missing[0]}", category="missing_style")

    if reject_unrequested_styles:
        extra_styles = sorted(key for key in payload if key in known_style_keys and key not in requested)
        if extra_styles:
            raise CaptionValidationError("response contains an unrequested style", category="extra_style")

    result: dict[str, str] = {}
    for style in requested:
        result[style] = validate_caption_text(payload[style])

    _reject_exact_duplicates(result)
    return result


def extract_valid_partial_captions(payload: object, styles: Sequence[str]) -> dict[str, str]:
    """Keep independently valid requested styles after an incomplete response."""

    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, str] = {}
    normalized_seen: set[str] = set()
    for style in styles:
        if style not in payload:
            continue
        try:
            cleaned = validate_caption_text(payload[style])
        except CaptionValidationError:
            continue
        comparison = cleaned.casefold()
        if comparison in normalized_seen:
            continue
        normalized_seen.add(comparison)
        result[style] = cleaned
    return result


def validate_review_payload(payload: object, styles: Sequence[str]) -> dict[str, str]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("captions"), Mapping):
        raise CaptionValidationError("review response has no captions object", category="review_shape")
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        raise CaptionValidationError("review response has no scores object", category="review_scores")
    for style in styles:
        score = scores.get(style)
        if not isinstance(score, Mapping):
            raise CaptionValidationError("review score is missing", category="review_scores")
        for name in ("accuracy", "style_match"):
            value = score.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise CaptionValidationError("review score is invalid", category="review_scores")
    return validate_caption_payload(payload["captions"], styles)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        parsed = json.loads(cleaned, object_pairs_hook=_object_pairs_without_duplicates)
    except DuplicateJSONKeyError as exc:
        raise CaptionValidationError("JSON contains duplicate keys", category="duplicate_json_key") from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise CaptionValidationError("response is not valid JSON", category="malformed_json") from exc
    if not isinstance(parsed, dict):
        raise CaptionValidationError("response is not a JSON object", category="not_object")
    return parsed


def _object_pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _reject_exact_duplicates(captions: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for style, caption in captions.items():
        comparison = caption.casefold()
        if comparison in seen:
            raise CaptionValidationError(
                f"caption duplicates {seen[comparison]}",
                category="duplicate_caption",
            )
        seen[comparison] = style
