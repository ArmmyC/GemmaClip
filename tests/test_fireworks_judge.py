from __future__ import annotations

import httpx
import pytest

from gemmaclip.captioner import (
    _normalize_exact_caption_keys,
    _validate_fireworks_judge_review,
    build_fireworks_judge_generation_messages,
    generate_captions,
)
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_FIREWORKS_JUDGE_FALLBACK_VISION_MODEL,
    DEFAULT_FIREWORKS_JUDGE_VISION_MODEL,
    DEFAULT_PROVIDER_FIREWORKS_JUDGE,
    FireworksJudgeConfig,
    FireworksVisionClient,
    load_gemma_config,
)
from gemmaclip.io import Task


def make_task() -> Task:
    return Task("clip-1", "https://example.com/video.mp4", ("formal", "sarcastic"))


def make_frames(tmp_path, count: int = 6) -> list[ExtractedFrame]:
    from PIL import Image

    frames: list[ExtractedFrame] = []
    for index in range(count):
        path = tmp_path / f"frame_{index + 1:03d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 288), color=(index * 30, 80, 120)).save(path, format="JPEG", quality=95)
        frames.append(ExtractedFrame(path=path, timestamp_seconds=float(index + 1)))
    return frames


def initial_captions() -> dict[str, str]:
    return {
        "formal": "A worker stands beside a desk in an office while looking toward a computer monitor.",
        "sarcastic": "A worker stands by a desk and monitor, delivering the kind of office suspense nobody ordered.",
    }


def review_payload(captions: dict[str, str]) -> dict[str, object]:
    return {
        "scores": {
            "formal": {"accuracy": 0.95, "style_match": 0.95},
            "sarcastic": {"accuracy": 0.92, "style_match": 0.91},
        },
        "captions": captions,
    }


def test_fireworks_judge_configuration_uses_minimax_defaults_and_custom_models():
    default_config = load_gemma_config(
        {"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"}
    )
    custom_config = load_gemma_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_judge",
            "FIREWORKS_API_KEY": "key",
            "FIREWORKS_VISION_MODEL": "accounts/fireworks/models/future-gemma",
            "FIREWORKS_FALLBACK_VISION_MODEL": "accounts/fireworks/models/custom-fallback",
        }
    )

    assert isinstance(default_config, FireworksJudgeConfig)
    assert default_config.provider == DEFAULT_PROVIDER_FIREWORKS_JUDGE
    assert default_config.vision_model == DEFAULT_FIREWORKS_JUDGE_VISION_MODEL
    assert default_config.fallback_vision_model == DEFAULT_FIREWORKS_JUDGE_FALLBACK_VISION_MODEL
    assert custom_config is not None
    assert custom_config.vision_model == "accounts/fireworks/models/future-gemma"
    assert custom_config.fallback_vision_model == "accounts/fireworks/models/custom-fallback"


def test_fireworks_judge_messages_send_six_separate_base64_images_without_contact_sheet(tmp_path):
    messages = build_fireworks_judge_generation_messages("clip-1", ("formal", "sarcastic"), make_frames(tmp_path))
    content = messages[1]["content"]
    image_parts = [part for part in content if part.get("type") == "image_url"]

    assert len(image_parts) == 6
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in image_parts)
    assert all(part.get("type") != "image_file" for part in content)
    assert "separate chronological video frames" in content[0]["text"]


def test_fireworks_vision_retries_primary_on_429_and_selects_primary_model():
    class RetryingClient:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            request = httpx.Request("POST", url, json=json, headers=headers)
            if len(self.models) == 1:
                return httpx.Response(429, request=request, json={"error": "busy"})
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": '{"formal":"ok"}'}}]})

    http_client = RetryingClient()
    client = FireworksVisionClient(
        FireworksJudgeConfig("secret", "https://example.test/v1", "primary-model", "fallback-model"),
        client=http_client,
        sleeper=lambda delay: None,
    )
    result = client.complete_json(
        [{"role": "user", "content": "hello"}],
        temperature=0.6,
        validator=lambda payload: payload if set(payload) == {"formal"} else (_ for _ in ()).throw(ValueError()),
    )

    assert result == {"formal": "ok"}
    assert http_client.models == ["primary-model", "primary-model"]


def test_fireworks_vision_retries_primary_on_5xx_then_falls_back_after_invalid_json():
    class SequencedClient:
        def __init__(self):
            self.models: list[str] = []
            self.responses = [500, 503, 200]

        def post(self, url, headers, json):
            self.models.append(json["model"])
            request = httpx.Request("POST", url, json=json, headers=headers)
            status = self.responses.pop(0)
            if status != 200:
                return httpx.Response(status, request=request, json={"error": "retry"})
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": '{"formal":"ok"}'}}]})

    http_client = SequencedClient()
    client = FireworksVisionClient(
        FireworksJudgeConfig("secret", "https://example.test/v1", "primary-model", "fallback-model"),
        client=http_client,
        sleeper=lambda delay: None,
    )

    assert client.complete_json([{"role": "user", "content": "hello"}], temperature=0.6, validator=lambda payload: payload) == {
        "formal": "ok"
    }
    assert http_client.models == ["primary-model", "primary-model", "fallback-model"]


def test_fireworks_vision_invalid_primary_json_triggers_configured_fallback():
    class InvalidThenFallbackClient:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            request = httpx.Request("POST", url, json=json, headers=headers)
            content = "not json" if json["model"] == "primary-model" else '{"formal":"ok"}'
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": content}}]})

    http_client = InvalidThenFallbackClient()
    client = FireworksVisionClient(
        FireworksJudgeConfig("secret", "https://example.test/v1", "primary-model", "fallback-model"),
        client=http_client,
        sleeper=lambda delay: None,
    )

    assert client.complete_json([{"role": "user", "content": "hello"}], temperature=0.6, validator=lambda payload: payload) == {
        "formal": "ok"
    }
    assert http_client.models == ["primary-model", "primary-model", "fallback-model"]


class ScriptedJudgeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages: list[list[dict[str, object]]] = []

    def complete_json(self, messages, *, temperature, validator, **kwargs):
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return validator(response)


def test_fireworks_judge_keeps_strong_captions_and_uses_requested_keys_only(tmp_path):
    first = initial_captions()
    scripted = ScriptedJudgeClient([first, review_payload(first)])
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: scripted,
        remaining_seconds=300.0,
    )

    assert captions == first
    assert set(captions) == {"formal", "sarcastic"}
    assert len(scripted.messages) == 2
    assert all(len([part for part in call[1]["content"] if part.get("type") == "image_url"]) == 6 for call in scripted.messages)


def test_fireworks_judge_replaces_weak_caption_and_preserves_first_when_review_is_invalid_or_fails(tmp_path):
    first = initial_captions()
    improved = dict(first, sarcastic="A worker stands by a desk and monitor, proving office drama can survive on remarkably little movement.")

    rewritten = generate_captions(
        make_task(), make_frames(tmp_path / "rewritten"),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: ScriptedJudgeClient([first, review_payload(improved)]), remaining_seconds=300.0,
    )
    invalid_review = generate_captions(
        make_task(), make_frames(tmp_path / "invalid"),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: ScriptedJudgeClient([first, {"captions": first}]), remaining_seconds=300.0,
    )
    failed_review = generate_captions(
        make_task(), make_frames(tmp_path / "failed"),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: ScriptedJudgeClient([first, httpx.ReadTimeout("timeout")]), remaining_seconds=300.0,
    )

    assert rewritten == improved
    assert invalid_review == first
    assert failed_review == first


def test_fireworks_judge_low_time_skips_review_and_client_logs_do_not_expose_secrets_or_base64(tmp_path, caplog):
    first = initial_captions()
    scripted = ScriptedJudgeClient([first])
    captions = generate_captions(
        make_task(), make_frames(tmp_path),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: scripted, remaining_seconds=149.0,
    )

    class RecordingClient:
        def post(self, url, headers, json):
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": '{"formal":"ok"}'}}]})

    caplog.set_level("INFO")
    secret = "do-not-log-this-key"
    encoded_image = "data:image/jpeg;base64,do-not-log-this-image"
    client = FireworksVisionClient(
        FireworksJudgeConfig(secret, "https://example.test/v1", "primary-model", "fallback-model"),
        client=RecordingClient(),
    )
    client.complete_json(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encoded_image}}]}],
        temperature=0.6,
        validator=lambda payload: payload,
    )

    assert captions == first
    assert len(scripted.messages) == 1
    assert secret not in caplog.text
    assert encoded_image not in caplog.text


def test_fireworks_judge_recalculates_runtime_after_generation_and_skips_review(tmp_path):
    first = initial_captions()
    remaining = {"seconds": 151.0}

    class GenerationConsumesBudget:
        def __init__(self):
            self.calls = 0

        def complete_json(self, messages, *, temperature, validator, **kwargs):
            self.calls += 1
            remaining["seconds"] = 149.0
            return validator(first)

    client = GenerationConsumesBudget()
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: client,
        remaining_time_fn=lambda: remaining["seconds"],
    )

    assert captions == first
    assert client.calls == 1


def test_fireworks_vision_skips_primary_retry_when_deadline_becomes_too_close():
    remaining = {"seconds": 100.0}

    class RetryWouldExceedBudget:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            remaining["seconds"] = 64.0
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(429, request=request, json={"error": "busy"})

    http_client = RetryWouldExceedBudget()
    client = FireworksVisionClient(
        FireworksJudgeConfig("secret", "https://example.test/v1", "primary-model", "fallback-model"),
        client=http_client,
        sleeper=lambda delay: None,
    )

    with pytest.raises(RuntimeError, match="Fireworks vision request failed"):
        client.complete_json(
            [{"role": "user", "content": "hello"}],
            temperature=0.6,
            validator=lambda payload: payload,
            remaining_time_fn=lambda: remaining["seconds"],
            minimum_remaining_seconds=65.0,
        )

    assert http_client.models == ["primary-model"]


def test_fireworks_vision_skips_fallback_model_when_deadline_becomes_too_close():
    remaining = {"seconds": 100.0}

    class PrimaryFailureConsumesBudget:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            remaining["seconds"] = 64.0
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(400, request=request, json={"error": "bad request"})

    http_client = PrimaryFailureConsumesBudget()
    client = FireworksVisionClient(
        FireworksJudgeConfig("secret", "https://example.test/v1", "primary-model", "fallback-model"),
        client=http_client,
        sleeper=lambda delay: None,
    )

    with pytest.raises(RuntimeError, match="Fireworks vision request failed"):
        client.complete_json(
            [{"role": "user", "content": "hello"}],
            temperature=0.6,
            validator=lambda payload: payload,
            remaining_time_fn=lambda: remaining["seconds"],
            minimum_remaining_seconds=65.0,
        )

    assert http_client.models == ["primary-model"]


def test_fireworks_judge_preserves_valid_first_captions_when_review_budget_expires(tmp_path):
    first = initial_captions()
    remaining = {"seconds": 200.0}

    class ReviewBudgetExpires:
        def __init__(self):
            self.calls = 0

        def complete_json(self, messages, *, temperature, validator, **kwargs):
            self.calls += 1
            remaining["seconds"] = 64.0
            return validator(first)

    client = ReviewBudgetExpires()
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: client,
        remaining_time_fn=lambda: remaining["seconds"],
    )

    assert captions == first
    assert client.calls == 1


def test_fireworks_generation_extracts_only_requested_style_from_all_four_styles():
    captions = _normalize_exact_caption_keys(
        {
            "formal": "A worker stands beside a desk in an office while looking toward a monitor.",
            "sarcastic": "A worker stands beside a desk, delivering office excitement at its most restrained.",
            "humorous_tech": "A worker stands beside a desk while the office buffer handles another quiet update.",
            "humorous_non_tech": "A worker stands beside a desk while the day practices its smallest joke.",
        },
        ("formal",),
    )

    assert captions == {"formal": "A worker stands beside a desk in an office while looking toward a monitor."}


def test_fireworks_generation_accepts_wrapped_captions_metadata_and_style_aliases():
    wrapped = _normalize_exact_caption_keys(
        {
            "captions": {
                "Humorous-Tech": "A worker sits at a desk while the office CPU quietly handles another routine task.",
                "notes": "safe metadata",
            },
            "request_id": "ignored",
        },
        ("humorous_tech",),
    )
    non_tech = _normalize_exact_caption_keys(
        {"humorous non tech": "A worker sits at a desk while the day rehearses one small office joke."},
        ("humorous_non_tech",),
    )

    assert set(wrapped) == {"humorous_tech"}
    assert set(non_tech) == {"humorous_non_tech"}


def test_fireworks_generation_rejects_missing_requested_style():
    with pytest.raises(ValueError, match="missing requested style: formal"):
        _normalize_exact_caption_keys({"notes": "no caption"}, ("formal",))


def test_fireworks_review_accepts_extra_styles_and_missing_scores(caplog):
    first = initial_captions()
    reviewed = _validate_fireworks_judge_review(
        {
            "captions": {
                **first,
                "humorous-tech": "An extra unrequested caption is harmless here because it is not requested.",
            },
            "scores": {
                "formal": {"accuracy": 0.9, "style_match": 0.9},
                "sarcastic": {"accuracy": 0.8, "style_match": 0.8},
                "humorous_tech": {"accuracy": 0.7, "style_match": 0.7},
            },
            "metadata": "ignored",
        },
        ("formal", "sarcastic"),
    )
    caplog.set_level("WARNING")
    without_scores = _validate_fireworks_judge_review({"captions": first}, ("formal", "sarcastic"))

    assert reviewed == first
    assert without_scores == first
    assert "review scores unavailable" in caplog.text


def test_fireworks_invalid_review_captions_preserve_first_generation(tmp_path):
    first = initial_captions()
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": "key"},
        client_factory=lambda config: ScriptedJudgeClient([first, {"captions": {"formal": first["formal"]}}]),
        remaining_seconds=300.0,
    )

    assert captions == first


def test_fireworks_validation_diagnostics_are_sanitized_and_debug_response_is_saved(tmp_path, caplog):
    secret = "do-not-log-this-key"
    base64_data = "data:image/jpeg;base64,do-not-log-this-image"

    class InvalidResponseClient:
        def post(self, url, headers, json):
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": "not json"}}]})

    caplog.set_level("WARNING")
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path / "frames"),
        env={"GEMMACLIP_PROVIDER": "fireworks_judge", "FIREWORKS_API_KEY": secret},
        client_factory=lambda config: FireworksVisionClient(config, client=InvalidResponseClient(), sleeper=lambda delay: None),
        debug_dir=tmp_path / "debug",
        remaining_seconds=300.0,
    )

    debug_files = list((tmp_path / "debug").glob("clip-1_fireworks_generation_attempt_*.txt"))
    assert captions["formal"].startswith("The video could not")
    assert debug_files
    assert debug_files[0].read_text(encoding="utf-8") == "not json"
    assert "operation=generation" in caplog.text
    assert "message=no JSON object found" in caplog.text
    assert secret not in caplog.text
    assert base64_data not in caplog.text
