from __future__ import annotations

import json as json_module
import logging
from pathlib import Path

import httpx
import pytest

from gemmaclip.frames import ExtractedFrame
from gemmaclip.io import Task
from gemmaclip.leaderboard.config import (
    DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL,
    DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
    DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL,
    load_fireworks_leaderboard_config,
)
from gemmaclip.leaderboard.fireworks import (
    FireworksLeaderboardClient,
    FireworksLeaderboardRequestError,
)
from gemmaclip.leaderboard.pipeline import (
    build_leaderboard_fallback_captions,
    generate_fireworks_leaderboard_captions,
)
from gemmaclip.leaderboard.prompts import build_generation_messages, build_review_messages
from gemmaclip.leaderboard.validation import (
    CaptionValidationError,
    caption_word_count,
    parse_json_object,
    validate_caption_payload,
)
from gemmaclip.web.services import WebServices
from gemmaclip.video import VideoMetadata


def make_task(styles=("formal", "sarcastic", "humorous_tech", "humorous_non_tech")) -> Task:
    return Task("clip-1", "https://example.test/clip.mp4", tuple(styles))


def make_frames(tmp_path: Path) -> list[ExtractedFrame]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(6):
        path = tmp_path / f"frame_{index + 1:03d}.jpg"
        path.write_bytes(f"jpeg-{index}".encode())
        frames.append(ExtractedFrame(path, float(index + 1)))
    return frames


def valid_captions(styles=make_task().styles) -> dict[str, str]:
    values = {
        "formal": "A person moves a large box across an indoor room while the surrounding furniture remains visible throughout the sequence.",
        "sarcastic": "A person moves a large box across an indoor room, apparently demonstrating that ordinary errands deserve their own carefully managed procession.",
        "humorous_tech": "A person moves a large box across an indoor room while the plan keeps buffering like an update that refuses to reach completion.",
        "humorous_non_tech": "A person moves a large box across an indoor room, turning a simple task into the kind of parade nobody scheduled.",
    }
    return {style: values[style] for style in styles}


def test_configuration_defaults_custom_values_and_secret_safe_repr():
    config = load_fireworks_leaderboard_config(
        {"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "secret-key"}
    )
    assert config is not None and config.is_configured
    assert config.generation_model == DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL
    assert config.review_model == DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL
    assert config.fallback_model == DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL
    assert "secret-key" not in repr(config)

    custom = load_fireworks_leaderboard_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_leaderboard",
            "FIREWORKS_API_KEY": "secret-key",
            "FIREWORKS_LEADERBOARD_GENERATION_MODEL": "custom-generation",
            "FIREWORKS_LEADERBOARD_REVIEW_MODEL": "custom-review",
            "FIREWORKS_LEADERBOARD_FALLBACK_MODEL": "custom-fallback",
            "FIREWORKS_LEADERBOARD_GENERATION_TEMPERATURE": "3.5",
            "FIREWORKS_LEADERBOARD_REPAIR_TEMPERATURE": "bad",
            "FIREWORKS_LEADERBOARD_REVIEW_TEMPERATURE": "-1",
            "FIREWORKS_LEADERBOARD_ENABLE_REVIEW": "off",
        }
    )
    assert custom is not None
    assert custom.generation_model == "custom-generation"
    assert custom.review_model == "custom-review"
    assert custom.fallback_model == "custom-fallback"
    assert custom.generation_temperature == 2.0
    assert custom.repair_temperature == 0.2
    assert custom.review_temperature == 0.0
    assert custom.enable_review is False

    unconfigured = load_fireworks_leaderboard_config({"GEMMACLIP_PROVIDER": "fireworks_leaderboard"})
    assert unconfigured is not None and not unconfigured.is_configured


def test_generation_request_has_six_separate_jpegs_timestamps_schema_and_json_mode(tmp_path):
    frames = make_frames(tmp_path)
    messages = build_generation_messages("clip-1", ("formal", "sarcastic"), frames)
    content = messages[1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 6
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in image_parts)
    text = content[0]["text"]
    assert "timestamp_seconds=1.000" in text
    assert text.index("timestamp_seconds=1.000") < text.index("timestamp_seconds=6.000")
    assert '"formal": "<18-35 word caption>"' in text
    assert "contact sheet" not in text.lower()

    captured = {}

    class RecordingHTTP:
        def post(self, url, headers, json):
            captured.update(json)
            request = httpx.Request("POST", url, headers=headers, json=json)
            return httpx.Response(200, request=request, json={"choices": [{"message": {"content": '{"formal":"ok"}'}}]})

    config = load_fireworks_leaderboard_config(
        {"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "secret-key"}
    )
    assert config is not None
    client = FireworksLeaderboardClient(config, client=RecordingHTTP())
    assert client.complete_json(messages, model="model", temperature=0.3, validator=lambda value: value) == {"formal": "ok"}
    assert captured["response_format"] == {"type": "json_object"}
    assert len([part for part in captured["messages"][1]["content"] if part["type"] == "image_url"]) == 6
    assert "secret-key" not in json_module.dumps(captured)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, float]] = []

    def complete_json(self, messages, *, model, temperature, validator, operation, **kwargs):
        self.calls.append((model, operation, temperature))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return validator(response)


def test_generation_uses_qwen_then_minimax_after_retryable_invalid_primary(tmp_path):
    client = ScriptedClient([
        FireworksLeaderboardRequestError("invalid", retryable=True, category="malformed_json"),
        valid_captions(),
    ])
    config = load_fireworks_leaderboard_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_leaderboard",
            "FIREWORKS_API_KEY": "key",
            "FIREWORKS_LEADERBOARD_ENABLE_REVIEW": "false",
        }
    )
    result = generate_fireworks_leaderboard_captions(
        make_task(), make_frames(tmp_path), config=config, client_factory=lambda _: client, remaining_seconds=200
    )
    assert result == valid_captions()
    assert [model for model, operation, _ in client.calls] == [
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL,
    ]


class HTTPSequence:
    def __init__(self, responses):
        self.responses = list(responses)
        self.models: list[str] = []

    def post(self, url, headers, json):
        self.models.append(json["model"])
        status, payload = self.responses.pop(0)
        request = httpx.Request("POST", url, headers=headers, json=json)
        if status != 200:
            return httpx.Response(status, request=request, json={"error": "provider failure"})
        return httpx.Response(
            status,
            request=request,
            json={"choices": [{"message": {"content": json_module.dumps(payload)}}]},
        )


def test_generation_404_uses_configured_fallback_model(tmp_path):
    captions = valid_captions(("formal", "sarcastic"))
    http_client = HTTPSequence([(404, None), (200, captions)])
    config = load_fireworks_leaderboard_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_leaderboard",
            "FIREWORKS_API_KEY": "key",
            "FIREWORKS_LEADERBOARD_ENABLE_REVIEW": "false",
        }
    )
    assert config is not None
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal", "sarcastic")),
        make_frames(tmp_path),
        config=config,
        client_factory=lambda value: FireworksLeaderboardClient(value, client=http_client),
        remaining_seconds=149,
    )
    assert result == captions
    assert http_client.models == [
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL,
    ]


def test_review_404_uses_qwen_review_fallback(tmp_path):
    captions = valid_captions(("formal", "sarcastic"))
    reviewed = dict(captions, sarcastic="A person moves a large box indoors while the supposedly simple task turns into a carefully choreographed performance for everyone nearby.")
    review_payload = {
        "scores": {
            style: {"accuracy": 0.9, "style_match": 0.9}
            for style in captions
        },
        "captions": reviewed,
    }
    http_client = HTTPSequence([(200, captions), (404, None), (200, review_payload)])
    config = load_fireworks_leaderboard_config(
        {"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "key"}
    )
    assert config is not None
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal", "sarcastic")),
        make_frames(tmp_path),
        config=config,
        client_factory=lambda value: FireworksLeaderboardClient(value, client=http_client),
        remaining_seconds=300,
    )
    assert result == reviewed
    assert http_client.models == [
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
    ]


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_or_permission_failure_does_not_try_fallback_model(tmp_path, status):
    http_client = HTTPSequence([(status, None)])
    config = load_fireworks_leaderboard_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_leaderboard",
            "FIREWORKS_API_KEY": "key",
            "FIREWORKS_LEADERBOARD_ENABLE_REVIEW": "false",
        }
    )
    assert config is not None
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal",)),
        make_frames(tmp_path),
        config=config,
        client_factory=lambda value: FireworksLeaderboardClient(value, client=http_client),
        remaining_seconds=149,
    )
    assert result == build_leaderboard_fallback_captions(("formal",))
    assert http_client.models == [DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL]


def test_focused_repair_preserves_valid_caption_and_only_requests_missing_style(tmp_path):
    captions = valid_captions(("formal", "sarcastic"))
    incomplete = FireworksLeaderboardRequestError(
        "incomplete", retryable=True, category="missing_style", partial_captions={"formal": captions["formal"]}
    )
    client = ScriptedClient([incomplete, incomplete, {"sarcastic": captions["sarcastic"]}])
    config = load_fireworks_leaderboard_config(
        {
            "GEMMACLIP_PROVIDER": "fireworks_leaderboard",
            "FIREWORKS_API_KEY": "key",
            "FIREWORKS_LEADERBOARD_ENABLE_REVIEW": "false",
        }
    )
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal", "sarcastic")),
        make_frames(tmp_path),
        config=config,
        client_factory=lambda _: client,
        remaining_seconds=200,
    )
    assert result == captions
    assert client.calls[-1] == (DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL, "repair", 0.2)


def test_review_uses_minimax_then_qwen_fallback_and_preserves_failed_review(tmp_path):
    captions = valid_captions(("formal", "sarcastic"))
    reviewed = dict(captions, sarcastic="A person moves a large box indoors while the supposedly simple task turns into a carefully choreographed performance for everyone nearby.")
    client = ScriptedClient([captions, FireworksLeaderboardRequestError("busy", retryable=True, category="rate_limited"), {"scores": {style: {"accuracy": 0.9, "style_match": 0.9} for style in captions}, "captions": reviewed}])
    config = load_fireworks_leaderboard_config({"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "key"})
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal", "sarcastic")), make_frames(tmp_path), config=config, client_factory=lambda _: client, remaining_seconds=300
    )
    assert result == reviewed
    assert [model for model, _, _ in client.calls] == [
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL,
        DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
    ]


def test_runtime_thresholds_skip_generation_and_review(tmp_path):
    client = ScriptedClient([])
    config = load_fireworks_leaderboard_config({"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "key"})
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal",)), make_frames(tmp_path), config=config, client_factory=lambda _: client, remaining_seconds=64
    )
    assert result == build_leaderboard_fallback_captions(("formal",))
    assert client.calls == []

    captions = valid_captions(("formal",))
    client = ScriptedClient([captions])
    config = load_fireworks_leaderboard_config({"GEMMACLIP_PROVIDER": "fireworks_leaderboard", "FIREWORKS_API_KEY": "key"})
    result = generate_fireworks_leaderboard_captions(
        make_task(("formal",)), make_frames(tmp_path / "second"), config=config, client_factory=lambda _: client, remaining_seconds=149
    )
    assert result == captions
    assert [operation for _, operation, _ in client.calls] == ["generation"]


@pytest.mark.parametrize(
    "caption",
    [
        "Too short.",
        "This caption is deliberately written with many words so it exceeds the strict thirty five word maximum and should therefore be rejected by deterministic validation before reaching the final output because the sentence keeps adding unnecessary explanatory material.",
        "!!!!!!!!!!!!!!!!!!!!",
        "The person speaks clearly while walking across the room and this unsupported audio assertion must be rejected by the local validator.",
    ],
)
def test_strict_validation_rejects_unsafe_or_wrong_length_captions(caption):
    with pytest.raises(CaptionValidationError):
        validate_caption_payload({"formal": caption}, ("formal",))


def test_validation_strips_code_fences_and_rejects_extra_only_payload():
    parsed = parse_json_object('```json\n{"formal": "caption text"}\n```')
    assert parsed == {"formal": "caption text"}
    with pytest.raises(CaptionValidationError, match="missing requested style"):
        validate_caption_payload({"sarcastic": "unused"}, ("formal",))


def test_fallback_captions_meet_local_word_bounds():
    captions = build_leaderboard_fallback_captions(make_task().styles)
    assert all(18 <= caption_word_count(caption) <= 35 for caption in captions.values())


def test_web_routed_environment_remains_gemma_only(tmp_path):
    from gemmaclip.web.storage import RunStorage

    services = WebServices(RunStorage(tmp_path), env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"})
    assert services._routed_env()["GEMMACLIP_PROVIDER"] == "routed_gemma"


def test_cli_dispatches_new_provider_to_six_frame_extractor_and_new_generator(tmp_path, monkeypatch):
    from gemmaclip.main import process_task
    from gemmaclip.video import VideoMetadata

    task = make_task(("formal",))
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    extraction_options = []
    generator_calls = []

    monkeypatch.setattr("gemmaclip.main.download_video", lambda task, **kwargs: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda path, **kwargs: VideoMetadata(5.0, 24.0, 640, 480, 120),
    )

    def extract(*args, **kwargs):
        extraction_options.append(kwargs)
        return [ExtractedFrame(frame, float(index)) for index in range(6)]

    def generate(*args, **kwargs):
        generator_calls.append(kwargs)
        return build_leaderboard_fallback_captions(("formal",))

    monkeypatch.setattr("gemmaclip.main.extract_frames", extract)
    monkeypatch.setattr("gemmaclip.main.generate_fireworks_leaderboard_captions", generate)
    monkeypatch.setattr("gemmaclip.main.generate_captions", lambda *args, **kwargs: pytest.fail("legacy generator used"))

    result, _ = process_task(
        task,
        tmp_path,
        env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"},
        remaining_seconds=200,
    )
    assert result["captions"] == build_leaderboard_fallback_captions(("formal",))
    assert extraction_options[0]["fireworks_judge"] is True
    assert extraction_options[0]["google_fast"] is False
    assert generator_calls


def test_leaderboard_download_failure_uses_strict_fallback(tmp_path, monkeypatch):
    from gemmaclip.main import process_task

    task = make_task()
    monkeypatch.setattr(
        "gemmaclip.main.download_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )
    result, manifest = process_task(task, tmp_path, env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"})
    assert manifest is None
    assert result["captions"] == build_leaderboard_fallback_captions(task.styles)
    assert all(18 <= caption_word_count(caption) <= 35 for caption in result["captions"].values())


def test_leaderboard_metadata_probe_failure_uses_strict_fallback(tmp_path, monkeypatch):
    from gemmaclip.main import process_task

    task = make_task()
    monkeypatch.setattr("gemmaclip.main.download_video", lambda *args, **kwargs: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ffprobe failed")),
    )
    result, manifest = process_task(task, tmp_path, env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"})
    assert manifest is None
    assert result["captions"] == build_leaderboard_fallback_captions(task.styles)
    assert all(18 <= caption_word_count(caption) <= 35 for caption in result["captions"].values())


def test_leaderboard_frame_extraction_failure_uses_strict_fallback(tmp_path, monkeypatch):
    from gemmaclip.main import process_task

    task = make_task()
    monkeypatch.setattr("gemmaclip.main.download_video", lambda *args, **kwargs: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda *args, **kwargs: VideoMetadata(5.0, 24.0, 640, 480, 120),
    )
    monkeypatch.setattr(
        "gemmaclip.main.extract_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("frame extraction failed")),
    )
    result, manifest = process_task(task, tmp_path, env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"})
    assert manifest is None
    assert result["captions"] == build_leaderboard_fallback_captions(task.styles)
    assert all(18 <= caption_word_count(caption) <= 35 for caption in result["captions"].values())


def test_leaderboard_pre_model_failure_does_not_log_signed_video_url(tmp_path, monkeypatch, caplog):
    from gemmaclip.main import process_task

    signed_url = "https://example.test/video.mp4?token=super-secret"
    caplog.set_level(logging.WARNING, logger="gemmaclip")
    monkeypatch.setattr(
        "gemmaclip.main.download_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(f"download failed for {signed_url}")),
    )
    result, _ = process_task(
        make_task(),
        tmp_path,
        env={"GEMMACLIP_PROVIDER": "fireworks_leaderboard"},
    )
    assert result["captions"] == build_leaderboard_fallback_captions(make_task().styles)
    assert signed_url not in caplog.text
    assert "super-secret" not in caplog.text
    assert "category=RuntimeError" in caplog.text
