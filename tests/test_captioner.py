from __future__ import annotations

from gemmaclip.captioner import (
    build_fallback_captions,
    build_placeholder_captions,
    generate_captions,
)
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import DEFAULT_FIREWORKS_BASE_URL, extract_message_text, load_gemma_config, parse_json_object
from gemmaclip.io import Task


def make_task() -> Task:
    return Task(
        task_id="clip-1",
        video_url="https://example.com/video.mp4",
        styles=("formal", "sarcastic"),
    )


def make_frames(tmp_path) -> list[ExtractedFrame]:
    frame_path = tmp_path / "frame_001.jpg"
    frame_path.write_bytes(b"jpeg-bytes")
    return [ExtractedFrame(path=frame_path, timestamp_seconds=0.25)]


def test_parse_json_object_extracts_wrapped_json():
    payload = parse_json_object('Model output follows:\n```json\n{"scene":"office","actions":["typing"]}\n```')

    assert payload == {
        "scene": "office",
        "actions": ["typing"],
    }


def test_extract_message_text_supports_content_parts():
    text = extract_message_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"scene":"garden"}'},
                        ]
                    }
                }
            ]
        }
    )

    assert text == '{"scene":"garden"}'


def test_load_gemma_config_uses_fireworks_fallbacks():
    config = load_gemma_config(
        {
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_MODEL": "accounts/fireworks/models/custom-model",
        }
    )

    assert config is not None
    assert config.api_key == "fireworks-key"
    assert config.base_url == DEFAULT_FIREWORKS_BASE_URL
    assert config.model == "accounts/fireworks/models/custom-model"


def test_load_gemma_config_prefers_gemma_values_over_fireworks_fallbacks():
    config = load_gemma_config(
        {
            "GEMMA_API_KEY": "gemma-key",
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_BASE_URL": "https://example.com/custom/v1",
            "GEMMA_MODEL": "accounts/123/deployments/456",
        }
    )

    assert config is not None
    assert config.api_key == "gemma-key"
    assert config.base_url == "https://example.com/custom/v1"
    assert config.model == "accounts/123/deployments/456"


def test_generate_captions_uses_placeholder_when_config_missing(tmp_path):
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={},
    )

    assert captions == build_placeholder_captions(("formal", "sarcastic"))


def test_generate_captions_falls_back_when_model_output_is_invalid(tmp_path):
    class FakeClient:
        def __init__(self, _config):
            self.calls = 0

        def chat_completion_json(self, messages, temperature):
            self.calls += 1
            if self.calls == 1:
                return {
                    "scene": "office",
                    "main_subjects": ["worker"],
                    "actions": ["typing"],
                    "setting": "office",
                    "visible_objects": ["computer"],
                    "mood": "focused",
                    "camera_notes": "static shot",
                    "uncertain_details": [],
                }
            return {
                "formal": "A worker uses a desktop computer in a modern office setting while focusing on routine tasks."
            }

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMMA_API_KEY": "key",
            "GEMMA_BASE_URL": "https://example.com/v1",
            "GEMMA_MODEL": "gemma-test",
        },
        client_factory=FakeClient,
    )

    assert captions == build_fallback_captions(("formal", "sarcastic"))


def test_generate_captions_dry_run_skips_api_calls(tmp_path):
    class FailingClient:
        def __init__(self, _config):
            raise AssertionError("API client should not be constructed during dry-run mode.")

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        dry_run=True,
        env={
            "GEMMA_API_KEY": "key",
            "GEMMA_BASE_URL": "https://example.com/v1",
            "GEMMA_MODEL": "gemma-test",
        },
        client_factory=FailingClient,
    )

    assert captions == build_placeholder_captions(("formal", "sarcastic"))
