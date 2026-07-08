from __future__ import annotations

from io import BytesIO

from gemmaclip.captioner import (
    build_fallback_captions,
    build_placeholder_captions,
    generate_captions,
    make_resized_jpeg_bytes,
    select_gemma_frames,
)
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_FIREWORKS_BASE_URL,
    DEFAULT_GEMMA_MAX_TOKENS,
    DEFAULT_TOP_K,
    GemmaConfig,
    build_chat_completion_payload,
    extract_message_text,
    load_gemma_config,
    parse_json_object,
)
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


def make_frame_sequence(tmp_path, count: int) -> list[ExtractedFrame]:
    frames: list[ExtractedFrame] = []
    for index in range(count):
        frame_path = tmp_path / f"frame_{index + 1:03d}.jpg"
        frame_path.write_bytes(f"jpeg-{index}".encode("ascii"))
        frames.append(ExtractedFrame(path=frame_path, timestamp_seconds=float(index)))
    return frames


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


def test_select_gemma_frames_returns_all_frames_when_count_is_within_limit(tmp_path):
    frames = make_frame_sequence(tmp_path, 5)

    selected = select_gemma_frames(frames, max_frames=12)

    assert selected == frames


def test_select_gemma_frames_returns_exactly_max_frames_when_input_is_larger(tmp_path):
    frames = make_frame_sequence(tmp_path, 20)

    selected = select_gemma_frames(frames, max_frames=12)

    assert len(selected) == 12


def test_select_gemma_frames_includes_first_and_last_frames(tmp_path):
    frames = make_frame_sequence(tmp_path, 20)

    selected = select_gemma_frames(frames, max_frames=12)

    assert selected[0] == frames[0]
    assert selected[-1] == frames[-1]


def test_select_gemma_frames_preserves_chronological_order(tmp_path):
    frames = make_frame_sequence(tmp_path, 20)

    selected = select_gemma_frames(frames, max_frames=12)

    assert [frame.timestamp_seconds for frame in selected] == sorted(
        frame.timestamp_seconds for frame in selected
    )


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
    assert config.max_tokens == DEFAULT_GEMMA_MAX_TOKENS
    assert config.use_response_format is False


def test_make_resized_jpeg_bytes_limits_max_side(tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 1000), color="red").save(image_path, format="JPEG", quality=95)

    resized_bytes = make_resized_jpeg_bytes(image_path, max_side=768, quality=85)

    with Image.open(BytesIO(resized_bytes)) as resized_image:
        assert max(resized_image.size) <= 768


def test_build_chat_completion_payload_matches_fireworks_defaults():
    config = GemmaConfig(
        api_key="key",
        base_url=DEFAULT_FIREWORKS_BASE_URL,
        model="accounts/fireworks/models/custom-model",
    )
    messages = [{"role": "user", "content": "hello"}]

    payload = build_chat_completion_payload(config, messages=messages, temperature=0.7)

    assert payload == {
        "model": "accounts/fireworks/models/custom-model",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": DEFAULT_GEMMA_MAX_TOKENS,
        "top_k": DEFAULT_TOP_K,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }


def test_build_chat_completion_payload_includes_response_format_when_enabled():
    config = GemmaConfig(
        api_key="key",
        base_url=DEFAULT_FIREWORKS_BASE_URL,
        model="accounts/123/deployments/456",
        max_tokens=1024,
        use_response_format=True,
    )

    payload = build_chat_completion_payload(config, messages=[{"role": "user", "content": "hello"}], temperature=0.1)

    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_tokens"] == 1024


def test_load_gemma_config_reads_max_tokens_and_response_format():
    config = load_gemma_config(
        {
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_MODEL": "accounts/fireworks/models/custom-model",
            "GEMMA_MAX_TOKENS": "4096",
            "GEMMA_USE_RESPONSE_FORMAT": "true",
        }
    )

    assert config is not None
    assert config.max_tokens == 4096
    assert config.use_response_format is True


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
