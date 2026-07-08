from __future__ import annotations

from io import BytesIO

from gemmaclip.captioner import (
    build_evidence_debug_payload,
    build_fallback_captions,
    build_placeholder_captions,
    build_verifier_messages,
    generate_captions,
    make_resized_jpeg_bytes,
    maybe_verify_captions,
    normalize_captions,
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
    from PIL import Image

    frame_path = tmp_path / "frame_001.jpg"
    Image.new("RGB", (80, 45), color="blue").save(frame_path, format="JPEG", quality=85)
    return [ExtractedFrame(path=frame_path, timestamp_seconds=0.25)]


def make_frame_sequence(tmp_path, count: int) -> list[ExtractedFrame]:
    frames: list[ExtractedFrame] = []
    for index in range(count):
        frame_path = tmp_path / f"frame_{index + 1:03d}.jpg"
        frame_path.write_bytes(f"jpeg-{index}".encode("ascii"))
        frames.append(ExtractedFrame(path=frame_path, timestamp_seconds=float(index)))
    return frames


def make_evidence(**overrides):
    evidence = {
        "scene": "office scene",
        "main_subjects": ["person"],
        "actions": ["working"],
        "setting": "office",
        "visible_objects": ["desk", "computer"],
        "mood": "neutral",
        "camera_notes": "static shot",
        "uncertain_details": [],
    }
    evidence.update(overrides)
    return evidence


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


def test_build_evidence_debug_payload_contains_task_id_selected_frames_and_evidence(tmp_path):
    frames = make_frame_sequence(tmp_path, 2)
    evidence = make_evidence(actions=["walking"])

    payload = build_evidence_debug_payload("clip-1", frames, evidence)

    assert payload["task_id"] == "clip-1"
    assert payload["selected_frame_count"] == 2
    assert payload["selected_frames"] == [
        {
            "path": str(frames[0].path),
            "timestamp_seconds": 0.0,
        },
        {
            "path": str(frames[1].path),
            "timestamp_seconds": 1.0,
        },
    ]
    assert payload["evidence"] == evidence


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


def test_normalize_captions_rejects_banned_speculation_phrase():
    try:
        normalize_captions(
            {"formal": "A person probably waits beside traffic.", "sarcastic": "A worker stands by while traffic does the usual thing."},
            ("formal", "sarcastic"),
            make_evidence(),
        )
    except ValueError as exc:
        assert "banned speculation phrase" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to reject banned speculation phrases.")


def test_normalize_captions_rejects_overlong_caption():
    overlong = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentytfour twentytfive twentysix"

    try:
        normalize_captions(
            {"formal": overlong, "sarcastic": "A worker stands quietly while the room acts extremely impressed by ordinary office activity."},
            ("formal", "sarcastic"),
            make_evidence(),
        )
    except ValueError as exc:
        assert "maximum allowed word count" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to reject overlong captions.")


def test_normalize_captions_rejects_unsupported_humorous_tech_script_claim():
    try:
        normalize_captions(
            {"humorous_tech": "A worker watches traffic like a script rerun that forgot to deploy anything new today."},
            ("humorous_tech",),
            make_evidence(actions=["walking"]),
        )
    except ValueError as exc:
        assert "unsupported coding or scripting claim" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to reject unsupported scripting claims.")


def test_normalize_captions_allows_humorous_tech_metaphor_without_coding_claim():
    captions = normalize_captions(
        {"humorous_tech": "Traffic moves like data packets hitting CPU limits during an unusually patient rush hour commute today."},
        ("humorous_tech",),
        make_evidence(actions=["moving"], setting="street"),
    )

    assert "humorous_tech" in captions


def test_verifier_is_skipped_when_disabled(tmp_path):
    class FailingClient:
        def chat_completion_json(self, messages, temperature):
            raise AssertionError("Verifier should not run when disabled.")

    captions = maybe_verify_captions(
        make_task(),
        {"formal": "A worker stands near a desk in a quiet office during a routine moment.", "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today."},
        make_evidence(),
        FailingClient(),
        {"GEMMACLIP_DISABLE_VERIFIER": "true"},
    )

    assert "formal" in captions


def test_verifier_failure_keeps_original_captions():
    class FailingClient:
        def chat_completion_json(self, messages, temperature):
            raise ValueError("Verifier failed.")

    original = {
        "formal": "A worker stands near a desk in a quiet office during a routine moment.",
        "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
    }

    captions = maybe_verify_captions(
        make_task(),
        original,
        make_evidence(),
        FailingClient(),
        {},
    )

    assert captions == original


def test_verifier_output_is_validated():
    class InvalidVerifierClient:
        def chat_completion_json(self, messages, temperature):
            return {
                "formal": "A worker probably stands here.",
                "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
            }

    original = {
        "formal": "A worker stands near a desk in a quiet office during a routine moment.",
        "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
    }

    captions = maybe_verify_captions(
        make_task(),
        original,
        make_evidence(),
        InvalidVerifierClient(),
        {},
    )

    assert captions == original


def test_build_verifier_messages_includes_task_evidence_and_captions():
    messages = build_verifier_messages(
        "clip-1",
        ("formal", "sarcastic"),
        make_evidence(),
        {
            "formal": "A worker stands near a desk in a quiet office during a routine moment.",
            "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
        },
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Task ID: clip-1" in messages[1]["content"]


def test_generate_captions_writes_debug_caption_files(tmp_path):
    class FakeClient:
        def __init__(self, _config):
            self.calls = 0

        def chat_completion_json(self, messages, temperature):
            self.calls += 1
            if self.calls == 1:
                return {
                    "scene": "office",
                    "main_subjects": ["worker"],
                    "actions": ["standing"],
                    "setting": "office",
                    "visible_objects": ["desk"],
                    "mood": "neutral",
                    "camera_notes": "static shot",
                    "uncertain_details": [],
                }
            if self.calls == 2:
                return {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            return {
                "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
            }

    debug_dir = tmp_path / "debug"
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMMA_API_KEY": "key",
            "GEMMA_BASE_URL": "https://example.com/v1",
            "GEMMA_MODEL": "gemma-test",
        },
        debug_dir=debug_dir,
        client_factory=FakeClient,
    )

    assert captions["formal"]
    assert (debug_dir / "clip-1_captions_raw.json").exists()
    assert (debug_dir / "clip-1_captions_verified.json").exists()


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
