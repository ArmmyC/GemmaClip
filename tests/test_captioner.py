from __future__ import annotations

import json
from io import BytesIO

import httpx

from gemmaclip.captioner import (
    build_caption_messages,
    build_evidence_messages,
    build_google_evidence_messages,
    build_evidence_debug_payload,
    build_fallback_captions,
    build_placeholder_captions,
    build_verifier_messages,
    extract_caption_json,
    extract_evidence_json,
    generate_captions,
    generate_evidence,
    make_resized_jpeg_bytes,
    maybe_verify_captions,
    normalize_evidence,
    normalize_captions,
    select_google_evidence_frames,
    select_gemma_frames,
)
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_FIREWORKS_BASE_URL,
    DEFAULT_GEMMA_MODEL,
    DEFAULT_GEMMA_TEXT_MODEL,
    DEFAULT_GEMMA_VISION_MODEL,
    DEFAULT_GEMMA_MAX_TOKENS,
    DEFAULT_GOOGLE_GEMMA_MODEL,
    GOOGLE_IMAGE_MAX_SIDE,
    GOOGLE_IMAGE_QUALITY,
    DEFAULT_PROVIDER_FIREWORKS,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_OPENROUTER,
    DEFAULT_TOP_K,
    GemmaClient,
    GemmaConfig,
    GemmaModelConfig,
    GoogleGeminiClient,
    OpenRouterClient,
    build_chat_completion_payload,
    build_openrouter_chat_completion_payload,
    extract_message_text,
    load_gemma_config,
    parse_json_object,
    _make_google_resized_jpeg_bytes,
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


def make_valid_frame_sequence(tmp_path, count: int) -> list[ExtractedFrame]:
    from PIL import Image

    frames: list[ExtractedFrame] = []
    for index in range(count):
        frame_path = tmp_path / f"frame_{index + 1:03d}.jpg"
        Image.new("RGB", (320, 180), color=(index * 17 % 255, index * 29 % 255, index * 37 % 255)).save(
            frame_path,
            format="JPEG",
            quality=90,
        )
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
        "temporal_progression": "The person keeps working at the desk across the clip.",
        "caption_focus": "person working at a desk",
        "verified_description": "A person works at a desk in an office. The framing stays focused on the same routine action.",
        "possible_misreads_to_avoid": ["Do not assume the person is speaking.", "Do not invent brand names."],
        "style_hooks": {
            "sarcastic": "The action stays steady enough for dry understatement.",
            "humorous_tech": "The routine pace can support a light CPU or buffering metaphor.",
            "humorous_non_tech": "The calm repetition can support gentle everyday humor.",
        },
    }
    evidence.update(overrides)
    return evidence


def test_parse_json_object_extracts_wrapped_json():
    payload = parse_json_object('Model output follows:\n```json\n{"scene":"office","actions":["typing"]}\n```')

    assert payload == {
        "scene": "office",
        "actions": ["typing"],
    }


def test_parse_json_object_rejects_empty_object():
    try:
        parse_json_object("{}")
    except ValueError as exc:
        assert "non-empty JSON object" in str(exc)
    else:
        raise AssertionError("Expected parse_json_object to reject empty objects.")


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


def test_normalize_evidence_accepts_new_fields():
    normalized = normalize_evidence(
        {
            "scene": "office scene",
            "main_subjects": ["worker"],
            "actions": ["standing"],
            "setting": "office",
            "visible_objects": ["desk"],
            "mood": "neutral",
            "camera_notes": "static shot",
            "temporal_progression": "The worker stays near the desk.",
            "caption_focus": "worker standing by a desk",
            "verified_description": "A worker stands near a desk in an office. The framing keeps that person centered.",
            "possible_misreads_to_avoid": ["Do not assume the worker is talking."],
            "style_hooks": {
                "sarcastic": "The stillness can support dry irony.",
                "humorous_tech": "The scene can support a calm buffering metaphor.",
                "humorous_non_tech": "The stillness can support a gentle everyday joke.",
            },
        }
    )

    assert normalized["verified_description"].startswith("A worker stands near a desk")
    assert normalized["possible_misreads_to_avoid"] == ["Do not assume the worker is talking."]
    assert normalized["style_hooks"]["humorous_tech"] == "The scene can support a calm buffering metaphor."


def test_load_gemma_config_uses_fireworks_fallbacks():
    config = load_gemma_config(
        {"FIREWORKS_API_KEY": "fireworks-key"}
    )

    assert config is not None
    assert config.provider == DEFAULT_PROVIDER_FIREWORKS
    assert config.api_key == "fireworks-key"
    assert config.base_url == DEFAULT_FIREWORKS_BASE_URL
    assert config.vision_model == DEFAULT_GEMMA_VISION_MODEL
    assert config.text_model == DEFAULT_GEMMA_TEXT_MODEL
    assert config.fallback_models == DEFAULT_FALLBACK_MODELS
    assert config.max_tokens == DEFAULT_GEMMA_MAX_TOKENS
    assert config.use_response_format is False


def test_load_gemma_config_prefers_google_provider_when_gemini_key_exists():
    config = load_gemma_config(
        {"GEMINI_API_KEY": "gemini-key"}
    )

    assert config is not None
    assert config.provider == DEFAULT_PROVIDER_GOOGLE
    assert config.api_key == "gemini-key"
    assert config.base_url is None
    assert config.vision_model == DEFAULT_GOOGLE_GEMMA_MODEL
    assert config.text_model == DEFAULT_GOOGLE_GEMMA_MODEL


def test_load_gemma_config_uses_openrouter_when_requested():
    config = load_gemma_config(
        {
            "GEMMACLIP_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
            "OPENROUTER_MODEL": "openrouter/model",
        }
    )

    assert config is not None
    assert config.provider == DEFAULT_PROVIDER_OPENROUTER
    assert config.api_key == "openrouter-key"
    assert config.vision_model == "openrouter/model"
    assert config.text_model == "openrouter/model"


def test_load_gemma_config_uses_gemma_model_override_for_both_roles():
    config = load_gemma_config(
        {
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_MODEL": "accounts/fireworks/models/custom-model",
        }
    )

    assert config is not None
    assert config.vision_model == "accounts/fireworks/models/custom-model"
    assert config.text_model == "accounts/fireworks/models/custom-model"


def test_load_gemma_config_parses_fallback_model_list():
    config = load_gemma_config(
        {
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_FALLBACK_MODELS": "accounts/fireworks/models/fallback-a, accounts/fireworks/models/fallback-b",
        }
    )

    assert config is not None
    assert config.fallback_models == (
        "accounts/fireworks/models/fallback-a",
        "accounts/fireworks/models/fallback-b",
    )


def test_make_resized_jpeg_bytes_limits_max_side(tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 1000), color="red").save(image_path, format="JPEG", quality=95)

    resized_bytes = make_resized_jpeg_bytes(image_path, max_side=768, quality=85)

    with Image.open(BytesIO(resized_bytes)) as resized_image:
        assert max(resized_image.size) <= 768


def _config_to_mapping(config):
    if isinstance(config, dict):
        return config
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_none=True)
    raise AssertionError(f"Unsupported config object in test: {type(config)}")


def test_google_gemini_client_uploads_files_instead_of_inline_parts(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"
            self.marker = "uploaded-file"

    class FakeGoogleFiles:
        def __init__(self):
            self.upload_calls: list[tuple[object, object]] = []
            self.delete_calls: list[str] = []

        def upload(self, *, file, config=None):
            self.upload_calls.append((file, config))
            return FakeGoogleFile(f"files/{len(self.upload_calls)}")

        def delete(self, *, name, config=None):
            self.delete_calls.append(name)
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self, text: str):
            self.calls: list[dict[str, object]] = []
            self._text = text

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return FakeGoogleResponse(self._text)

    class FakeGoogleSdkClient:
        def __init__(self, text: str):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels(text)

    frames = make_valid_frame_sequence(tmp_path, 2)
    sdk_client = FakeGoogleSdkClient(
        """{
          "scene": "garden path",
          "main_subjects": ["cat"],
          "actions": ["walking"],
          "setting": "garden",
          "visible_objects": ["plants"],
          "mood": "calm",
          "camera_notes": "",
          "uncertain_details": []
        }"""
    )
    client = GoogleGeminiClient(
        GemmaModelConfig(
            api_key="gemini-key",
            base_url=None,
            model=DEFAULT_GOOGLE_GEMMA_MODEL,
            provider=DEFAULT_PROVIDER_GOOGLE,
            fallback_models=(),
        ),
        client=sdk_client,
    )

    evidence = generate_evidence("clip-1", frames, client)
    contents = sdk_client.models.calls[0]["contents"]

    assert evidence["scene"] == "garden path"
    assert evidence["main_subjects"] == ["cat"]
    assert sdk_client.models.calls[0]["model"] == DEFAULT_GOOGLE_GEMMA_MODEL
    assert len(sdk_client.files.upload_calls) == 2
    assert all(getattr(item, "marker", "") == "uploaded-file" for item in contents if not isinstance(item, str))
    assert sdk_client.files.delete_calls == ["files/1", "files/2"]


def test_google_evidence_request_config_does_not_include_response_mime_type(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"

    class FakeGoogleFiles:
        def upload(self, *, file, config=None):
            return FakeGoogleFile("files/1")

        def delete(self, *, name, config=None):
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return FakeGoogleResponse(
                '{"scene":"garden","main_subjects":["cat"],"actions":["walking"],"setting":"garden","visible_objects":["plants"],"mood":"calm","camera_notes":"","uncertain_details":[]}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()
    client = GoogleGeminiClient(
        GemmaModelConfig(
            api_key="gemini-key",
            base_url=None,
            model=DEFAULT_GOOGLE_GEMMA_MODEL,
            provider=DEFAULT_PROVIDER_GOOGLE,
            fallback_models=(),
        ),
        client=sdk_client,
    )

    generate_evidence("clip-1", make_valid_frame_sequence(tmp_path, 2), client)
    config_mapping = _config_to_mapping(sdk_client.models.calls[0]["config"])

    assert "response_mime_type" not in config_mapping


def test_google_request_config_includes_minimal_thinking_when_supported(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"

    class FakeGoogleFiles:
        def upload(self, *, file, config=None):
            return FakeGoogleFile("files/1")

        def delete(self, *, name, config=None):
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return FakeGoogleResponse(
                '{"scene":"garden","main_subjects":["cat"],"actions":["walking"],"setting":"garden","visible_objects":["plants"],"mood":"calm","camera_notes":"","uncertain_details":[]}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()
    client = GoogleGeminiClient(
        GemmaModelConfig(
            api_key="gemini-key",
            base_url=None,
            model=DEFAULT_GOOGLE_GEMMA_MODEL,
            provider=DEFAULT_PROVIDER_GOOGLE,
            fallback_models=(),
        ),
        client=sdk_client,
    )

    generate_evidence("clip-1", make_valid_frame_sequence(tmp_path, 1), client)
    config_mapping = _config_to_mapping(sdk_client.models.calls[0]["config"])

    assert str(config_mapping["thinking_config"]["thinking_level"]).lower().endswith("minimal")


def test_google_upload_resize_uses_1536_max_side_and_quality_85(tmp_path):
    from PIL import Image

    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (3000, 1800), color="green").save(image_path, format="JPEG", quality=95)

    resized_bytes = _make_google_resized_jpeg_bytes(image_path)

    with Image.open(BytesIO(resized_bytes)) as resized_image:
        assert max(resized_image.size) <= 1536

    assert GOOGLE_IMAGE_MAX_SIDE == 1536
    assert GOOGLE_IMAGE_QUALITY == 85


def test_openrouter_image_file_is_converted_to_base64_image_url(tmp_path):
    from PIL import Image

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), color="blue").save(image_path, format="JPEG", quality=85)

    class RecordingHttpClient:
        def __init__(self):
            self.payloads: list[dict[str, object]] = []

        def post(self, url, headers, json):
            self.payloads.append(json)
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

    http_client = RecordingHttpClient()
    client = OpenRouterClient(
        GemmaModelConfig(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="openrouter/model",
            provider=DEFAULT_PROVIDER_OPENROUTER,
            fallback_models=(),
        ),
        client=http_client,
    )

    response_text = client.chat_completion_text(
        [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_file", "path": str(image_path), "mime_type": "image/jpeg"},
                ],
            },
        ],
        0.1,
    )

    content = http_client.payloads[0]["messages"][1]["content"]
    assert response_text == "ok"
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_caption_prompt_includes_verified_description_and_style_hooks():
    messages = build_caption_messages(
        "clip-1",
        ("formal", "sarcastic"),
        make_evidence(),
    )

    assert "verified_description" in messages[1]["content"]
    assert "style_hooks" in messages[1]["content"]
    assert "Prioritize evidence fields in this order: verified_description" in messages[1]["content"]


def test_openrouter_text_only_caption_call_works():
    class RecordingHttpClient:
        def post(self, url, headers, json):
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": '{"formal":"caption"}'}}]},
            )

    client = OpenRouterClient(
        GemmaModelConfig(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="openrouter/model",
            provider=DEFAULT_PROVIDER_OPENROUTER,
            fallback_models=(),
        ),
        client=RecordingHttpClient(),
    )

    text = client.chat_completion_text(
        [{"role": "user", "content": "caption from evidence"}],
        0.7,
    )

    assert text == '{"formal":"caption"}'


def test_openrouter_retries_on_429_and_5xx():
    class RetryingHttpClient:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers, json):
            self.calls += 1
            request = httpx.Request("POST", url, json=json, headers=headers)
            if self.calls == 1:
                return httpx.Response(429, request=request, json={"error": "retry"})
            if self.calls == 2:
                return httpx.Response(503, request=request, json={"error": "retry"})
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "final"}}]},
            )

    delays: list[float] = []
    client = OpenRouterClient(
        GemmaModelConfig(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="openrouter/model",
            provider=DEFAULT_PROVIDER_OPENROUTER,
            fallback_models=(),
        ),
        client=RetryingHttpClient(),
        sleeper=delays.append,
    )

    text = client.chat_completion_text([{"role": "user", "content": "hello"}], 0.2)

    assert text == "final"
    assert delays == [0.5, 1.0]


def test_openrouter_does_not_log_secrets(caplog):
    class RecordingHttpClient:
        def post(self, url, headers, json):
            request = httpx.Request("POST", url, json=json, headers=headers)
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": "safe"}}]},
            )

    caplog.set_level("INFO")
    secret = "super-secret-key"
    client = OpenRouterClient(
        GemmaModelConfig(
            api_key=secret,
            base_url="https://openrouter.ai/api/v1/chat/completions",
            model="openrouter/model",
            provider=DEFAULT_PROVIDER_OPENROUTER,
            fallback_models=(),
        ),
        client=RecordingHttpClient(),
    )

    client.chat_completion_text([{"role": "user", "content": "hello"}], 0.2)

    assert "provider=openrouter" in caplog.text
    assert "model=openrouter/model" in caplog.text
    assert "status_code=200" in caplog.text
    assert secret not in caplog.text


def test_google_v7_provider_uses_six_frames(tmp_path):
    class FakeClient:
        prompt_text = ""
        call_count = 0

        def __init__(self, _config):
            self._config = _config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            FakeClient.call_count += 1
            if isinstance(messages[1]["content"], list):
                FakeClient.prompt_text = next(
                    item["text"]
                    for item in messages[1]["content"]
                    if isinstance(item, dict) and item.get("type") == "text"
                )
                return json.dumps(
                    {
                        "scene": "office scene",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "temporal_progression": "The worker remains near the desk throughout the clip.",
                        "caption_focus": "worker standing near a desk",
                    }
                )
            return json.dumps(
                {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            )

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert captions["formal"]
    assert FakeClient.call_count == 2
    assert FakeClient.prompt_text.count("timestamp_seconds=") == 6


def test_fireworks_provider_still_uses_twelve_frames_and_inline_data_urls(tmp_path):
    messages = build_evidence_messages("clip-1", make_valid_frame_sequence(tmp_path, 12))
    image_parts = [item for item in messages[1]["content"] if item.get("type") == "image_url"]

    assert len(image_parts) == 12
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in image_parts)


def test_google_v7_uploads_exactly_one_contact_sheet_for_evidence(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"
            self.marker = "uploaded-file"

    class FakeGoogleFiles:
        def __init__(self):
            self.upload_calls: list[tuple[object, object]] = []
            self.delete_calls: list[str] = []

        def upload(self, *, file, config=None):
            self.upload_calls.append((file, config))
            return FakeGoogleFile(f"files/{len(self.upload_calls)}")

        def delete(self, *, name, config=None):
            self.delete_calls.append(name)
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeGoogleResponse(
                    '{"scene":"office scene","main_subjects":["worker"],"actions":["standing"],"setting":"office","visible_objects":["desk"],"mood":"neutral","camera_notes":"static shot","temporal_progression":"The worker stays near the desk across the clip.","caption_focus":"worker standing near a desk"}'
                )
            return FakeGoogleResponse(
                '{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()

    def google_factory(config):
        return GoogleGeminiClient(config, client=sdk_client)

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=google_factory,
    )

    assert captions["formal"]
    assert len(sdk_client.files.upload_calls) == 1
    assert len(sdk_client.models.calls) == 2


def test_google_v7_caption_json_failure_triggers_one_text_repair_call(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"
            self.marker = "uploaded-file"

    class FakeGoogleFiles:
        def __init__(self):
            self.upload_calls: list[tuple[object, object]] = []

        def upload(self, *, file, config=None):
            self.upload_calls.append((file, config))
            return FakeGoogleFile(f"files/{len(self.upload_calls)}")

        def delete(self, *, name, config=None):
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeGoogleResponse(
                    '{"scene":"office scene","main_subjects":["worker"],"actions":["standing"],"setting":"office","visible_objects":["desk"],"mood":"neutral","camera_notes":"static shot","temporal_progression":"The worker stays near the desk across the clip.","caption_focus":"worker standing near a desk"}'
                )
            if len(self.calls) == 2:
                return FakeGoogleResponse("not valid json")
            return FakeGoogleResponse(
                '{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()

    def google_factory(config):
        return GoogleGeminiClient(config, client=sdk_client)

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=google_factory,
    )

    first_contents = sdk_client.models.calls[0]["contents"]
    second_contents = sdk_client.models.calls[1]["contents"]
    third_contents = sdk_client.models.calls[2]["contents"]

    assert captions["formal"]
    assert len(sdk_client.files.upload_calls) == 1
    assert len(sdk_client.models.calls) == 3
    assert any(getattr(item, "marker", "") == "uploaded-file" for item in first_contents if not isinstance(item, str))
    assert all(isinstance(item, str) for item in second_contents)
    assert all(isinstance(item, str) for item in third_contents)


def test_google_v7_evidence_failure_falls_back_to_v6_direct_mode(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"
            self.marker = "uploaded-file"

    class FakeGoogleFiles:
        def __init__(self):
            self.upload_calls: list[tuple[object, object]] = []

        def upload(self, *, file, config=None):
            self.upload_calls.append((file, config))
            return FakeGoogleFile(f"files/{len(self.upload_calls)}")

        def delete(self, *, name, config=None):
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return FakeGoogleResponse("{}")
            return FakeGoogleResponse(
                '{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()

    def google_factory(config):
        return GoogleGeminiClient(config, client=sdk_client)

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=google_factory,
    )

    assert captions["formal"]
    assert len(sdk_client.models.calls) == 2
    assert len(sdk_client.files.upload_calls) == 2


def test_google_v7_low_remaining_time_uses_v6_direct_mode(tmp_path):
    class FakeGoogleResponse:
        def __init__(self, text: str):
            self.text = text

    class FakeGoogleFile:
        def __init__(self, name: str):
            self.name = name
            self.uri = f"gs://gemini/{name}"
            self.mime_type = "image/jpeg"
            self.marker = "uploaded-file"

    class FakeGoogleFiles:
        def __init__(self):
            self.upload_calls: list[tuple[object, object]] = []

        def upload(self, *, file, config=None):
            self.upload_calls.append((file, config))
            return FakeGoogleFile(f"files/{len(self.upload_calls)}")

        def delete(self, *, name, config=None):
            return {"deleted": name}

    class FakeGoogleModels:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return FakeGoogleResponse(
                '{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}'
            )

    class FakeGoogleSdkClient:
        def __init__(self):
            self.files = FakeGoogleFiles()
            self.models = FakeGoogleModels()

    sdk_client = FakeGoogleSdkClient()

    def google_factory(config):
        return GoogleGeminiClient(config, client=sdk_client)

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=google_factory,
        remaining_seconds=100.0,
    )

    assert captions["formal"]
    assert len(sdk_client.files.upload_calls) == 1
    assert len(sdk_client.models.calls) == 1


def test_openrouter_evidence_failure_falls_back_to_google_v7(tmp_path):
    class OpenRouterFailingClient:
        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            return "{}"

    class GoogleFallbackClient:
        call_count = 0

        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            GoogleFallbackClient.call_count += 1
            if isinstance(messages[1]["content"], list):
                return json.dumps(
                    {
                        "scene": "office scene",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "temporal_progression": "The worker stays near the desk across the clip.",
                        "caption_focus": "worker standing near a desk",
                    }
                )
            return json.dumps(
                {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            )

    created_providers: list[str] = []

    def client_factory(config):
        created_providers.append(config.provider)
        if config.provider == DEFAULT_PROVIDER_OPENROUTER:
            return OpenRouterFailingClient(config)
        if config.provider == DEFAULT_PROVIDER_GOOGLE:
            return GoogleFallbackClient(config)
        raise AssertionError(f"Unexpected provider: {config.provider}")

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMMACLIP_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
            "OPENROUTER_MODEL": "openrouter/model",
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=client_factory,
    )

    assert captions["formal"]
    assert created_providers == [DEFAULT_PROVIDER_OPENROUTER, DEFAULT_PROVIDER_GOOGLE]
    assert GoogleFallbackClient.call_count == 2


def test_openrouter_caption_failure_falls_back_to_google_caption_generation_from_evidence(tmp_path, caplog):
    class OpenRouterClientStub:
        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            if isinstance(messages[1]["content"], list):
                return json.dumps(
                    {
                        "scene": "office scene",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "temporal_progression": "The worker stays near the desk across the clip.",
                        "caption_focus": "worker standing near a desk",
                    }
                )
            return "."

    class GoogleCaptionClientStub:
        call_count = 0

        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            GoogleCaptionClientStub.call_count += 1
            return json.dumps(
                {
                    "formal": "A worker stands by a desk in a quiet office while the clip keeps its attention there.",
                    "sarcastic": "A worker stands by a desk in a quiet office, heroically sustaining this blockbuster level of movement.",
                }
            )

    created_providers: list[str] = []

    def client_factory(config):
        created_providers.append(config.provider)
        if config.provider == DEFAULT_PROVIDER_OPENROUTER:
            return OpenRouterClientStub(config)
        if config.provider == DEFAULT_PROVIDER_GOOGLE:
            return GoogleCaptionClientStub(config)
        raise AssertionError(f"Unexpected provider: {config.provider}")

    caplog.set_level("WARNING")
    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMMACLIP_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
            "OPENROUTER_MODEL": "openrouter/model",
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=client_factory,
    )

    assert captions["formal"] == "A worker stands by a desk in a quiet office while the clip keeps its attention there."
    assert captions["sarcastic"] == "A worker stands by a desk in a quiet office, heroically sustaining this blockbuster level of movement."
    assert created_providers == [DEFAULT_PROVIDER_OPENROUTER, DEFAULT_PROVIDER_GOOGLE]
    assert GoogleCaptionClientStub.call_count == 1
    assert "OpenRouter caption failed, falling back to Google caption generation from OpenRouter evidence" in caplog.text


def test_openrouter_and_google_caption_failure_uses_evidence_based_fallback(tmp_path):
    class OpenRouterClientStub:
        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            if isinstance(messages[1]["content"], list):
                return json.dumps(
                    {
                        "scene": "office scene",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "temporal_progression": "The worker stays near the desk across the clip.",
                        "caption_focus": "worker standing near a desk",
                    }
                )
            return "."

    class GoogleCaptionFailingClientStub:
        def __init__(self, config):
            self._config = config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            return "."

    created_providers: list[str] = []

    def client_factory(config):
        created_providers.append(config.provider)
        if config.provider == DEFAULT_PROVIDER_OPENROUTER:
            return OpenRouterClientStub(config)
        if config.provider == DEFAULT_PROVIDER_GOOGLE:
            return GoogleCaptionFailingClientStub(config)
        raise AssertionError(f"Unexpected provider: {config.provider}")

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "GEMMACLIP_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
            "OPENROUTER_MODEL": "openrouter/model",
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=client_factory,
    )

    assert captions["formal"] == "The clip captures worker standing in office, with desk visible nearby throughout."
    assert "video could not be fully processed" not in captions["formal"].lower()
    assert created_providers == [DEFAULT_PROVIDER_OPENROUTER, DEFAULT_PROVIDER_GOOGLE]


def test_fireworks_provider_still_uses_old_evidence_caption_behavior(tmp_path):
    class FakeClient:
        construction_index = 0
        evidence_calls = 0
        caption_calls = 0

        def __init__(self, _config):
            self._config = _config
            FakeClient.construction_index += 1

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
                FakeClient.evidence_calls += 1
                image_parts = [item for item in messages[1]["content"] if item.get("type") == "image_url"]
                assert len(image_parts) == 12
                return json.dumps(
                    {
                        "scene": "office",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "uncertain_details": [],
                    }
                )

            FakeClient.caption_calls += 1
            return json.dumps(
                {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            )

    captions = generate_captions(
        make_task(),
        make_valid_frame_sequence(tmp_path, 12),
        env={
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert captions["formal"]
    assert FakeClient.evidence_calls == 1
    assert FakeClient.caption_calls == 1


def test_build_chat_completion_payload_matches_fireworks_defaults():
    config = GemmaModelConfig(
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
    config = GemmaModelConfig(
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
    assert config.vision_model == "accounts/123/deployments/456"
    assert config.text_model == "accounts/123/deployments/456"


def test_load_gemma_config_uses_split_vision_and_text_models():
    config = load_gemma_config(
        {
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMA_VISION_MODEL": "accounts/fireworks/models/vision-model",
            "GEMMA_TEXT_MODEL": "accounts/fireworks/models/text-model",
        }
    )

    assert config is not None
    assert config.vision_model == "accounts/fireworks/models/vision-model"
    assert config.text_model == "accounts/fireworks/models/text-model"


def test_load_gemma_config_returns_none_without_any_api_key():
    assert load_gemma_config({"GEMMA_MODEL": "accounts/fireworks/models/custom-model"}) is None


def test_gemma_client_falls_back_when_primary_model_is_unavailable():
    class RecordingHttpClient:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            request = httpx.Request("POST", url, json=json, headers=headers)
            if json["model"] == "primary-model":
                return httpx.Response(
                    404,
                    request=request,
                    json={"error": {"message": "model not found"}},
                )
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": '{"scene":"office"}'}}]},
            )

    http_client = RecordingHttpClient()
    client = GemmaClient(
        GemmaModelConfig(
            api_key="key",
            base_url=DEFAULT_FIREWORKS_BASE_URL,
            model="primary-model",
            fallback_models=("fallback-model",),
        ),
        client=http_client,
    )

    payload = client.chat_completion_json([{"role": "user", "content": "hello"}], temperature=0.1)

    assert payload == {"scene": "office"}
    assert http_client.models == ["primary-model", "fallback-model"]


def test_gemma_client_caches_first_working_model_after_fallback():
    class RecordingHttpClient:
        def __init__(self):
            self.models: list[str] = []

        def post(self, url, headers, json):
            self.models.append(json["model"])
            request = httpx.Request("POST", url, json=json, headers=headers)
            if json["model"] == "primary-model":
                return httpx.Response(
                    403,
                    request=request,
                    json={"error": {"message": "model unavailable"}},
                )
            return httpx.Response(
                200,
                request=request,
                json={"choices": [{"message": {"content": '{"scene":"office"}'}}]},
            )

    http_client = RecordingHttpClient()
    client = GemmaClient(
        GemmaModelConfig(
            api_key="key",
            base_url=DEFAULT_FIREWORKS_BASE_URL,
            model="primary-model",
            fallback_models=("fallback-model",),
        ),
        client=http_client,
    )

    first_payload = client.chat_completion_json([{"role": "user", "content": "first"}], temperature=0.1)
    second_payload = client.chat_completion_json([{"role": "user", "content": "second"}], temperature=0.1)

    assert first_payload == {"scene": "office"}
    assert second_payload == {"scene": "office"}
    assert http_client.models == ["primary-model", "fallback-model", "fallback-model"]


def test_normalize_captions_rejects_banned_speculation_phrase():
    captions = normalize_captions(
        {"formal": "A person probably waits beside traffic.", "sarcastic": "A worker seems to be standing by while traffic does the usual thing."},
        ("formal", "sarcastic"),
        make_evidence(),
    )

    assert captions["formal"] == "A person waits beside traffic."
    assert captions["sarcastic"] == "A worker is standing by while traffic does the usual thing."


def test_normalize_captions_softens_expanded_banned_phrases():
    captions = normalize_captions(
        {
            "formal": "A person seems calm beside traffic, as if hoping for a quiet crossing today.",
            "sarcastic": "A worker seemingly waits by the desk while the room does its usual dramatic office impression.",
        },
        ("formal", "sarcastic"),
        make_evidence(),
    )

    assert captions["formal"] == "A person looks calm beside traffic, while waiting for a quiet crossing today."
    assert captions["sarcastic"] == "A worker waits by the desk while the room does its usual dramatic office impression."


def test_normalize_captions_rejects_overlong_caption():
    overlong = (
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentytfour twentytfive "
        "twentysix twentyseven twentyeight twentynine thirty thirtyone thirtytwo thirtythree thirtyfour "
        "thirtyfive thirtysix thirtyseven thirtyeight thirtynine forty fortyone"
    )

    captions = normalize_captions(
        {"formal": overlong, "sarcastic": "A worker stands quietly while the room acts extremely impressed by ordinary office activity."},
        ("formal", "sarcastic"),
        make_evidence(),
    )

    assert len(captions["formal"].split()) == 40
    assert "fortyone" not in captions["formal"]


def test_normalize_captions_does_not_trim_caption_below_35_words():
    caption = (
        "A worker stands beside a desk in a quiet office while the camera follows the same deliberate routine, "
        "showing steady movements, nearby monitors, and a calm workspace without inventing extra drama, hidden "
        "dialogue, or unrelated action elsewhere."
    )

    captions = normalize_captions(
        {"formal": caption},
        ("formal",),
        make_evidence(),
    )

    assert captions["formal"] == caption
    assert len(captions["formal"].split()) >= 35


def test_normalize_captions_rejects_unsupported_humorous_tech_script_claim():
    captions = normalize_captions(
        {"humorous_tech": "A worker watches traffic like a script rerun that forgot to debug anything new for the developer today."},
        ("humorous_tech",),
        make_evidence(actions=["walking"]),
    )

    assert captions["humorous_tech"] == "A worker watches traffic like a process rerun that forgot to troubleshoot anything new for the worker today."


def test_normalize_captions_allows_humorous_tech_metaphor_without_coding_claim():
    captions = normalize_captions(
        {"humorous_tech": "Traffic moves like data packets hitting CPU limits during an unusually patient rush hour commute today."},
        ("humorous_tech",),
        make_evidence(actions=["moving"], setting="street"),
    )

    assert "humorous_tech" in captions


def test_normalize_captions_missing_style_key_still_raises():
    try:
        normalize_captions(
            {"formal": "A worker stands near a desk in a quiet office during a routine moment."},
            ("formal", "sarcastic"),
            make_evidence(),
        )
    except ValueError as exc:
        assert "style sarcastic" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to raise for a missing style key.")


def test_normalize_captions_empty_caption_still_raises():
    try:
        normalize_captions(
            {"formal": "   ", "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today."},
            ("formal", "sarcastic"),
            make_evidence(),
        )
    except ValueError as exc:
        assert "style formal" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to raise for an empty caption.")


def test_normalize_captions_rejects_useless_period_caption():
    try:
        normalize_captions(
            {"formal": ".", "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today."},
            ("formal", "sarcastic"),
            make_evidence(),
        )
    except ValueError as exc:
        assert "style formal" in str(exc)
    else:
        raise AssertionError("Expected normalize_captions to reject a punctuation-only caption.")


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

    assert captions["formal"] == "A worker stands here."
    assert captions["sarcastic"] == original["sarcastic"]


def test_invalid_verifier_output_empty_object_keeps_original_captions():
    class InvalidVerifierClient:
        def chat_completion_text(self, messages, temperature, use_response_format=None):
            return "{}"

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


def test_extract_caption_json_prefers_final_caption_object_after_thinking_process():
    text = """
Thinking Process:
{"scene":"office","actions":["typing"]}
More analysis here.
{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}
"""

    payload = extract_caption_json(text, ("formal", "sarcastic"))

    assert payload == {
        "formal": "A worker stands near a desk in a quiet office during a routine moment.",
        "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
    }


def test_extract_caption_json_ignores_evidence_json_when_style_keys_are_required():
    text = """
{"scene":"office","main_subjects":["person"],"actions":["typing"]}
{"formal":"A worker stands near a desk in a quiet office during a routine moment.","sarcastic":"A worker performs thrilling office stillness at an admirably ordinary pace today."}
"""

    payload = extract_caption_json(text, ("formal", "sarcastic"))

    assert "formal" in payload
    assert "scene" not in payload


def test_google_caption_response_with_fenced_json_parses_correctly():
    text = """
```json
{
  "formal": "A worker stands near a desk in a quiet office during a routine moment.",
  "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today."
}
```
"""

    payload = extract_caption_json(text, ("formal", "sarcastic"))

    assert payload["formal"] == "A worker stands near a desk in a quiet office during a routine moment."
    assert payload["sarcastic"] == "A worker performs thrilling office stillness at an admirably ordinary pace today."


def test_extract_evidence_json_rejects_empty_object():
    try:
        extract_evidence_json("{}")
    except ValueError as exc:
        assert "useful evidence JSON object" in str(exc)
    else:
        raise AssertionError("Expected extract_evidence_json to reject empty evidence objects.")


def test_extract_evidence_json_parses_qwen_style_fenced_json():
    text = """
```json
{
  "scene": "office scene",
  "main_subjects": ["person"],
  "actions": ["working"],
  "setting": "office",
  "visible_objects": ["desk"],
  "mood": "neutral",
  "camera_notes": "static shot",
  "uncertain_details": []
}
```
"""

    payload = extract_evidence_json(text)

    assert payload["scene"] == "office scene"
    assert payload["main_subjects"] == ["person"]


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
            pass

        def chat_completion_json(self, messages, temperature):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
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
            if "caption writer" in system_prompt:
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


def test_generate_captions_does_not_fallback_for_soft_caption_issues(tmp_path):
    class FakeClient:
        def __init__(self, _config):
            pass

        def chat_completion_json(self, messages, temperature):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
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
            return {
                "formal": "A worker probably stands near a desk in a quiet office during a routine moment.",
                "sarcastic": "A worker seems calm by the desk, as if hoping the office will become less thrilling today.",
            }

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMMA_API_KEY": "key",
            "GEMMA_BASE_URL": "https://example.com/v1",
            "GEMMA_MODEL": "gemma-test",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert captions != build_fallback_captions(("formal", "sarcastic"))
    assert captions["formal"] == "A worker stands near a desk in a quiet office during a routine moment."
    assert captions["sarcastic"] == "A worker looks calm by the desk, while waiting the office will become less thrilling today."


def test_generate_captions_non_json_response_triggers_repair_retry(tmp_path):
    class FakeClient:
        construction_index = 0
        caption_user_prompts: list[str] = []
        caption_attempts = 0

        def __init__(self, _config):
            self._config = _config
            self.kind = "vision" if FakeClient.construction_index == 0 else "text"
            FakeClient.construction_index += 1

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
                return json.dumps(
                    {
                        "scene": "office",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "uncertain_details": [],
                    }
                )

            FakeClient.caption_attempts += 1
            FakeClient.caption_user_prompts.append(messages[1]["content"])
            if FakeClient.caption_attempts == 1:
                return "not valid json at all"
            return json.dumps(
                {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            )

    debug_dir = tmp_path / "debug"
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        debug_dir=debug_dir,
        client_factory=FakeClient,
    )

    assert FakeClient.caption_attempts == 2
    assert "Previous raw model response" in FakeClient.caption_user_prompts[1]
    assert captions["formal"] == "A worker stands near a desk in a quiet office during a routine moment."
    assert (debug_dir / "clip-1_caption_attempt_1.txt").exists()
    assert (debug_dir / "clip-1_caption_attempt_2.txt").exists()


def test_generate_captions_uses_evidence_based_fallback_after_repeated_caption_failure(tmp_path):
    class FakeClient:
        construction_index = 0

        def __init__(self, _config):
            self._config = _config
            self.kind = "vision" if FakeClient.construction_index == 0 else "text"
            FakeClient.construction_index += 1

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
                return json.dumps(
                    {
                        "scene": "office",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "uncertain_details": [],
                    }
                )
            return "."

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "FIREWORKS_API_KEY": "fireworks-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert captions != build_fallback_captions(("formal", "sarcastic"))
    assert "worker" in captions["formal"]
    assert "standing" in captions["formal"]
    assert "fully processed" not in captions["formal"]


def test_google_caption_calls_use_response_format_false(tmp_path):
    class FakeClient:
        call_flags: list[bool | None] = []

        def __init__(self, _config):
            self._config = _config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            FakeClient.call_flags.append(use_response_format)
            if isinstance(messages[1]["content"], list):
                return json.dumps(
                    {
                        "scene": "office scene",
                        "main_subjects": ["worker"],
                        "actions": ["standing"],
                        "setting": "office",
                        "visible_objects": ["desk"],
                        "mood": "neutral",
                        "camera_notes": "static shot",
                        "temporal_progression": "The worker stays near the desk across the clip.",
                        "caption_focus": "worker standing near a desk",
                    }
                )
            return json.dumps(
                {
                    "formal": "A worker stands near a desk in a quiet office during a routine moment.",
                    "sarcastic": "A worker performs thrilling office stillness at an admirably ordinary pace today.",
                }
            )

    generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMINI_API_KEY": "gemini-key",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert FakeClient.call_flags == [False, False]


def test_generate_captions_uses_placeholder_when_config_missing(tmp_path):
    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={},
    )

    assert captions == build_placeholder_captions(("formal", "sarcastic"))


def test_generate_captions_force_fallback_skips_api_client_construction(tmp_path):
    class FailingClient:
        def __init__(self, _config):
            raise AssertionError("API client should not be constructed during forced fallback mode.")

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_FORCE_FALLBACK": "true"},
        client_factory=FailingClient,
    )

    assert captions == build_fallback_captions(("formal", "sarcastic"))


def test_generate_captions_force_placeholder_skips_api_client_construction(tmp_path):
    class FailingClient:
        def __init__(self, _config):
            raise AssertionError("API client should not be constructed during forced placeholder mode.")

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"GEMMACLIP_FORCE_PLACEHOLDER": "true"},
        client_factory=FailingClient,
    )

    assert captions == build_placeholder_captions(("formal", "sarcastic"))


def test_generate_captions_uses_generic_fallback_when_evidence_generation_fails(tmp_path):
    class FakeClient:
        def __init__(self, _config):
            self._config = _config

        def chat_completion_text(self, messages, temperature, use_response_format=None):
            return "{}"

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={"FIREWORKS_API_KEY": "fireworks-key"},
        client_factory=FakeClient,
    )

    assert captions == build_fallback_captions(("formal", "sarcastic"))


def test_generate_captions_falls_back_when_model_output_is_invalid(tmp_path):
    class FakeClient:
        def __init__(self, _config):
            pass

        def chat_completion_json(self, messages, temperature):
            system_prompt = messages[0]["content"]
            if "factual video analyst" in system_prompt:
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
                "formal": "."
            }

    captions = generate_captions(
        make_task(),
        make_frames(tmp_path),
        env={
            "GEMMA_API_KEY": "key",
            "GEMMA_BASE_URL": "https://example.com/v1",
            "GEMMA_MODEL": "gemma-test",
            "GEMMACLIP_DISABLE_VERIFIER": "true",
        },
        client_factory=FakeClient,
    )

    assert captions != build_fallback_captions(("formal", "sarcastic"))
    assert "worker" in captions["formal"]


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
