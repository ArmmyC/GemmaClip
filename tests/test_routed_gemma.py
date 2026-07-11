from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaclip.audio import AudioEvidenceCandidate, AudioSettings
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_PROVIDER_FIREWORKS,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_ROUTED_GEMMA,
    load_gemma_config,
)
from gemmaclip.io import Task
from gemmaclip.captioner import build_fallback_captions
from gemmaclip.routed import (
    RouteDecision,
    build_evidence_messages,
    build_final_caption_messages,
    build_focused_repair_messages,
    decide_evidence_route,
    empty_evidence,
    normalize_routed_evidence,
    generate_routed_captions,
    _call_role_with_fallback,
)


def _audio(**changes):
    values = dict(path=Path("audio.wav"), available=True, speech_candidate=True, silent=False, start_seconds=4.0, duration_seconds=20.0, sample_rate=16000, rms=0.1, reason="selected")
    values.update(changes)
    return AudioEvidenceCandidate(**values)


def _frames(tmp_path):
    result = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"jpeg")
        result.append(ExtractedFrame(path, float(index), "anchor" if index < 4 else "dynamic"))
    return result


def test_routed_provider_config_preserves_same_role_models_and_credential_combinations():
    env = {
        "GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google",
        "FIREWORKS_GEMMA_VISUAL_MODEL": "fw-26", "FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL": "fw-12", "FIREWORKS_GEMMA_CAPTION_MODEL": "fw-31",
        "GOOGLE_GEMMA_VISUAL_MODEL": "g-26", "GOOGLE_GEMMA_AUDIO_VISUAL_MODEL": "g-12", "GOOGLE_GEMMA_CAPTION_MODEL": "g-31",
    }
    config = load_gemma_config(env)
    assert config.provider == DEFAULT_PROVIDER_ROUTED_GEMMA
    assert [(item.provider, item.model) for item in config.role_configs("visual")] == [(DEFAULT_PROVIDER_FIREWORKS, "fw-26"), (DEFAULT_PROVIDER_GOOGLE, "g-26")]
    assert [item.model for item in config.role_configs("audio_visual")] == ["fw-12", "g-12"]
    assert [item.model for item in config.role_configs("caption")] == ["fw-31", "g-31"]
    assert len(load_gemma_config({**env, "GOOGLE_API_KEY": ""}).role_configs("caption")) == 1
    assert load_gemma_config({**env, "FIREWORKS_API_KEY": ""}).role_configs("caption")[0].provider == DEFAULT_PROVIDER_GOOGLE
    assert not load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma"}).has_credentials


def test_deterministic_audio_routing_modes_and_runtime():
    useful = _audio()
    silent = _audio(path=None, speech_candidate=False, silent=True)
    assert decide_evidence_route(AudioSettings(mode="off"), useful, 500).route == "visual"
    assert decide_evidence_route(AudioSettings(mode="auto"), silent, 500).route == "visual"
    assert decide_evidence_route(AudioSettings(mode="auto"), useful, 500).route == "audio_visual"
    assert decide_evidence_route(AudioSettings(mode="always"), useful, 500).route == "audio_visual"
    assert decide_evidence_route(AudioSettings(mode="auto", min_remaining_seconds=130), useful, 100).route == "visual"


def test_visual_message_parts_put_images_before_text_and_forbid_audio_claims(tmp_path):
    frames = _frames(tmp_path)
    messages = build_evidence_messages("v1", frames, _audio(path=None, available=False), RouteDecision("visual", False, "off"), google=True)
    parts = messages[1]["content"]
    assert [part["type"] for part in parts[:6]] == ["image_file"] * 6
    assert parts[6]["type"] == "text"
    assert "No audio was provided" in parts[6]["text"]
    assert "Never infer sound" in parts[6]["text"]
    assert "chain-of-thought" in parts[6]["text"]


def test_audio_visual_message_order_is_images_text_audio(tmp_path):
    frames = _frames(tmp_path)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")
    messages = build_evidence_messages("v1", frames, _audio(path=audio_path), RouteDecision("audio_visual", True, "selected"), google=True)
    types = [part["type"] for part in messages[1]["content"]]
    assert types == ["image_file"] * 6 + ["text", "audio_file"]
    assert "start=4.000s duration=20.000s" in messages[1]["content"][6]["text"]


def test_final_prompt_has_exact_keys_frames_evidence_and_audio_gate(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    messages = build_final_caption_messages(task, _frames(tmp_path), empty_evidence(), google=True)
    parts = messages[1]["content"]
    assert [part["type"] for part in parts[:6]] == ["image_file"] * 6
    prompt = parts[-1]["text"]
    assert '"formal": "<18-35 word caption>"' in prompt
    assert '"sarcastic": "<18-35 word caption>"' in prompt
    assert "allowed_caption_facts" in prompt
    assert "audio.status is usable" in prompt
    assert "scores" not in prompt


def test_malformed_evidence_and_audio_status_are_normalized_safely():
    payload = {"scene": 42, "main_subjects": "bad", "audio": {"status": "invented", "allowed_caption_facts": ["hello"]}}
    normalized = normalize_routed_evidence(payload, _audio(), RouteDecision("audio_visual", True, "selected"))
    assert normalized["scene"] == "42"
    assert normalized["main_subjects"] == []
    assert normalized["audio"]["status"] == "uncertain"
    assert normalized["audio"]["allowed_caption_facts"] == []


def test_focused_repair_requests_only_missing_styles(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    messages = build_focused_repair_messages(task, ["sarcastic"], {"formal": "A person walks through a room."}, _frames(tmp_path), empty_evidence(), google=True)
    prompt = messages[1]["content"][-1]["text"]
    assert '"sarcastic": "<18-35 word caption>"' in prompt
    assert '"formal": "<18-35 word caption>"' not in prompt
    assert "Do not return or rewrite valid styles" in prompt


@pytest.mark.parametrize("role,fireworks_model,google_model", [
    ("visual", "fw-26", "g-26"),
    ("audio_visual", "fw-12", "g-12"),
    ("caption", "fw-31", "g-31"),
])
def test_each_model_role_falls_back_from_fireworks_to_google(role, fireworks_model, google_model):
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google",
        "FIREWORKS_GEMMA_VISUAL_MODEL": "fw-26", "FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL": "fw-12", "FIREWORKS_GEMMA_CAPTION_MODEL": "fw-31",
        "GOOGLE_GEMMA_VISUAL_MODEL": "g-26", "GOOGLE_GEMMA_AUDIO_VISUAL_MODEL": "g-12", "GOOGLE_GEMMA_CAPTION_MODEL": "g-31",
    })
    calls = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append((self.model_config.provider, self.model_config.model))
            if self.model_config.provider == DEFAULT_PROVIDER_FIREWORKS:
                raise RuntimeError("primary failed")
            return '{"ok": true}'
    text = _call_role_with_fallback("v1", role, config, Client, lambda provider: [], temperature=0, logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None))
    assert text == '{"ok": true}'
    assert calls == [(DEFAULT_PROVIDER_FIREWORKS, fireworks_model), (DEFAULT_PROVIDER_GOOGLE, google_model)]


def test_normal_successful_routed_path_uses_exactly_two_model_calls(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    responses = [
        '{"scene":"room","main_subjects":["a person"],"actions":["walks across the room"],"setting":"a room","verified_description":"A person walks across a room."}',
        '{"formal":"A person walks steadily across a room while nearby furnishings remain visible around the simple indoor scene throughout the brief clip.","sarcastic":"A person crosses the room with the heroic efficiency normally reserved for reaching the other side of a room."}',
    ]
    calls = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            return responses.pop(0)
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0)
    assert set(captions) == {"formal", "sarcastic"}
    assert calls == [config.google_visual_model, config.google_caption_model]


def test_final_failure_after_evidence_uses_grounded_evidence_fallback(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            if self.model_config.model == config.google_visual_model:
                return '{"main_subjects":["a worker"],"actions":["standing beside a desk"],"setting":"an office","visible_objects":["a computer"]}'
            raise RuntimeError("caption endpoint failed")
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0)
    assert "worker" in captions["formal"]
    assert "standing beside a desk" in captions["formal"]
    assert "fully processed" not in captions["formal"]


def test_both_providers_failing_evidence_returns_safe_fallback_without_secret_logs(tmp_path, caplog):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "secret-fireworks", "GOOGLE_API_KEY": "secret-google"})
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, *args, **kwargs): raise RuntimeError("request failed")
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0)
    assert captions == build_fallback_captions(task.styles)
    assert "secret-fireworks" not in caplog.text
    assert "secret-google" not in caplog.text
    assert "base64" not in caplog.text


def test_dockerfile_has_routed_configuration_without_literal_credentials():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "GEMMACLIP_AUDIO_MODE" in dockerfile
    assert "FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL" in dockerfile
    assert "GOOGLE_GEMMA_CAPTION_MODEL" in dockerfile
    assert "secret-fireworks" not in dockerfile
    assert "secret-google" not in dockerfile
