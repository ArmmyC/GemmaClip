from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemmaclip.audio import AudioEvidenceCandidate, AudioSettings
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_PROVIDER_AMD_CLOUD,
    DEFAULT_PROVIDER_FIREWORKS,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_ROUTED_GEMMA,
    load_gemma_config,
)
from gemmaclip.io import Task
from gemmaclip.captioner import build_fallback_captions
from gemmaclip.routed import (
    EvidenceExecution,
    EVIDENCE_ATTEMPT_MIN_SECONDS,
    FINAL_SYNTHESIS_MIN_SECONDS,
    FOCUSED_REPAIR_MIN_SECONDS,
    SINGLE_CALL_ATTEMPT_MIN_SECONDS,
    RouteDecision,
    RoutedRuntimeBudgetError,
    build_evidence_messages,
    build_final_caption_messages,
    build_focused_repair_messages,
    decide_evidence_route,
    empty_evidence,
    normalize_routed_evidence,
    generate_routed_captions,
    load_routed_stage_settings,
    _call_role_with_fallback,
    GENERATION_OUTCOME_DETERMINISTIC_FALLBACK,
    GENERATION_OUTCOME_EVIDENCE_FALLBACK,
    GENERATION_OUTCOME_MODEL,
)


def _audio(**changes):
    values = dict(path=Path("audio.wav"), available=True, energy_candidate=True, silent=False, start_seconds=4.0, duration_seconds=20.0, sample_rate=16000, rms=0.1, reason="selected")
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


def test_amd_cloud_override_replaces_only_the_audio_visual_fireworks_role():
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma",
        "FIREWORKS_API_KEY": "fw",
        "GOOGLE_API_KEY": "google",
        "AMD_GEMMA_AUDIO_VISUAL_API_KEY": "amd",
        "AMD_GEMMA_AUDIO_VISUAL_BASE_URL": "http://amd.example/v1",
        "AMD_GEMMA_AUDIO_VISUAL_MODEL": "gemma-4-12b-it",
    })

    assert config.has_credentials
    assert config.amd_audio_visual_configured
    assert [(item.provider, item.model) for item in config.role_configs("visual")] == [
        (DEFAULT_PROVIDER_FIREWORKS, config.fireworks_visual_model),
        (DEFAULT_PROVIDER_GOOGLE, config.google_visual_model),
    ]
    assert [(item.provider, item.model, item.base_url) for item in config.role_configs("audio_visual")] == [
        (DEFAULT_PROVIDER_AMD_CLOUD, "gemma-4-12b-it", "http://amd.example/v1"),
        (DEFAULT_PROVIDER_GOOGLE, config.google_audio_visual_model, None),
    ]
    assert [(item.provider, item.model) for item in config.role_configs("caption")] == [
        (DEFAULT_PROVIDER_FIREWORKS, config.fireworks_caption_model),
        (DEFAULT_PROVIDER_GOOGLE, config.google_caption_model),
    ]


def test_default_routed_models_are_gemma_for_every_role():
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma",
        "FIREWORKS_API_KEY": "fw",
        "GOOGLE_API_KEY": "google",
    })
    models = [item.model.lower() for role in ("visual", "audio_visual", "caption") for item in config.role_configs(role)]
    assert models
    assert all("gemma" in model for model in models)
    assert config.google_visual_model == "gemma-4-31b-it"
    assert config.google_caption_model == "gemma-4-31b-it"


def test_deterministic_audio_routing_modes_and_runtime():
    useful = _audio()
    silent = _audio(path=None, energy_candidate=False, silent=True)
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


def test_custom_word_bounds_are_consistent_in_system_and_user_prompts(tmp_path):
    messages = build_final_caption_messages(
        Task("v1", "video", ("formal",)),
        _frames(tmp_path),
        empty_evidence(),
        google=True,
        min_words=8,
        max_words=16,
    )
    system = messages[0]["content"]
    user = messages[1]["content"][-1]["text"]
    assert "8 to 16 words" in system
    assert "18 to 35 words" not in system
    assert "8-16" in user


def test_prompts_use_dynamic_frame_count(tmp_path):
    source = _frames(tmp_path)[0]
    frames = [ExtractedFrame(source.path, float(index)) for index in range(12)]
    evidence_prompt = build_evidence_messages("v1", frames, _audio(path=None, available=False), RouteDecision("visual", False, "off"), google=True)[1]["content"][12]["text"]
    caption_prompt = build_final_caption_messages(Task("v1", "video", ("formal",)), frames, empty_evidence(), google=True)
    assert "12 chronological frame timestamps" in evidence_prompt
    assert "12 chronological timestamps" in caption_prompt[1]["content"][12]["text"]
    assert "12 separate chronological frames" in caption_prompt[0]["content"]


def test_malformed_evidence_and_audio_status_are_normalized_safely():
    payload = {"scene": 42, "main_subjects": "bad", "audio": {"status": "invented", "allowed_caption_facts": ["hello"]}}
    normalized = normalize_routed_evidence(payload, _audio(), RouteDecision("audio_visual", True, "selected"))
    assert normalized["scene"] == "42"
    assert normalized["main_subjects"] == []
    assert normalized["audio"]["status"] == "uncertain"
    assert normalized["audio"]["allowed_caption_facts"] == []
    assert set(["sound", "dialogue", "speech", "music", "noise"]) <= set(normalized["unsupported_claim_types"])


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
    text = _call_role_with_fallback(
        "v1", role, config, Client, lambda provider: [], temperature=0,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        remaining_time_fn=lambda: 500.0, minimum_remaining_seconds=1.0,
    )
    assert text == '{"ok": true}'
    assert calls == [(DEFAULT_PROVIDER_FIREWORKS, fireworks_model), (DEFAULT_PROVIDER_GOOGLE, google_model)]


def test_amd_audio_visual_role_falls_back_to_google_visual_without_changing_other_roles():
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma",
        "FIREWORKS_API_KEY": "fw",
        "GOOGLE_API_KEY": "google",
        "AMD_GEMMA_AUDIO_VISUAL_API_KEY": "amd",
        "AMD_GEMMA_AUDIO_VISUAL_BASE_URL": "http://amd.example/v1",
        "AMD_GEMMA_AUDIO_VISUAL_MODEL": "gemma-4-12b-it",
    })
    calls = []

    class Client:
        def __init__(self, model_config):
            self.model_config = model_config

        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append((self.model_config.provider, self.model_config.model))
            if self.model_config.provider == DEFAULT_PROVIDER_AMD_CLOUD:
                raise RuntimeError("AMD deployment unavailable")
            return '{"ok": true}'

    text = _call_role_with_fallback(
        "v1", "audio_visual", config, Client, lambda provider: [], temperature=0,
        logger=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
        remaining_time_fn=lambda: 500.0, minimum_remaining_seconds=1.0,
    )

    assert text == '{"ok": true}'
    assert calls == [
        (DEFAULT_PROVIDER_AMD_CLOUD, "gemma-4-12b-it"),
        (DEFAULT_PROVIDER_GOOGLE, config.google_audio_visual_model),
    ]


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
    outcomes = []
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0, outcome_callback=outcomes.append)
    assert set(captions) == {"formal", "sarcastic"}
    assert calls == [config.google_visual_model, config.google_caption_model]
    assert outcomes == [GENERATION_OUTCOME_MODEL]


def test_final_failure_after_evidence_uses_grounded_evidence_fallback(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            if self.model_config.model == config.google_visual_model:
                return '{"main_subjects":["a worker"],"actions":["standing beside a desk"],"setting":"an office","visible_objects":["a computer"]}'
            raise RuntimeError("caption endpoint failed")
    outcomes = []
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0, outcome_callback=outcomes.append)
    assert "worker" in captions["formal"]
    assert "standing beside a desk" in captions["formal"]
    assert "fully processed" not in captions["formal"]
    assert outcomes == [GENERATION_OUTCOME_EVIDENCE_FALLBACK]


def test_both_providers_failing_evidence_returns_safe_fallback_without_secret_logs(tmp_path, caplog):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "secret-fireworks", "GOOGLE_API_KEY": "secret-google"})
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, *args, **kwargs): raise RuntimeError("request failed")
    outcomes = []
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0, outcome_callback=outcomes.append)
    assert captions == build_fallback_captions(task.styles)
    assert outcomes == [GENERATION_OUTCOME_DETERMINISTIC_FALLBACK]
    assert "secret-fireworks" not in caplog.text
    assert "secret-google" not in caplog.text
    assert "base64" not in caplog.text
    for field in (
        "task_id=", "operation=", "route=", "provider=", "model=", "attempt=",
        "status=", "elapsed_seconds=", "remaining_seconds=", "minimum_remaining_seconds=",
        "fallback_used=", "audio_available=", "audio_selected=", "audio_window_seconds=",
    ):
        assert field in caplog.text


def test_dockerfile_has_routed_configuration_without_literal_credentials():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "GEMMACLIP_AUDIO_MODE" in dockerfile
    assert "FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL" in dockerfile
    assert "GOOGLE_GEMMA_CAPTION_MODEL" in dockerfile
    assert "secret-fireworks" not in dockerfile
    assert "secret-google" not in dockerfile
    assert "GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE" in dockerfile
    assert "GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE" in dockerfile


def test_audio_preprocessing_is_skipped_below_threshold_and_visual_route_is_used(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    prepared = []
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", lambda *args, **kwargs: prepared.append(True))
    calls = []
    responses = [
        '{"main_subjects":["a person"],"actions":["walking through a room"],"setting":"a room"}',
        '{"formal":"A person walks through a room while nearby furnishings remain visible throughout the brief and otherwise uneventful indoor scene."}',
    ]
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            return responses.pop(0)
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: 150.0,
    )
    assert prepared == []
    assert calls == [config.google_visual_model, config.google_caption_model]
    assert set(captions) == {"formal"}


def test_audio_route_is_rejected_when_preprocessing_consumes_budget(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    clock = {"remaining": 180.0}
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    def prepare(*args, **kwargs):
        clock["remaining"] = 160.0
        return _audio(path=audio_path)
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", prepare)
    calls = []
    responses = [
        '{"main_subjects":["a person"],"actions":["walking through a room"],"setting":"a room"}',
        '{"formal":"A person walks through a room while nearby furnishings remain visible throughout the brief and otherwise uneventful indoor scene."}',
    ]
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            return responses.pop(0)
    generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: clock["remaining"],
    )
    assert calls[0] == config.google_visual_model
    assert config.google_audio_visual_model not in calls
    assert not audio_path.exists()


def test_post_audio_budget_between_65_and_130_degrades_to_single_visual_call(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    clock = {"remaining": 180.0}
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    def prepare(*args, **kwargs):
        clock["remaining"] = 100.0
        return _audio(path=audio_path)
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", prepare)
    calls = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            return '{"formal":"A person walks across a room while the visible furnishings remain still throughout this brief and straightforward indoor clip."}'
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: clock["remaining"],
    )
    assert calls == [config.google_visual_model]
    assert len(calls) == 1
    assert set(captions) == {"formal"}
    assert not audio_path.exists()


def test_caption_stage_minimums_exceed_default_sixty_second_request_timeout():
    assert FINAL_SYNTHESIS_MIN_SECONDS == 70.0
    assert FOCUSED_REPAIR_MIN_SECONDS == 70.0
    assert SINGLE_CALL_ATTEMPT_MIN_SECONDS == 70.0


def test_selected_audio_without_fireworks_uses_google_visual_and_cleans_selected_file(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", lambda *args, **kwargs: _audio(path=audio_path))
    calls = []
    responses = [
        '{"main_subjects":["a person"],"actions":["walking through a room"],"setting":"a room","audio":{"status":"usable","transcript":"do not keep","allowed_caption_facts":["A person speaks briefly."]}}',
        '{"formal":"A person walks through a room and speaks briefly while nearby furnishings remain visible throughout the short indoor scene."}',
    ]
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append((self.model_config.model, [part["type"] for part in messages[1]["content"]]))
            return responses.pop(0)
    generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: 500.0,
        debug_dir=tmp_path / "debug",
    )
    assert [model for model, _ in calls] == [config.google_visual_model, config.google_caption_model]
    assert config.google_audio_visual_model not in [model for model, _ in calls]
    assert "audio_file" not in calls[0][1] and "input_audio" not in calls[0][1]
    evidence = json.loads((tmp_path / "debug" / "v1_routed_evidence.json").read_text(encoding="utf-8"))
    assert evidence["audio"]["status"] == "unavailable"
    assert evidence["audio"]["transcript"] == ""
    assert evidence["audio"]["allowed_caption_facts"] == []
    assert not audio_path.exists()


def test_fireworks_audio_failure_falls_back_to_google_visual_without_audio(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google"})
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", lambda *args, **kwargs: _audio(path=audio_path))
    monkeypatch.setattr("gemmaclip.captioner.make_jpeg_data_url", lambda path: "data:image/jpeg;base64,anBlZw==")
    calls = []
    executions: list[EvidenceExecution] = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            parts = messages[1]["content"]
            calls.append((self.model_config.provider, self.model_config.model, [part["type"] for part in parts]))
            if self.model_config.provider == DEFAULT_PROVIDER_FIREWORKS and self.model_config.model == config.fireworks_audio_visual_model:
                raise RuntimeError("deployment unavailable")
            if self.model_config.provider == DEFAULT_PROVIDER_GOOGLE and messages[0]["content"].startswith("Return JSON"):
                return '{"main_subjects":["a person"],"actions":["walking"],"audio":{"status":"usable","transcript":"must disappear","allowed_caption_facts":["speech"]}}'
            return '{"formal":"A person walks through the visible indoor space while six sampled frames keep the grounded scene clear, concise, and straightforward."}'
    outcomes = []
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: 500.0, outcome_callback=outcomes.append,
        evidence_execution_callback=executions.append,
    )
    assert set(captions) == {"formal"}
    assert calls[0][1] == config.fireworks_audio_visual_model and "input_audio" in calls[0][2]
    assert calls[1][1] == config.google_visual_model and "input_audio" not in calls[1][2] and "audio_file" not in calls[1][2]
    assert executions == [EvidenceExecution(DEFAULT_PROVIDER_GOOGLE, config.google_visual_model, "visual", True, False, True, "The Fireworks audio-visual model was unavailable, so GemmaClip continued with Google Gemma 4 31B using frames only.")]
    assert outcomes == [GENERATION_OUTCOME_MODEL]
    assert not audio_path.exists()


def test_fireworks_unified_success_uses_audio_and_skips_google(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google"})
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", lambda *args, **kwargs: _audio(path=audio_path))
    monkeypatch.setattr("gemmaclip.captioner.make_jpeg_data_url", lambda path: "data:image/jpeg;base64,anBlZw==")
    calls = []
    executions = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append((self.model_config.provider, self.model_config.model, [part["type"] for part in messages[1]["content"]]))
            if self.model_config.model == config.fireworks_audio_visual_model:
                return '{"main_subjects":["a person"],"actions":["walking"],"audio":{"status":"usable","visual_consistency":"consistent","allowed_caption_facts":["brief speech"]}}'
            return '{"formal":"A person walks through the visible indoor space while speaking briefly, grounded by the selected audio and six chronological frames."}'
    generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client, remaining_time_fn=lambda: 500.0, evidence_execution_callback=executions.append)
    assert calls[0][0:2] == (DEFAULT_PROVIDER_FIREWORKS, config.fireworks_audio_visual_model)
    assert "input_audio" in calls[0][2]
    assert not any(provider == DEFAULT_PROVIDER_GOOGLE for provider, _, _ in calls)
    assert executions[0].audio_used is True and executions[0].fallback_used is False
    assert not audio_path.exists()


def test_malformed_fireworks_evidence_and_caption_json_fall_through_to_google(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google"})
    calls = []
    monkeypatch.setattr("gemmaclip.captioner.make_jpeg_data_url", lambda path: "data:image/jpeg;base64,anBlZw==")
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append((self.model_config.provider, self.model_config.model))
            if self.model_config.provider == DEFAULT_PROVIDER_FIREWORKS:
                return "not valid json"
            if self.model_config.provider == DEFAULT_PROVIDER_GOOGLE and messages[0]["content"].startswith("Return JSON"):
                return '{"main_subjects":["a person"],"actions":["walking"]}'
            return '{"formal":"A person walks through the visible scene while six chronological frames provide enough grounded detail for this concise and accurate caption."}'
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0)
    assert set(captions) == {"formal"}
    assert calls == [
        (DEFAULT_PROVIDER_FIREWORKS, config.fireworks_visual_model),
        (DEFAULT_PROVIDER_GOOGLE, config.google_visual_model),
        (DEFAULT_PROVIDER_FIREWORKS, config.fireworks_caption_model),
        (DEFAULT_PROVIDER_GOOGLE, config.google_caption_model),
    ]


def test_partial_fireworks_captions_are_preserved_while_google_receives_only_missing_styles(tmp_path, monkeypatch):
    styles = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
    task = Task("v1", "https://example.com/v.mp4", styles)
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "FIREWORKS_API_KEY": "fw", "GOOGLE_API_KEY": "google"})
    monkeypatch.setattr("gemmaclip.captioner.make_jpeg_data_url", lambda path: "data:image/jpeg;base64,anBlZw==")
    caption_prompts = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            prompt = messages[1]["content"][-1]["text"]
            if messages[0]["content"].startswith("Return JSON"):
                return '{"main_subjects":["a person"],"actions":["walking through a room"],"setting":"an indoor room"}'
            caption_prompts.append((self.model_config.provider, prompt))
            if self.model_config.provider == DEFAULT_PROVIDER_FIREWORKS:
                return '{"formal":"A person walks through an indoor room while six chronological frames preserve the visible setting and straightforward movement with clear detail."}'
            return '{"sarcastic":"A person boldly crosses an ordinary room, completing the legendary indoor journey while the furniture somehow survives another historic event.","humorous_tech":"A person deploys across the room with stable uptime, predictable routing, and no visible navigation exceptions during the brief indoor sequence.","humorous_non_tech":"A person crosses the room while the furniture enjoys front-row seats to the day most committed journey toward somewhere nearby."}'
    captions = generate_routed_captions(task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client, remaining_time_fn=lambda: 500.0)
    assert set(captions) == set(styles)
    assert captions["formal"].startswith("A person walks")
    assert [provider for provider, _ in caption_prompts] == [DEFAULT_PROVIDER_FIREWORKS, DEFAULT_PROVIDER_GOOGLE]
    google_prompt = caption_prompts[1][1]
    assert '"formal": "<18-35 word caption>"' not in google_prompt
    assert all(f'"{style}": "<18-35 word caption>"' in google_prompt for style in styles[1:])


@pytest.mark.parametrize(
    "role,operation,minimum",
    [
        ("visual", "evidence", EVIDENCE_ATTEMPT_MIN_SECONDS),
        ("audio_visual", "evidence", EVIDENCE_ATTEMPT_MIN_SECONDS),
        ("caption", "caption", FINAL_SYNTHESIS_MIN_SECONDS),
        ("caption", "focused_repair", FOCUSED_REPAIR_MIN_SECONDS),
        ("visual", "single_call", SINGLE_CALL_ATTEMPT_MIN_SECONDS),
    ],
)
def test_provider_fallback_stops_when_primary_consumes_live_budget(role, operation, minimum, caplog):
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma",
        "FIREWORKS_API_KEY": "fw",
        "GOOGLE_API_KEY": "google",
    })
    clock = {"remaining": minimum + 10.0}
    calls = []
    class Client:
        def __init__(self, model_config):
            calls.append(model_config.provider)
            self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            clock["remaining"] = minimum - 1.0
            raise RuntimeError("primary failed")
    with pytest.raises(RoutedRuntimeBudgetError):
        _call_role_with_fallback(
            "v1", role, config, Client, lambda provider: [], temperature=0.0,
            logger=logging.getLogger("test.runtime"),
            remaining_time_fn=lambda: clock["remaining"],
            minimum_remaining_seconds=minimum, operation=operation,
        )
    assert calls == [DEFAULT_PROVIDER_FIREWORKS]
    assert "status=skipped_runtime" in caplog.text
    assert "secret" not in caplog.text


def test_evidence_survives_when_final_synthesis_budget_is_unsafe(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    clock = {"remaining": 140.0}
    calls = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            clock["remaining"] = FINAL_SYNTHESIS_MIN_SECONDS - 1.0
            return '{"main_subjects":["a worker"],"actions":["standing beside a desk"],"setting":"an office","visible_objects":["a computer"]}'
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client,
        remaining_time_fn=lambda: clock["remaining"],
    )
    assert calls == [config.google_visual_model]
    assert "worker" in captions["formal"]
    assert "standing beside a desk" in captions["formal"]


def test_unsafe_focused_repair_preserves_valid_caption_and_fills_missing_style(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    clock = {"remaining": 140.0}
    valid_formal = "A worker stands beside a desk and computer while the surrounding office remains visible throughout the short indoor scene."
    calls = []
    responses = [
        '{"main_subjects":["a worker"],"actions":["standing beside a desk"],"setting":"an office","visible_objects":["a computer"]}',
        '{"formal":"' + valid_formal + '"}',
    ]
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            response = responses.pop(0)
            if len(calls) == 2:
                clock["remaining"] = FOCUSED_REPAIR_MIN_SECONDS - 1.0
            return response
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "off"}, client_factory=Client,
        remaining_time_fn=lambda: clock["remaining"],
    )
    assert len(calls) == 2
    assert captions["formal"] == valid_formal
    assert "worker" in captions["sarcastic"]
    assert set(captions) == {"formal", "sarcastic"}


def test_safe_focused_repair_requests_only_missing_style_and_uses_configured_temperatures(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal", "sarcastic"))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    valid_formal = "A worker stands beside a desk and computer while the surrounding office remains visible throughout the short indoor scene."
    responses = [
        '{"main_subjects":["a worker"],"actions":["standing beside a desk"],"setting":"an office"}',
        '{"formal":"' + valid_formal + '"}',
        '{"sarcastic":"A worker stands beside a desk, bravely turning an ordinary office pause into the most exclusive event on today’s calendar."}',
    ]
    temperatures = []
    prompts = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            temperatures.append(temperature)
            prompts.append(messages)
            return responses.pop(0)
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={
            "GEMMACLIP_AUDIO_MODE": "off",
            "GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE": "0.1",
            "GEMMACLIP_ROUTED_CAPTION_TEMPERATURE": "0.8",
            "GEMMACLIP_ROUTED_REPAIR_TEMPERATURE": "0.3",
        },
        client_factory=Client, remaining_time_fn=lambda: 500.0,
    )
    assert temperatures == [0.1, 0.8, 0.3]
    repair_prompt = prompts[2][1]["content"][-1]["text"]
    assert "Only repair these missing or invalid styles: sarcastic" in repair_prompt
    assert '"formal": "<18-35 word caption>"' not in repair_prompt
    assert captions["formal"] == valid_formal


def test_single_call_uses_configured_temperature_and_no_evidence_call(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    temperatures = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            temperatures.append(temperature)
            return '{"formal":"A person walks across a room while the visible furnishings remain still throughout this brief and straightforward indoor clip."}'
    outcomes = []
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE": "0.9"},
        client_factory=Client, remaining_time_fn=lambda: 100.0, outcome_callback=outcomes.append,
    )
    assert temperatures == [0.9]
    assert set(captions) == {"formal"}
    assert outcomes == [GENERATION_OUTCOME_MODEL]


def test_failed_single_call_reports_deterministic_fallback(tmp_path):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    outcomes = []
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, *args, **kwargs): raise RuntimeError("provider failed")
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config, env={},
        client_factory=Client, remaining_time_fn=lambda: 100.0, outcome_callback=outcomes.append,
    )
    assert captions == build_fallback_captions(task.styles)
    assert outcomes == [GENERATION_OUTCOME_DETERMINISTIC_FALLBACK]


def test_temperature_parsing_defaults_clamps_and_rejects_nonfinite_values():
    configured = load_routed_stage_settings({
        "GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE": "0.1",
        "GEMMACLIP_ROUTED_CAPTION_TEMPERATURE": "0.8",
        "GEMMACLIP_ROUTED_REPAIR_TEMPERATURE": "-1",
        "GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE": "3",
    })
    assert (configured.evidence_temperature, configured.caption_temperature) == (0.1, 0.8)
    assert configured.repair_temperature == 0.0
    assert configured.single_call_temperature == 2.0
    malformed = load_routed_stage_settings({
        "GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE": "bad",
        "GEMMACLIP_ROUTED_CAPTION_TEMPERATURE": "nan",
        "GEMMACLIP_ROUTED_REPAIR_TEMPERATURE": "inf",
    })
    assert malformed.evidence_temperature == 0.0
    assert malformed.caption_temperature == 0.4
    assert malformed.repair_temperature == 0.25
    assert malformed.single_call_temperature == 0.4


@pytest.mark.parametrize("visual_consistency", ["consistent", "unknown"])
def test_usable_audio_does_not_globally_block_safe_audio_categories(visual_consistency):
    normalized = normalize_routed_evidence(
        {"audio": {"status": "usable", "visual_consistency": visual_consistency, "allowed_caption_facts": ["A person says hello."]}},
        _audio(), RouteDecision("audio_visual", True, "selected"),
    )
    assert normalized["audio"]["allowed_caption_facts"] == ["A person says hello."]
    assert not set(["sound", "dialogue", "speech", "music", "noise"]) & set(normalized["unsupported_claim_types"])
    assert "motive" in normalized["unsupported_claim_types"]


@pytest.mark.parametrize("status", ["uncertain", "silent", "failed"])
def test_nonusable_audio_marks_all_audio_claim_categories_unsupported(status):
    normalized = normalize_routed_evidence(
        {"audio": {"status": status, "visual_consistency": "consistent", "allowed_caption_facts": ["unsafe"]}},
        _audio(), RouteDecision("audio_visual", True, "selected"),
    )
    assert set(["sound", "dialogue", "speech", "music", "noise"]) <= set(normalized["unsupported_claim_types"])
    assert normalized["audio"]["allowed_caption_facts"] == []


def test_contradictory_audio_blocks_categories_and_clears_allowed_facts():
    normalized = normalize_routed_evidence(
        {"audio": {"status": "usable", "visual_consistency": "contradictory", "allowed_caption_facts": ["unsafe"]}},
        _audio(), RouteDecision("audio_visual", True, "selected"),
    )
    assert set(["sound", "dialogue", "speech", "music", "noise"]) <= set(normalized["unsupported_claim_types"])
    assert normalized["audio"]["allowed_caption_facts"] == []


def test_unavailable_audio_marks_audio_claim_categories_unsupported():
    normalized = normalize_routed_evidence(
        {"audio": {"status": "usable", "allowed_caption_facts": ["unsafe"]}},
        _audio(path=None, available=False, energy_candidate=False), RouteDecision("visual", False, "unavailable"),
    )
    assert normalized["audio"]["status"] == "unavailable"
    assert set(["sound", "dialogue", "speech", "music", "noise"]) <= set(normalized["unsupported_claim_types"])
