from __future__ import annotations

import logging
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


def test_default_routed_models_are_gemma_for_every_role():
    config = load_gemma_config({
        "GEMMACLIP_PROVIDER": "routed_gemma",
        "FIREWORKS_API_KEY": "fw",
        "GOOGLE_API_KEY": "google",
    })
    models = [item.model.lower() for role in ("visual", "audio_visual", "caption") for item in config.role_configs(role)]
    assert models
    assert all("gemma" in model for model in models)


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


def test_selected_audio_route_uses_unified_and_cleans_selected_file(tmp_path, monkeypatch):
    task = Task("v1", "https://example.com/v.mp4", ("formal",))
    config = load_gemma_config({"GEMMACLIP_PROVIDER": "routed_gemma", "GOOGLE_API_KEY": "google"})
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr("gemmaclip.routed.prepare_audio_candidate", lambda *args, **kwargs: _audio(path=audio_path))
    calls = []
    responses = [
        '{"main_subjects":["a person"],"actions":["walking through a room"],"setting":"a room","audio":{"status":"usable","visual_consistency":"consistent","allowed_caption_facts":["A person speaks briefly."]}}',
        '{"formal":"A person walks through a room and speaks briefly while nearby furnishings remain visible throughout the short indoor scene."}',
    ]
    class Client:
        def __init__(self, model_config): self.model_config = model_config
        def chat_completion_text(self, messages, temperature, **kwargs):
            calls.append(self.model_config.model)
            return responses.pop(0)
    generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_AUDIO_MODE": "auto"}, client_factory=Client,
        remaining_time_fn=lambda: 500.0,
    )
    assert calls == [config.google_audio_visual_model, config.google_caption_model]
    assert not audio_path.exists()


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
    captions = generate_routed_captions(
        task, _frames(tmp_path), tmp_path / "video.mp4", config=config,
        env={"GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE": "0.9"},
        client_factory=Client, remaining_time_fn=lambda: 100.0,
    )
    assert temperatures == [0.9]
    assert set(captions) == {"formal"}


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
