from pathlib import Path

import pytest

from gemmaclip.frames import ExtractedFrame
from gemmaclip.routed import EvidenceExecution
from gemmaclip.web.adapters import adapt_captions, adapt_evidence, adapt_frames, backend_style_to_frontend, frontend_style_to_backend, selected_route_from_evidence


def test_style_names_map_explicitly_both_directions():
    assert frontend_style_to_backend("humorous-tech") == "humorous_tech"
    assert backend_style_to_frontend("humorous_non_tech") == "humorous-non-tech"
    evidence = {"scene": "room", "audio": {"status": "usable", "visual_consistency": "consistent", "allowed_caption_facts": ["music plays"]}}
    adapted = adapt_captions({"formal": "A factual caption.", "humorous_tech": "A grounded tech joke."}, evidence)
    assert [item["style"] for item in adapted] == ["formal", "humorous-tech"]
    assert adapted[0]["groundingContext"] == {"visualEvidenceAvailable": True, "audioEvidenceAvailable": True}


def test_evidence_and_frames_are_browser_safe():
    evidence = adapt_evidence({"main_subjects": ["person"], "visible_objects": ["chair"], "camera_notes": "static", "audio": {"status": "usable", "visual_consistency": "contradictory", "allowed_caption_facts": ["music"]}}, selected_route="gemma-4-12b-unified", route_reason="usable audio")
    assert evidence["subjects"] == ["person"]
    assert evidence["visibleObjects"] == ["chair"]
    assert evidence["audio"]["visualConsistency"] == "contradictory"
    frames = adapt_frames("run_aaaaaaaaaaaaaaaaaaaa", [ExtractedFrame(Path("C:/private/frame.jpg"), 1.5, "dynamic", 0.8)])
    assert frames[0]["thumbnailUrl"].startswith("/api/")
    assert "private" not in frames[0]["thumbnailUrl"]


@pytest.mark.parametrize("status", ["usable", "uncertain", "silent", "failed"])
def test_available_audio_reports_actual_unified_route(status):
    route, reason = selected_route_from_evidence({"audio": {"available": True, "status": status}})
    assert route == "gemma-4-12b-unified"
    assert "12B Unified" in reason


def test_unavailable_audio_reports_visual_route():
    route, reason = selected_route_from_evidence({"audio": {"available": False, "status": "unavailable"}})
    assert route == "gemma-4-26b-a4b"
    assert "26B A4B" in reason


def test_execution_provenance_overrides_audio_inference():
    execution = EvidenceExecution("google", "gemma-4-31b-it", "visual", True, False, True, "Fireworks unavailable; frames only.")
    route, reason = selected_route_from_evidence({"audio": {"available": False}}, execution)
    adapted = adapt_evidence({"audio": {"available": False}}, selected_route=route, route_reason=reason, execution=execution)
    assert route == "gemma-4-31b"
    assert adapted["routeProvider"] == "google"
    assert adapted["routeModel"] == "Gemma 4 31B"
    assert adapted["routeModality"] == "visual"
    assert adapted["audioFallbackOccurred"] is True


def test_grounding_availability_never_claims_per_caption_use():
    adapted = adapt_captions({"formal": "A grounded caption."}, {"audio": {"status": "usable", "visual_consistency": "contradictory", "allowed_caption_facts": ["speech"]}})
    assert adapted[0]["groundingContext"] == {"visualEvidenceAvailable": False, "audioEvidenceAvailable": False}
    assert "evidenceUsed" not in adapted[0]
