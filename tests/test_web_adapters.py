from pathlib import Path

from gemmaclip.frames import ExtractedFrame
from gemmaclip.web.adapters import adapt_captions, adapt_evidence, adapt_frames, backend_style_to_frontend, frontend_style_to_backend


def test_style_names_map_explicitly_both_directions():
    assert frontend_style_to_backend("humorous-tech") == "humorous_tech"
    assert backend_style_to_frontend("humorous_non_tech") == "humorous-non-tech"
    assert [item["style"] for item in adapt_captions({"formal": "A factual caption.", "humorous_tech": "A grounded tech joke."})] == ["formal", "humorous-tech"]


def test_evidence_and_frames_are_browser_safe():
    evidence = adapt_evidence({"main_subjects": ["person"], "visible_objects": ["chair"], "camera_notes": "static", "audio": {"status": "usable", "visual_consistency": "contradictory", "allowed_caption_facts": ["music"]}}, selected_route="gemma-4-12b-unified", route_reason="usable audio")
    assert evidence["subjects"] == ["person"]
    assert evidence["visibleObjects"] == ["chair"]
    assert evidence["audio"]["visualConsistency"] == "contradictory"
    frames = adapt_frames("run_aaaaaaaaaaaaaaaaaaaa", [ExtractedFrame(Path("C:/private/frame.jpg"), 1.5, "dynamic", 0.8)])
    assert frames[0]["thumbnailUrl"].startswith("/api/")
    assert "private" not in frames[0]["thumbnailUrl"]
