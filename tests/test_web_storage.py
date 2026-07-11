import json
from datetime import UTC, datetime, timedelta

import pytest

from gemmaclip.web.storage import InvalidRunId, RunNotFound, RunStorage, UnsafeAsset, validate_run_id


def test_run_storage_uses_safe_ids_atomic_metadata_and_controlled_delete(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("../private clip.mp4", ".mp4", "video/mp4")
    assert run["id"].startswith("run_")
    assert run["video"]["filename"] == "private clip.mp4"
    assert "_uploadSuffix" not in run
    assert run["generationOutcome"] is None
    assert run["degraded"] is False
    storage.upload_path(run["id"], ".mp4").write_bytes(b"video")
    storage.update_run(run["id"], lambda value: value.update(status="processing"))
    assert json.loads((storage.run_dir(run["id"]) / "run.json").read_text())["status"] == "processing"
    storage.delete_run(run["id"])
    with pytest.raises(RunNotFound): storage.read_run(run["id"])


@pytest.mark.parametrize("value", ["../run_bad", "run_bad/path", "run_bad\\path", "run_short", "run_aaaaaaaaaaaaaaaaaaaa\n"])
def test_unsafe_run_ids_are_rejected(value):
    with pytest.raises(InvalidRunId): validate_run_id(value)


def test_asset_lookup_cannot_escape_run(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    with pytest.raises(UnsafeAsset): storage.frame_path(run["id"], "../secret.jpg")
    with pytest.raises(UnsafeAsset): storage.write_artifact_json(run["id"], "../../secret.json", {})


@pytest.mark.parametrize("initial_status", ["ready", "pending"])
def test_recovery_leaves_nonprocessing_runs_unchanged(tmp_path, initial_status):
    storage = RunStorage(tmp_path)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    storage.update_run(run["id"], lambda payload: payload.update(status=initial_status))
    assert storage.recover_interrupted_runs() == []
    assert storage.read_run(run["id"])["status"] == initial_status


def test_recovery_marks_processing_run_error_and_preserves_artifacts(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    artifact = storage.run_dir(run["id"]) / "frames" / "frame_001.jpg"; artifact.write_bytes(b"jpeg")
    storage.update_run(run["id"], lambda payload: payload.update(status="processing", activeStage="frames"))
    assert storage.recover_interrupted_runs() == [run["id"]]
    recovered = storage.read_run(run["id"])
    assert recovered["status"] == "error" and recovered["stages"]["frames"] == "error"
    assert recovered["error"] == "Processing was interrupted. Please start a new run."
    assert artifact.read_bytes() == b"jpeg"


def test_recovery_skips_malformed_metadata(tmp_path):
    storage = RunStorage(tmp_path)
    malformed = storage.root / "run_aaaaaaaaaaaaaaaaaaaa"; malformed.mkdir()
    (malformed / "run.json").write_text("not-json")
    assert storage.recover_interrupted_runs() == []


@pytest.mark.parametrize("status", ["ready", "error", "pending"])
def test_cleanup_deletes_expired_completed_run(tmp_path, status):
    now = datetime.now(UTC)
    storage = RunStorage(tmp_path, run_ttl_seconds=60)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    storage.update_run(run["id"], lambda payload: payload.update(status=status, createdAt=(now - timedelta(seconds=61)).isoformat()))
    assert storage.cleanup_expired_runs(now=now) == [run["id"]]
    with pytest.raises(RunNotFound): storage.read_run(run["id"])


def test_cleanup_retains_recent_and_active_runs(tmp_path):
    now = datetime.now(UTC)
    storage = RunStorage(tmp_path, run_ttl_seconds=60)
    recent = storage.create_run("recent.mp4", ".mp4", "video/mp4")
    active = storage.create_run("active.mp4", ".mp4", "video/mp4")
    storage.update_run(active["id"], lambda payload: payload.update(status="pending", createdAt=(now - timedelta(days=2)).isoformat()))
    assert storage.cleanup_expired_runs({active["id"]}, now=now) == []
    assert storage.read_run(recent["id"])["status"] == "pending"
    assert storage.read_run(active["id"])["status"] == "pending"


def test_nonpositive_ttl_disables_cleanup(tmp_path):
    storage = RunStorage(tmp_path, run_ttl_seconds=0)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    storage.update_run(run["id"], lambda payload: payload.update(status="ready", createdAt="2000-01-01T00:00:00+00:00"))
    assert storage.cleanup_expired_runs() == []
    assert storage.read_run(run["id"])["status"] == "ready"
