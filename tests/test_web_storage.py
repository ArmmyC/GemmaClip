import json

import pytest

from gemmaclip.web.storage import InvalidRunId, RunNotFound, RunStorage, UnsafeAsset, validate_run_id


def test_run_storage_uses_safe_ids_atomic_metadata_and_controlled_delete(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("../private clip.mp4", ".mp4", "video/mp4")
    assert run["id"].startswith("run_")
    assert run["video"]["filename"] == "private clip.mp4"
    assert "_uploadSuffix" not in run
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
