"""The bundled model runs restore into a working MLflow store and carry no local paths.

mlruns_snapshot/ is what lets a fresh clone skip training (`python -m src.mlruns_snapshot restore`). These tests fail if it
is rebuilt with a machine's paths in it, or if a served model family goes missing.
"""

from __future__ import annotations

import re

import pytest

from src import mlruns_snapshot
from src.models.registry import _latest_model_run_id

# an absolute home-directory path other than the neutral placeholder the builder writes
LOCAL_HOME_PATH = re.compile(r"/(?:home|Users)/(?!user\b)[^/\s]+")


@pytest.fixture(scope="module")
def restored(tmp_path_factory):
    dest = tmp_path_factory.mktemp("restore") / "mlruns"
    mlruns_snapshot.restore_store(dest=dest)
    return dest


def _experiment_dir(root):
    experiment_dirs = [p for p in root.iterdir() if p.is_dir() and (p / "meta.yaml").exists()]
    assert len(experiment_dirs) == 1, f"expected one experiment folder in {root}, found {experiment_dirs}"
    return experiment_dirs[0]


def test_store_text_files_have_no_local_paths_or_user_fields():
    files = list(mlruns_snapshot._text_files(mlruns_snapshot.STORE_DIR))
    assert files, "mlruns_snapshot/ is empty; run `python -m src.mlruns_snapshot build`"
    for path, text in files:
        assert not LOCAL_HOME_PATH.search(text), f"{path} contains a local home path"
        assert "/tmp/" not in text, f"{path} contains a temp path"
    assert not list(mlruns_snapshot.STORE_DIR.rglob("mlflow.user")), "the MLflow user tag should be dropped"
    for path, text in files:
        for line in text.splitlines():
            assert not re.match(r"user_id: [^']", line), f"{path} records a user id"


def test_every_served_family_is_saved_and_restores_to_a_finished_run(restored):
    client = mlruns_snapshot._client(restored)
    experiment = client.get_experiment_by_name(mlruns_snapshot.DEFAULT_EXPERIMENT_NAME)
    assert experiment is not None
    for tag in mlruns_snapshot.SERVED_FAMILY_TAGS:
        run_id = _latest_model_run_id(client, experiment.experiment_id, tag)
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        logged_model_id = run.data.tags.get("logged_model_id")
        if logged_model_id:
            assert (restored / experiment.experiment_id / "models" / logged_model_id / "artifacts").is_dir()


def test_restore_fills_in_every_placeholder(restored):
    experiment_dir = _experiment_dir(restored)
    for path, text in mlruns_snapshot._text_files(restored):
        assert mlruns_snapshot.MLRUNS_TOKEN not in text and mlruns_snapshot.ROOT_TOKEN not in text, path
    assert str(restored.resolve()) in (experiment_dir / "meta.yaml").read_text(encoding="utf-8")


def test_restore_refuses_a_non_empty_destination_unless_merging(tmp_path):
    dest = tmp_path / "mlruns"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit):
        mlruns_snapshot.restore_store(dest=dest)
    mlruns_snapshot.restore_store(dest=dest, merge=True)
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "existing"
    assert any(p.is_dir() for p in dest.iterdir())
