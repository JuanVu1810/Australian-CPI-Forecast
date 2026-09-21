"""The Docker build must relocate restored MLflow artifacts from any POSIX checkout."""

import pytest

from scripts import rewrite_mlruns_paths


@pytest.mark.parametrize(
    "source_root,location",
    [
        ("/tmp/project/mlruns", "/tmp/project/mlruns/123"),
        ("/home/user/project/mlruns", "/home/user/project/mlruns/123"),
        ("/Users/user/project/mlruns", "file:///Users/user/project/mlruns/123"),
    ],
)
def test_rewrite_paths_from_experiment_metadata(tmp_path, monkeypatch, source_root, location):
    experiment = tmp_path / "mlruns" / "123"
    experiment.mkdir(parents=True)
    (experiment / "meta.yaml").write_text(
        f"artifact_location: {location}\nexperiment_id: '123'\n", encoding="utf-8"
    )
    model = experiment / "models" / "m-1"
    model.mkdir(parents=True)
    metadata = model / "meta.yaml"
    metadata.write_text(
        f"artifact_location: {source_root}/123/models/m-1/artifacts\n", encoding="utf-8"
    )
    binary = model / "model.pkl"
    binary.write_bytes(b"\xff\xfe" + source_root.encode())

    monkeypatch.chdir(tmp_path)
    rewrite_mlruns_paths.main()

    assert "/app/mlruns/123" in (experiment / "meta.yaml").read_text(encoding="utf-8")
    assert metadata.read_text(encoding="utf-8") == (
        "artifact_location: /app/mlruns/123/models/m-1/artifacts\n"
    )
    assert binary.read_bytes() == b"\xff\xfe" + source_root.encode()
