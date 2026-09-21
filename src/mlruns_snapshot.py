"""Bundle the MLflow runs the API serves, so a fresh clone can skip training.

``python -m src.mlruns_snapshot build`` copies the latest finished run of each served model family from ``mlruns/`` into
``mlruns_snapshot/``. In the text metadata, absolute paths from this machine are replaced by placeholders and the MLflow
user fields are dropped. The pickled models are copied as they are; one of them still holds the author's Python
install path in code-object debug info, which is harmless and not something to patch inside a pickle.

``python -m src.mlruns_snapshot restore`` copies ``mlruns_snapshot/`` into ``mlruns/`` and fills the placeholders in for the
new location. The API then serves the same models as before, with no training step.

Retraining afterwards is unaffected: new runs are newer, so the API picks them up as the latest finished run.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from src.models.registry import _latest_model_run_id
from src.models.tracking import DEFAULT_EXPERIMENT_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
STORE_DIR = PROJECT_ROOT / "mlruns_snapshot"

# The ``model_family`` tags the API asks for: headline, then the trimmed-mean equivalents (see src/models/registry.py).
SERVED_FAMILY_TAGS = (
    "sarima",
    "elastic_net",
    "ensemble",
    "trimmed_mean_sarima",
    "trimmed_mean_elastic_net",
    "trimmed_mean_ensemble",
)

MLRUNS_TOKEN = "__MLRUNS_ROOT__"
ROOT_TOKEN = "__PROJECT_ROOT__"
DROPPED_TAG_FILES = ("mlflow.user",)
BINARY_SUFFIXES = {".pkl", ".pickle", ".statsmodels", ".joblib", ".npy", ".npz", ".png"}
HOME_PATH = re.compile(r"/(?:home|Users)/[^/\s'\"]+")
USER_ID_LINE = re.compile(r"^user_id: .*$", re.MULTILINE)


def _text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in BINARY_SUFFIXES:
            continue
        try:
            yield path, path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue


def _client(mlruns_dir: Path):
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return MlflowClient(tracking_uri=str(mlruns_dir))


def build_store(mlruns_dir: Path = MLRUNS_DIR, store_dir: Path = STORE_DIR) -> list[tuple[str, str]]:
    """Copy the served runs (and their logged models) into ``store_dir``; return ``(family tag, run id)`` pairs."""
    client = _client(mlruns_dir)
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"No {DEFAULT_EXPERIMENT_NAME!r} experiment in {mlruns_dir}; train the models first.")
    source = mlruns_dir / experiment.experiment_id
    target = store_dir / experiment.experiment_id

    if store_dir.exists():
        shutil.rmtree(store_dir)
    target.mkdir(parents=True)
    shutil.copy2(source / "meta.yaml", target / "meta.yaml")

    kept = []
    for tag in SERVED_FAMILY_TAGS:
        run_id = _latest_model_run_id(client, experiment.experiment_id, tag)
        shutil.copytree(source / run_id, target / run_id)
        logged_model_id = client.get_run(run_id).data.tags.get("logged_model_id")
        if logged_model_id:
            shutil.copytree(source / "models" / logged_model_id, target / "models" / logged_model_id)
        kept.append((tag, run_id))

    replacements = [(str(mlruns_dir.resolve()), MLRUNS_TOKEN), (str(PROJECT_ROOT), ROOT_TOKEN)]
    for path in sorted(store_dir.rglob("*")):
        if path.is_file() and path.name in DROPPED_TAG_FILES and path.parent.name == "tags":
            path.unlink()
    for path, text in _text_files(store_dir):
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        updated = USER_ID_LINE.sub("user_id: ''", HOME_PATH.sub("/home/user", updated))
        if path.name == "mlflow.source.name" and updated.strip().startswith("/"):
            updated = Path(updated.strip()).name      # a launcher script outside the project: keep only its file name
        if updated != text:
            path.write_text(updated, encoding="utf-8")
    return kept


def restore_store(dest: Path = MLRUNS_DIR, store_dir: Path = STORE_DIR, merge: bool = False) -> None:
    """Copy ``store_dir`` into ``dest`` and fill the placeholders in for that location."""
    if not store_dir.is_dir():
        raise SystemExit(f"{store_dir} not found; nothing to restore.")
    if dest.exists() and any(dest.iterdir()) and not merge:
        raise SystemExit(f"{dest} already has content. Pass --merge to add the saved runs alongside it.")
    shutil.copytree(store_dir, dest, dirs_exist_ok=True)
    replacements = [(MLRUNS_TOKEN, str(dest.resolve())), (ROOT_TOKEN, str(PROJECT_ROOT))]
    for path, text in _text_files(dest):
        if MLRUNS_TOKEN in text or ROOT_TOKEN in text:
            for old, new in replacements:
                text = text.replace(old, new)
            path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or restore the bundled MLflow runs the API serves.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="copy the served runs from mlruns/ into mlruns_snapshot/")
    build.add_argument("--mlruns", type=Path, default=MLRUNS_DIR)
    build.add_argument("--store", type=Path, default=STORE_DIR)
    restore = sub.add_parser("restore", help="copy mlruns_snapshot/ into mlruns/")
    restore.add_argument("--dest", type=Path, default=MLRUNS_DIR)
    restore.add_argument("--store", type=Path, default=STORE_DIR)
    restore.add_argument("--merge", action="store_true", help="allow restoring into a non-empty mlruns/")
    args = parser.parse_args(argv)

    if args.command == "build":
        kept = build_store(args.mlruns, args.store)
        print(f"Saved {len(kept)} runs to {args.store}:")
        for tag, run_id in kept:
            print(f"  {tag}: {run_id}")
    else:
        restore_store(args.dest, args.store, args.merge)
        print(f"Restored the saved runs into {args.dest}")


if __name__ == "__main__":
    main()
