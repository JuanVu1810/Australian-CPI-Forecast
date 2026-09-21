"""Rewrite host-local mlruns/ absolute paths to /app/mlruns for the Docker image.

Run inside the image build (Dockerfile) after COPY mlruns mlruns. Windows-native
paths are not handled.
"""

from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml


def _source_roots(root: Path) -> set[str]:
    roots = set()
    for metadata in root.glob("*/meta.yaml"):
        experiment = yaml.safe_load(metadata.read_text(encoding="utf-8"))
        location = experiment.get("artifact_location") if isinstance(experiment, dict) else None
        if not isinstance(location, str):
            continue
        parsed = urlsplit(location)
        if parsed.scheme not in ("", "file"):
            continue
        source = Path(unquote(parsed.path if parsed.scheme else location)).parent
        if source.is_absolute() and source.name == root.name:
            roots.add(str(source))
    return roots


def main() -> None:
    root = Path("mlruns")
    source_roots = _source_roots(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for source in sorted(source_roots, key=len, reverse=True):
            updated = updated.replace(f"{source}/", "/app/mlruns/")
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
