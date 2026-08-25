"""Rewrite host-local mlruns/ absolute paths to /app/mlruns for the Docker image.

Run inside the image build (Dockerfile) after COPY mlruns mlruns. Rewrites
POSIX-style host paths from Linux/WSL/macOS builds; Windows-native paths are
not handled.
"""

from pathlib import Path
import re

PATTERN = re.compile(r"/(?:home|Users)/[^\n\r]*?/mlruns")


def main() -> None:
    for path in Path("mlruns").rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = PATTERN.sub("/app/mlruns", text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
