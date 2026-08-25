# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV MLFLOW_TRACKING_URI=mlruns
ENV MLFLOW_EXPERIMENT_NAME="CPI Forecast"
ENV MLFLOW_ALLOW_FILE_STORE=true

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api api
COPY src src
COPY data/curated data/curated
# .dockerignore excludes reports/* except the handful of CSVs api/main.py
# and ensemble.py read at request time (interval calibration/coverage
# diagnostics, headline's dynamic ensemble weight source) -- this only
# copies those.
COPY reports reports
COPY mlruns mlruns
COPY scripts/rewrite_mlruns_paths.py scripts/rewrite_mlruns_paths.py
# Rewrites POSIX-style host paths from Linux/WSL/macOS builds; Windows-native
# paths are not handled. A plain script file (not a heredoc) so this builds
# on Cloud Build's legacy gcr.io/cloud-builders/gcb-internal step too, which
# does not honor the BuildKit `# syntax=` directive or heredoc RUN syntax.
RUN python scripts/rewrite_mlruns_paths.py

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3).read()"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
