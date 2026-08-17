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
COPY mlruns mlruns

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3).read()"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
