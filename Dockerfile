FROM python:3.11-slim

WORKDIR /app

# constraints.txt pins the packages to the versions the checked-in results were produced with
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY api api
COPY src src
COPY models models
COPY data/curated data/curated
# The credit stress test reads its PD/LGD/EAD assumption CSVs from data/metadata
COPY data/metadata data/metadata
# The API reads the interval and ensemble-weight reports at request time
COPY reports reports

# Cloud Run sets PORT; 8000 is the default for running locally
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
