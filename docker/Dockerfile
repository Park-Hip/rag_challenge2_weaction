FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# -r : recursive
# -f: force
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# -m: create home
# -r: system user
RUN useradd -m -r aiuser 

WORKDIR /app

COPY --chown=aiuser:aiuser ./src src/

COPY --chown=aiuser:aiuser ./config  config/

COPY --chown=aiuser:aiuser ./data data/

USER aiuser

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
