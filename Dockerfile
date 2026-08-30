FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    MODEL_PATH=/app/artifacts/conditioned-v2-best.pt

ARG MODEL_URL=""
ARG MODEL_SHA256=""

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY training ./training
COPY web ./web
COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir -p /app/artifacts \
    && if [ -n "$MODEL_URL" ]; then \
         python -m scripts.download_model \
           --url "$MODEL_URL" \
           --destination "$MODEL_PATH" \
           --sha256 "$MODEL_SHA256"; \
       fi \
    && addgroup --system motif \
    && adduser --system --ingroup motif motif \
    && chown -R motif:motif /app
USER motif

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=3)"

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
