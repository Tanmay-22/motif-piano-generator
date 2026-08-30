FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY training ./training
COPY web ./web
COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN addgroup --system motif && adduser --system --ingroup motif motif \
    && mkdir -p /app/artifacts && chown -R motif:motif /app
USER motif

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=3)"

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]

