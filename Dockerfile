# Sonic-RAG backend, sized for a 512MB free instance.
#
# Three decisions worth explaining, because each was measured rather than
# assumed:
#
#   Python 3.12, not 3.14. faiss-cpu publishes manylinux wheels for 3.10-3.13
#   only; on 3.14 pip would fall back to building FAISS from source, which is a
#   long and fragile way to arrive at the same library.
#
#   The index artifacts are baked into the image rather than fetched at boot.
#   They are ~775MB, and a free instance sleeps after 15 minutes -- downloading
#   that on every wake would add minutes to a cold start that is already the
#   worst part of the free tier.
#
#   EMBED_THREADS=1 is set here as well as in config, so it survives someone
#   running the image without an env file. On one shared vCPU the ONNX default
#   thread pool measured 223ms p50 and 1224ms p95 against 47ms and 50ms with a
#   single thread: the pool contends for the one core it has.

FROM node:20-slim AS frontend

WORKDIR /app/frontend
# Copy manifests alone first: dependencies change far less often than source,
# so this layer survives most rebuilds.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS backend

# curl for the artifact download, then removed -- it is a build tool, not a
# runtime dependency, and it does not need to ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Published as Release assets because both files are far past GitHub's 100MB
# per-file limit. --fail so an HTML error page becomes a failed build instead
# of a corrupt artifact that only breaks at startup.
ARG INDEX_RELEASE_URL=https://github.com/Surjune/Sonic-Rag/releases/download/index-10k
RUN mkdir -p backend/artifacts \
    && curl -fL --retry 3 -o backend/artifacts/vector_index.faiss "$INDEX_RELEASE_URL/vector_index.faiss" \
    && curl -fL --retry 3 -o backend/artifacts/chunks.db "$INDEX_RELEASE_URL/chunks.db" \
    && apt-get purge -y curl && apt-get autoremove -y

COPY backend/ ./backend/
COPY --from=frontend /app/frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBED_THREADS=1 \
    INDEX_QUANTIZED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed

# Download the embedding model at build time. Otherwise the first request after
# every cold start pays a ~90MB download before it can embed anything.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', threads=1)"

WORKDIR /app/backend

# Render supplies $PORT and expects the process to bind it on 0.0.0.0.
# One worker on purpose: a second would load its own copy of the index and
# double the memory on a host that has none to spare.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
