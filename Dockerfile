# ============================================================================
#  AI 3D Structural Inspection System -- production container
# ============================================================================
#  Build:  docker build -t ai3d-inspection .
#  Run:    docker run -p 8000:8000 -v inspection-data:/var/data ai3d-inspection
#  Render: set the service to "Docker" and point it at this file; the port is
#          taken from $PORT, so Render's injected value is honoured as-is.
#
#  This image installs the native COLMAP CLI, so the reconstruction stage runs
#  the full feature_extractor -> exhaustive_matcher -> mapper chain and falls
#  back to the pure-OpenCV hybrid pipeline only if COLMAP is unavailable.
# ============================================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="AI 3D Structural Inspection System" \
      org.opencontainers.image.description="ESP32-CAM 8-view capture, COLMAP 3D reconstruction, AI defect analysis and a Three.js digital twin for concrete pier inspection" \
      org.opencontainers.image.source="https://github.com/dommetipavan26-tech/ai-3d-inspection-system"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    DATA_DIR=/var/data \
    OPENCV_IO_MAX_IMAGE_PIXELS=0 \
    OMP_NUM_THREADS=2

# ---------------------------------------------------------------------------
# 1. System packages
#    * colmap          -- native SfM back-end (Debian bookworm ships 3.7/3.8)
#    * libgl1/libglib  -- required by opencv even in the headless build for a
#                         few drawing primitives used by the report renderer
#    * libgomp1        -- OpenMP runtime for the OpenCV parallel filters
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        colmap \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tini \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# 2. Python dependencies (cached layer: requirements change far less often
#    than the application source)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Application source
# ---------------------------------------------------------------------------
COPY . .

# keep every generated artefact on the mounted volume so a deploy does not
# destroy an inspection session
RUN mkdir -p /var/data /app/data
VOLUME ["/var/data"]

# ---------------------------------------------------------------------------
# 4. Runtime
# ---------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=6s --start-period=25s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/healthz" || exit 1

# tini reaps the uvicorn worker so SIGTERM from the platform shuts down cleanly
ENTRYPOINT ["/usr/bin/tini", "--"]

# single worker: the pipeline keeps in-process state (job runner, material
# atlas, detector masks) and both the simulator and the reconstruction are
# CPU-bound, so scaling is done with more replicas, not more workers.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120 --limit-concurrency 32"]
