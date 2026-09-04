# CUDA runtime image with cuDNN 9 already on the library path — no pip cu12
# wheels or LD_LIBRARY_PATH juggling needed.
# Pinned to CUDA 12.6 (not bleeding-edge 13.x): CTranslate2 / faster-whisper are
# tested against 12.x. Driver >= 525 runs it; 595.91.07 is fine.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Persist the downloaded large-v3 model across runs:
    HF_HOME=/models

# python3.12 is the system python on ubuntu 24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Neutral non-root default; deployments override via compose `user:`.
# /models and /data are just skeleton dirs for bare runs — ownership of
# mounted paths comes from the host, not the image.
RUN useradd --create-home appuser \
    && mkdir -p /models /data
USER appuser

# Interactive shell entrypoint — run the kotodama CLI inside yourself:
#   docker run -it --gpus all -v /media:/data -v /models:/models \
#     kotodama
#   $ kotodama /data/video.mp4 -c /app/config.toml
# (falls back to CPU+int8 automatically if no GPU is visible)
ENTRYPOINT ["/bin/bash"]
