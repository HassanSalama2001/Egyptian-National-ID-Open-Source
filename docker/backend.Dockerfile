# Use an official Python runtime as a parent image
FROM python:3.11-slim

# System dependencies for OpenCV (headless) and PaddlePaddle's compiled
# backend - libgomp1 (OpenMP runtime) is easy to miss because it's
# usually already present on a full OS image (a GitHub Actions
# ubuntu-latest runner has it), so its absence only shows up on a
# genuinely minimal base image like this one: paddle failed to import
# with "ImportError: libgomp.so.1: cannot open shared object file"
# until this was added (confirmed by testing this exact base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package itself. Copying pyproject.toml (dependency list)
# separately from src/ lets Docker cache the (slow - paddlepaddle is a
# large download) dependency install layer across rebuilds that only
# change application code.
COPY pyproject.toml .
COPY src/egyptian_national_id_ocr/__init__.py src/egyptian_national_id_ocr/__init__.py
RUN pip install --no-cache-dir -e .

COPY src /app/src
COPY assets /app/assets

# Pre-download PaddleOCR's models at BUILD time, not on the container's
# first request - otherwise the first real request after every
# container start pays a slow, network-dependent model download
# (paddlex's own cache dir) before it can answer, which is a bad first
# impression for a freshly `docker run` container and an outright
# failure in a network-restricted deployment.
RUN python -c "from egyptian_national_id_ocr.ocr.paddle_ocr_engine import PaddleOCREngine; PaddleOCREngine()"

EXPOSE 8000

# Same invocation as local development (see README.md's HTTP API
# section: `uvicorn src.app:app`) - one documented way to run this,
# not a second one that could silently drift from it.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
