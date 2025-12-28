FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download Demucs model to reduce cold starts
RUN python - << 'PY'
from demucs.pretrained import get_model
get_model('htdemucs')
PY

# Handler
COPY handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "/app/handler.py"]
