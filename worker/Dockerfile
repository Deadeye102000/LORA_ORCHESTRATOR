FROM python:3.11-slim

WORKDIR /app

# System deps: gcc required by some torch/bitsandbytes builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to keep image size small for local testing.
# On a GPU host, replace this line with the CUDA-enabled torch wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Directory where adapter weights are saved before S3 upload
RUN mkdir -p /app/weights

CMD ["python", "daemon.py"]
