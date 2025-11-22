# ==========================================================
#  AI-DAC / TLL Base Image (GPU + CPU compatible)
#  Author: William K
# ==========================================================

# ---------- 1) Base Image (CUDA + Python)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# System-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        build-essential git wget curl \
        libssl-dev libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Ensure python3 is default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# ---------- 2) Python dependencies (GPU-compatible)
FROM base AS builder

WORKDIR /opt/aida

# Copy requirement file if you have one (recommended)
# COPY requirements.txt ./

# Example: install ML, RL, GAN, DB adapters
RUN pip install --upgrade pip && pip install \
        torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 \
            --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
        numpy scipy scikit-learn pandas matplotlib seaborn \
        faiss-cpu \
        gymnasium==0.29.1 \
        tqdm pyyaml \
        transformers==4.39.0 \
        sentencepiece \
        python-dotenv \
        psycopg2-binary \
        pymssql \
        opencv-python \
        rich \
        && rm -rf /root/.cache/pip

# ---------- 3) APP STAGE
FROM base AS runtime

WORKDIR /opt/aida

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy AI-DAC / TLL project files
COPY meta_adapt.py .
COPY train_rl.py .
COPY latency_test.py .
# For MAD-GAN, add:
# COPY mad_gan_generator.py .
# COPY mad_gan_discriminator.py .
# COPY mad_gan_train.py .

# Optional: Copy entire source tree
# COPY src/ ./src/

# Create non-root user (best practice)
RUN useradd -ms /bin/bash aida
USER aida

# Default command (can be overridden)
CMD ["/bin/bash"]

# To run:
#   docker build -t aida-tll .
#   docker run --gpus all -it aida-tll bash
