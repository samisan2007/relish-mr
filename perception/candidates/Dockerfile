FROM relish-sam31:local

# Reuse the working CUDA/PyTorch image; do not install research training extras.
RUN python -m pip install --no-cache-dir hydra-core==1.3.2 omegaconf==2.3.0 timm==1.0.30
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/local/hf
WORKDIR /app
