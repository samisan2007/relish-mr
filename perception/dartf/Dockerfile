FROM nvcr.io/nvidia/tensorrt:25.06-py3

ENV PYTHONUNBUFFERED=1 HF_HOME=/hf
RUN python -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN python -m pip install --no-cache-dir \
    transformers==5.16.1 huggingface_hub==1.29.0 \
    onnx==1.20.1 onnxscript onnxruntime onnxslim \
    "numpy<2" scipy pillow opencv-python-headless==4.11.0.86 \
    ftfy==6.1.1 regex iopath timm einops safetensors pycocotools psutil
RUN python -m pip install --no-cache-dir decord==0.6.0

WORKDIR /dart
