FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    transformers \
    torch \
    pillow \
    runpod \
    numpy

# Pre-download model (faster cold starts)
RUN python -c "from transformers import AutoModel, AutoProcessor; \
    AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); \
    AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
