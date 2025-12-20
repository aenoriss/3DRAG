FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies for EGL/OpenGL rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    libglfw3 \
    libglfw3-dev \
    libosmesa6 \
    libosmesa6-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    torch \
    pillow \
    runpod \
    numpy \
    trimesh \
    pyrender \
    PyOpenGL \
    PyOpenGL_accelerate

# Pre-download model (faster cold starts)
RUN python -c "from transformers import AutoModel, AutoProcessor; \
    AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); \
    AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')"

COPY runpod/handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
