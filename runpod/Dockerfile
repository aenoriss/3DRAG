FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies for EGL/OpenGL headless rendering + X11 libs for pyglet
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    libglfw3 \
    libosmesa6 \
    freeglut3-dev \
    xvfb \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "transformers>=4.40.0,<4.50.0" \
    torch \
    pillow \
    runpod \
    "numpy<2.0" \
    trimesh \
    pyrender \
    PyOpenGL \
    PyOpenGL_accelerate \
    objaverse \
    requests \
    einops \
    timm \
    sentence-transformers \
    huggingface_hub

# Copy application code
COPY runpod/modules /app/modules
COPY runpod/handler.py /app/handler.py
COPY runpod/download_models.py /app/download_models.py

# Pre-download all models (Florence-2 + sentence-transformers)
RUN python download_models.py

CMD ["python", "-u", "handler.py"]
