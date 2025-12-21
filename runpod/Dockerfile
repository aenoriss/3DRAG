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
    PyOpenGL_accelerate \
    objaverse \
    requests \
    einops \
    timm \
    sentence-transformers

# Copy application code first (needed for pre-download with patches)
COPY runpod/modules /app/modules
COPY runpod/handler.py /app/handler.py

# Florence-2 downloads at runtime (flash_attn patch required, handled by captioner module)

# Pre-download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-mpnet-base-v2')"

CMD ["python", "-u", "handler.py"]
