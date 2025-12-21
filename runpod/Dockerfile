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

# Pre-download Florence-2 model weights (with flash_attn patch, CPU only during build)
RUN python -c "\
from unittest.mock import patch; \
from transformers.dynamic_module_utils import get_imports; \
def fix(f): i=get_imports(f); 'flash_attn' in i and i.remove('flash_attn'); return i; \
with patch('transformers.dynamic_module_utils.get_imports', lambda f: fix(f) if 'florence2' in f else get_imports(f)): \
    from transformers import AutoProcessor, AutoModelForCausalLM; \
    AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)"

# Pre-download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-mpnet-base-v2')"

CMD ["python", "-u", "handler.py"]
