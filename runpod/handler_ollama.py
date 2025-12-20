"""
RunPod Serverless Handler - Ollama (Gemma 3 27B + EmbeddingGemma)

Receives rendered 3D model views, returns semantic embedding.

Pipeline:
1. Receive stitched grid image (or stitch on server)
2. Gemma 3 27B vision -> structured JSON description
3. EmbeddingGemma -> 768-dim embedding
"""

import runpod
import subprocess
import time
import base64
import json
import requests
from PIL import Image
import io

# Ollama API (running locally on the pod)
OLLAMA_URL = "http://localhost:11434"

# GPU cost per second (RTX 4090 = $0.00019/sec)
GPU_COST_PER_SEC = float(__import__('os').getenv("GPU_COST_PER_SEC", "0.00019"))

# Cumulative stats
STATS = {
    "total_requests": 0,
    "total_embeddings": 0,
    "total_text_queries": 0,
    "total_time_sec": 0.0,
    "total_vision_tokens": 0,
    "started_at": time.time()
}

# Models
VISION_MODEL = "minicpm-v"  # 8B, GPT-4o level, 75% fewer vision tokens
EMBEDDING_MODEL = "embeddinggemma"

# JSON Schema for structured output
DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "subcategory": {"type": "string"},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "purpose": {"type": "string"},
        "similar_to": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["category", "subcategory", "attributes", "purpose", "similar_to"]
}

# Vision prompt
VISION_PROMPT = """This image is a 2x2 grid of 4 rendered views of a single 3D model:
- Top-left: FRONT view
- Top-right: SIDE view
- Bottom-left: BACK view
- Bottom-right: TOP view

Analyze all 4 views together to identify this 3D object. Describe it for search indexing. Be concise. Use common search terms."""


def stitch_views_grid(images_b64: list, grid_size: int = 2) -> str:
    """Stitch multiple view images into a single grid image."""
    images = []
    for img_b64 in images_b64[:grid_size * grid_size]:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    if not images:
        raise ValueError("No images to stitch")

    w, h = images[0].size
    grid_w = w * grid_size
    grid_h = h * grid_size
    grid = Image.new("RGB", (grid_w, grid_h), (128, 128, 128))

    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        grid.paste(img, (col * w, row * h))

    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def describe_image(image_b64: str) -> dict:
    """Use Gemma 3 27B to describe the 3D object."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": VISION_MODEL,
            "prompt": VISION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": DESCRIPTION_SCHEMA,
            "options": {
                "temperature": 0,
                "num_predict": 150
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    result = resp.json()

    response_text = result.get("response", "")
    try:
        description = json.loads(response_text)
    except json.JSONDecodeError:
        description = {"raw": response_text}

    return {
        "description": description,
        "eval_count": result.get("eval_count", 0),
        "eval_duration_ms": result.get("eval_duration", 0) / 1e6
    }


def description_to_text(description: dict) -> str:
    """Convert structured description to embedding-friendly text."""
    parts = []
    if "category" in description:
        parts.append(description["category"])
    if "subcategory" in description:
        parts.append(description["subcategory"])
    if "attributes" in description and isinstance(description["attributes"], list):
        parts.extend(description["attributes"])
    if "purpose" in description:
        parts.append(description["purpose"])
    if "similar_to" in description and isinstance(description["similar_to"], list):
        parts.extend(description["similar_to"])
    if not parts and "raw" in description:
        return description["raw"][:500]
    return " ".join(parts) if parts else "unknown object"


def embed_text(text: str) -> list:
    """Generate embedding using EmbeddingGemma."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=30
    )
    resp.raise_for_status()
    result = resp.json()
    embeddings = result.get("embeddings", [])
    return embeddings[0] if embeddings else []


def wait_for_ollama(timeout: int = 60):
    """Wait for Ollama to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


# ============================================================================
# STARTUP - Start Ollama and pull models
# ============================================================================

print("Starting Ollama server...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if not wait_for_ollama(60):
    raise RuntimeError("Ollama failed to start")

print("Ollama is running. Checking models...")

# Check if models are available
resp = requests.get(f"{OLLAMA_URL}/api/tags")
models = [m["name"] for m in resp.json().get("models", [])]

if not any(VISION_MODEL in m for m in models):
    print(f"Pulling {VISION_MODEL}...")
    subprocess.run(["ollama", "pull", VISION_MODEL], check=True)

if not any(EMBEDDING_MODEL in m for m in models):
    print(f"Pulling {EMBEDDING_MODEL}...")
    subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)

print("Models ready!")


# ============================================================================
# HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"images": ["<base64>", ...]}  -> stitch + describe + embed
    - {"image": "<base64>"}          -> single grid image, describe + embed
    - {"text": "query"}              -> embed text for search
    - {"stats": true}                -> return system stats
    """
    try:
        input_data = event.get("input", {})

        # Reset stats
        if input_data.get("reset"):
            STATS["total_requests"] = 0
            STATS["total_embeddings"] = 0
            STATS["total_text_queries"] = 0
            STATS["total_time_sec"] = 0.0
            STATS["total_vision_tokens"] = 0
            STATS["started_at"] = time.time()
            return {"status": "reset", "message": "Stats reset successfully"}

        # Stats endpoint
        if input_data.get("stats"):
            resp = requests.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            uptime = time.time() - STATS["started_at"]
            avg_time = STATS["total_time_sec"] / STATS["total_requests"] if STATS["total_requests"] > 0 else 0
            estimated_cost = STATS["total_time_sec"] * GPU_COST_PER_SEC
            cost_per_model = estimated_cost / STATS["total_embeddings"] if STATS["total_embeddings"] > 0 else 0

            return {
                "status": "ready",
                "models": models,
                "vision_model": VISION_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": 768,
                "cumulative": {
                    "total_requests": STATS["total_requests"],
                    "total_embeddings": STATS["total_embeddings"],
                    "total_text_queries": STATS["total_text_queries"],
                    "total_time_sec": round(STATS["total_time_sec"], 3),
                    "total_vision_tokens": STATS["total_vision_tokens"],
                    "avg_time_sec": round(avg_time, 3),
                    "uptime_sec": round(uptime, 1),
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "cost_per_model_usd": round(cost_per_model, 6),
                    "gpu_cost_per_sec": GPU_COST_PER_SEC
                }
            }

        # Multiple images -> stitch + process
        if "images" in input_data:
            start = time.time()

            # Select 4 views and stitch
            images = input_data["images"]
            indices = [0, 2, 4, 8]
            selected = [images[i] for i in indices if i < len(images)]
            while len(selected) < 4 and len(selected) < len(images):
                selected.append(images[len(selected)])

            grid_image = stitch_views_grid(selected)

            # Describe
            desc_result = describe_image(grid_image)
            description = desc_result["description"]

            # Convert to text
            text = description_to_text(description)

            # Embed
            embedding = embed_text(text)

            elapsed = time.time() - start
            vision_tokens = desc_result.get("eval_count", 0)

            # Update cumulative stats
            STATS["total_requests"] += 1
            STATS["total_embeddings"] += 1
            STATS["total_time_sec"] += elapsed
            STATS["total_vision_tokens"] += vision_tokens

            return {
                "embedding": embedding,
                "description": description,
                "text": text,
                "dimension": len(embedding),
                "stats": {
                    "time_sec": round(elapsed, 3),
                    "vision_tokens": vision_tokens
                }
            }

        # Single grid image
        if "image" in input_data:
            start = time.time()

            desc_result = describe_image(input_data["image"])
            description = desc_result["description"]
            text = description_to_text(description)
            embedding = embed_text(text)

            elapsed = time.time() - start
            vision_tokens = desc_result.get("eval_count", 0)

            # Update cumulative stats
            STATS["total_requests"] += 1
            STATS["total_embeddings"] += 1
            STATS["total_time_sec"] += elapsed
            STATS["total_vision_tokens"] += vision_tokens

            return {
                "embedding": embedding,
                "description": description,
                "text": text,
                "dimension": len(embedding),
                "stats": {
                    "time_sec": round(elapsed, 3),
                    "vision_tokens": vision_tokens
                }
            }

        # Text query embedding
        if "text" in input_data:
            start = time.time()
            embedding = embed_text(input_data["text"])
            elapsed = time.time() - start

            # Update cumulative stats
            STATS["total_requests"] += 1
            STATS["total_text_queries"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "dimension": len(embedding),
                "stats": {"time_sec": round(elapsed, 3)}
            }

        return {"error": "No valid input. Use 'images', 'image', 'text', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
