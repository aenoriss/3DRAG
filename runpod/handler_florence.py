"""
RunPod Serverless Handler - Florence-2 + EmbeddingGemma

Fast captioning with Florence-2 (0.77B) + text embedding via Ollama.

Pipeline:
1. Receive stitched grid image
2. Florence-2 -> natural caption (~0.3s)
3. EmbeddingGemma -> 768-dim embedding
"""

import runpod
import subprocess
import time
import base64
import requests
import torch
from PIL import Image
import io

# Florence-2 model (loaded once at startup)
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None

# Ollama for embeddings
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"

# GPU cost per second (RTX 4090 = $0.00019/sec)
GPU_COST_PER_SEC = float(__import__('os').getenv("GPU_COST_PER_SEC", "0.00019"))

# Cumulative stats
STATS = {
    "total_requests": 0,
    "total_embeddings": 0,
    "total_text_queries": 0,
    "total_time_sec": 0.0,
    "started_at": time.time()
}


def load_florence():
    """Load Florence-2 model."""
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-base"  # 0.23B, fastest
    # model_id = "microsoft/Florence-2-large"  # 0.77B, better quality

    print(f"Loading Florence-2 from {model_id}...")

    FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to("cuda")

    print("Florence-2 loaded!")


def caption_image(image: Image.Image) -> str:
    """Generate caption using Florence-2."""
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    task = "<MORE_DETAILED_CAPTION>"

    inputs = FLORENCE_PROCESSOR(
        text=task,
        images=image,
        return_tensors="pt"
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = FLORENCE_MODEL.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            do_sample=False
        )

    caption = FLORENCE_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)[0]

    # Remove task prefix if present
    if caption.startswith(task):
        caption = caption[len(task):].strip()

    return caption


def stitch_views_grid(images_b64: list, grid_size: int = 2) -> Image.Image:
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

    return grid


def embed_text(text: str) -> list:
    """Generate embedding using EmbeddingGemma via Ollama."""
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
# STARTUP
# ============================================================================

print("Starting Ollama server...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if not wait_for_ollama(60):
    raise RuntimeError("Ollama failed to start")

print("Ollama is running. Checking embedding model...")

# Check if embedding model is available
resp = requests.get(f"{OLLAMA_URL}/api/tags")
models = [m["name"] for m in resp.json().get("models", [])]

if not any(EMBEDDING_MODEL in m for m in models):
    print(f"Pulling {EMBEDDING_MODEL}...")
    subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)

print("Embedding model ready!")

# Load Florence-2
load_florence()

print("All models ready!")


# ============================================================================
# HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"images": ["<base64>", ...]}  -> stitch + caption + embed
    - {"image": "<base64>"}          -> single image, caption + embed
    - {"text": "query"}              -> embed text for search
    - {"stats": true}                -> return system stats
    - {"reset": true}                -> reset stats
    """
    try:
        input_data = event.get("input", {})

        # Reset stats
        if input_data.get("reset"):
            STATS["total_requests"] = 0
            STATS["total_embeddings"] = 0
            STATS["total_text_queries"] = 0
            STATS["total_time_sec"] = 0.0
            STATS["started_at"] = time.time()
            return {"status": "reset", "message": "Stats reset successfully"}

        # Stats endpoint
        if input_data.get("stats"):
            uptime = time.time() - STATS["started_at"]
            avg_time = STATS["total_time_sec"] / STATS["total_requests"] if STATS["total_requests"] > 0 else 0
            estimated_cost = STATS["total_time_sec"] * GPU_COST_PER_SEC
            cost_per_model = estimated_cost / STATS["total_embeddings"] if STATS["total_embeddings"] > 0 else 0

            return {
                "status": "ready",
                "vision_model": "Florence-2-base",
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dim": 768,
                "cumulative": {
                    "total_requests": STATS["total_requests"],
                    "total_embeddings": STATS["total_embeddings"],
                    "total_text_queries": STATS["total_text_queries"],
                    "total_time_sec": round(STATS["total_time_sec"], 3),
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

            # Caption with Florence-2
            caption = caption_image(grid_image)

            # Embed caption
            embedding = embed_text(caption)

            elapsed = time.time() - start

            # Update cumulative stats
            STATS["total_requests"] += 1
            STATS["total_embeddings"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "text": caption,
                "dimension": len(embedding),
                "stats": {
                    "time_sec": round(elapsed, 3)
                }
            }

        # Single image
        if "image" in input_data:
            start = time.time()

            img_bytes = base64.b64decode(input_data["image"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            caption = caption_image(image)
            embedding = embed_text(caption)

            elapsed = time.time() - start

            # Update cumulative stats
            STATS["total_requests"] += 1
            STATS["total_embeddings"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "text": caption,
                "dimension": len(embedding),
                "stats": {
                    "time_sec": round(elapsed, 3)
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
