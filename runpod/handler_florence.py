"""
RunPod Serverless Handler - Florence-2 + EmbeddingGemma

Fast captioning with Florence-2 (0.23B) + text embedding via Ollama.
Supports batch processing for high throughput.

Pipeline:
1. Receive batch of images
2. Florence-2 batch inference -> captions
3. EmbeddingGemma batch -> 768-dim embeddings
"""

import runpod
import subprocess
import time
import base64
import requests
import torch
from PIL import Image
import io
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Florence-2 model (loaded once at startup)
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None

# Thread pool for parallel embedding requests
EMBED_POOL = ThreadPoolExecutor(max_workers=8)


def _fixed_get_imports(filename: str) -> list:
    """Patch to remove flash_attn from imports (not actually needed)."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

# Ollama for embeddings
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"

# GPU cost per second (RTX 4090 = $0.00019/sec)
GPU_COST_PER_SEC = float(os.getenv("GPU_COST_PER_SEC", "0.00019"))

# Cumulative stats
STATS = {
    "total_requests": 0,
    "total_embeddings": 0,
    "total_text_queries": 0,
    "total_time_sec": 0.0,
    "started_at": time.time()
}


def load_florence():
    """Load Florence-2 model with flash_attn import patch."""
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    from transformers import AutoProcessor, AutoModelForCausalLM

    model_id = "microsoft/Florence-2-base"  # 0.23B, fastest

    print(f"Loading Florence-2 from {model_id}...")

    # Patch to remove flash_attn requirement
    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        ).to("cuda")

    print("Florence-2 loaded!")


def clean_caption(caption: str) -> str:
    """Clean up caption for better embedding quality."""
    import re

    # Remove task tokens
    for token in ["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<CAPTION>"]:
        caption = caption.replace(token, "")

    # Remove common filler phrases
    filler_patterns = [
        r"^the image (shows?|contains?|depicts?|features?|displays?)\s*",
        r"^this image (shows?|contains?|depicts?|features?|displays?)\s*",
        r"^in this image,?\s*",
        r"^i (can )?see\s*",
        r"^there (is|are)\s*",
        r"^we (can )?see\s*",
        r"^the picture (shows?|contains?)\s*",
    ]

    for pattern in filler_patterns:
        caption = re.sub(pattern, "", caption, flags=re.IGNORECASE)

    caption = " ".join(caption.split())
    if caption:
        caption = caption[0].upper() + caption[1:]

    return caption


def caption_images_batch(images: list[Image.Image]) -> list[str]:
    """
    Batch caption multiple images with Florence-2.

    Processes all images in a single forward pass for maximum GPU efficiency.
    """
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if not images:
        return []

    task = "<MORE_DETAILED_CAPTION>"

    # Process all images in batch
    inputs = FLORENCE_PROCESSOR(
        text=[task] * len(images),
        images=images,
        return_tensors="pt",
        padding=True
    ).to("cuda", torch.float16)

    with torch.no_grad():
        outputs = FLORENCE_MODEL.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=1,
            do_sample=False
        )

    # Decode all outputs
    captions = FLORENCE_PROCESSOR.batch_decode(outputs, skip_special_tokens=True)

    # Clean all captions
    return [clean_caption(c) for c in captions]


def embed_text(text: str) -> list:
    """Generate embedding using EmbeddingGemma via Ollama."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30
    )
    resp.raise_for_status()
    result = resp.json()
    embeddings = result.get("embeddings", [])
    return embeddings[0] if embeddings else []


def embed_texts_batch(texts: list[str]) -> list[list]:
    """
    Batch embed multiple texts.

    Ollama supports batch embedding via input list.
    """
    if not texts:
        return []

    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    return result.get("embeddings", [])


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

resp = requests.get(f"{OLLAMA_URL}/api/tags")
models = [m["name"] for m in resp.json().get("models", [])]

if not any(EMBEDDING_MODEL in m for m in models):
    print(f"Pulling {EMBEDDING_MODEL}...")
    subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)

print("Embedding model ready!")

load_florence()

print("All models ready!")


# ============================================================================
# HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"batch": [{"image": "<b64>"}, ...]}  -> batch process multiple models
    - {"image": "<base64>"}                 -> single image
    - {"text": "query"}                     -> embed text for search
    - {"stats": true}                       -> return system stats
    - {"reset": true}                       -> reset stats
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

        # ====================================================================
        # BATCH PROCESSING - Multiple models in one request
        # ====================================================================
        if "batch" in input_data:
            start = time.time()
            batch = input_data["batch"]

            # Decode all images
            images = []
            for item in batch:
                img_b64 = item.get("image")
                if img_b64:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)

            if not images:
                return {"error": "No valid images in batch"}

            # Batch caption with Florence-2
            captions = caption_images_batch(images)

            # Batch embed with Ollama
            embeddings = embed_texts_batch(captions)

            elapsed = time.time() - start

            # Update stats
            STATS["total_requests"] += 1
            STATS["total_embeddings"] += len(images)
            STATS["total_time_sec"] += elapsed

            # Return results
            results = []
            for i, (caption, embedding) in enumerate(zip(captions, embeddings)):
                results.append({
                    "index": i,
                    "embedding": embedding,
                    "text": caption,
                    "dimension": len(embedding) if embedding else 0
                })

            return {
                "results": results,
                "batch_size": len(images),
                "stats": {
                    "time_sec": round(elapsed, 3),
                    "time_per_model": round(elapsed / len(images), 3) if images else 0
                }
            }

        # Single image (legacy support)
        if "image" in input_data:
            start = time.time()

            img_bytes = base64.b64decode(input_data["image"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            captions = caption_images_batch([image])
            caption = captions[0] if captions else ""
            embedding = embed_text(caption)

            elapsed = time.time() - start

            STATS["total_requests"] += 1
            STATS["total_embeddings"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "text": caption,
                "dimension": len(embedding),
                "stats": {"time_sec": round(elapsed, 3)}
            }

        # Text query embedding
        if "text" in input_data:
            start = time.time()
            embedding = embed_text(input_data["text"])
            elapsed = time.time() - start

            STATS["total_requests"] += 1
            STATS["total_text_queries"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "dimension": len(embedding),
                "stats": {"time_sec": round(elapsed, 3)}
            }

        return {"error": "No valid input. Use 'batch', 'image', 'text', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
