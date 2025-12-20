"""
Ollama Client - Vision + Embedding for 3D model indexing.

Uses:
- Gemma 3 27B for vision (describe 3D object from rendered views)
- EmbeddingGemma for text embeddings (768-dim)

Supports two modes:
1. Local Ollama: Set OLLAMA_LOCAL=true (for dev/testing)
2. RunPod Serverless: Set RUNPOD_OLLAMA_ENDPOINT_ID (for production)

Same architecture as SigLIP2 pipeline - render locally, embed on RunPod.
"""
from __future__ import annotations

import base64
import json
import os
import httpx
from pathlib import Path
from typing import Optional
from PIL import Image
import io

# Mode configuration
USE_LOCAL_OLLAMA = os.getenv("OLLAMA_LOCAL", "false").lower() == "true"

# Local Ollama endpoint
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# RunPod serverless endpoint (for Ollama handler)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_OLLAMA_ENDPOINT_ID = os.getenv("RUNPOD_OLLAMA_ENDPOINT_ID")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_OLLAMA_ENDPOINT_ID}" if RUNPOD_OLLAMA_ENDPOINT_ID else None

# Timeouts
RUNPOD_TIMEOUT = 180.0  # 3 min for cold start
LOCAL_TIMEOUT = 120.0

# Models
VISION_MODEL = "minicpm-v"  # 8B, GPT-4o level, 75% fewer vision tokens
EMBEDDING_MODEL = "embeddinggemma"

# JSON Schema for structured output (strict enforcement)
DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "Main object type (e.g., furniture, vehicle, weapon, tool)"
        },
        "subcategory": {
            "type": "string",
            "description": "Specific type (e.g., chair, car, sword, hammer)"
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 key visual features (shape, material, color, style)"
        },
        "purpose": {
            "type": "string",
            "description": "What it's used for (1-5 words)"
        },
        "similar_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 related/similar objects"
        }
    },
    "required": ["category", "subcategory", "attributes", "purpose", "similar_to"]
}

# Prompt template for describing 3D objects (optimized for retrieval)
# Explicitly describes grid layout for the model
VISION_PROMPT = """This image is a 2x2 grid of 4 rendered views of a single 3D model:
- Top-left: FRONT view
- Top-right: SIDE view
- Bottom-left: BACK view
- Bottom-right: TOP view

Analyze all 4 views together to identify this 3D object. Describe it for search indexing. Be concise. Use common search terms."""


def stitch_views_grid(images_b64: list[str], grid_size: int = 2) -> str:
    """
    Stitch multiple view images into a single grid image.

    Args:
        images_b64: List of base64-encoded images
        grid_size: Grid dimension (2 = 2x2 grid of 4 images)

    Returns:
        Base64-encoded grid image
    """
    from PIL import Image

    # Decode images
    images = []
    for img_b64 in images_b64[:grid_size * grid_size]:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    if not images:
        raise ValueError("No images to stitch")

    # Get dimensions (assume all same size)
    w, h = images[0].size

    # Create grid canvas
    grid_w = w * grid_size
    grid_h = h * grid_size
    grid = Image.new("RGB", (grid_w, grid_h), (128, 128, 128))

    # Paste images into grid
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        grid.paste(img, (col * w, row * h))

    # Encode back to base64
    buffer = io.BytesIO()
    grid.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


async def _runpod_request(payload: dict, timeout: float = RUNPOD_TIMEOUT) -> dict:
    """Make a request to RunPod serverless Ollama endpoint."""
    if not RUNPOD_BASE_URL or not RUNPOD_API_KEY:
        return {"status": "error", "error": "RUNPOD_OLLAMA_ENDPOINT_ID or RUNPOD_API_KEY not set"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{RUNPOD_BASE_URL}/runsync",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                json={"input": payload}
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") == "COMPLETED":
                output = result.get("output", {})
                if "error" in output:
                    return {"status": "error", "error": output["error"]}
                return {"status": "ok", **output}
            else:
                return {"status": "error", "error": f"RunPod status: {result.get('status')}"}

        except Exception as e:
            return {"status": "error", "error": str(e)}


async def reset_stats() -> dict:
    """Reset cumulative stats on RunPod endpoint."""
    if USE_LOCAL_OLLAMA:
        return {"status": "ok", "message": "Local mode - no stats to reset"}
    else:
        return await _runpod_request({"reset": True}, timeout=30.0)


async def check_ollama() -> dict:
    """Check if Ollama is running and models are available."""
    if USE_LOCAL_OLLAMA:
        # Check local Ollama
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]

                has_vision = any(VISION_MODEL in name for name in model_names)
                has_embedding = any(EMBEDDING_MODEL in name for name in model_names)

                return {
                    "status": "ok",
                    "mode": "local",
                    "endpoint": OLLAMA_BASE_URL,
                    "models": model_names,
                    "has_vision": has_vision,
                    "has_embedding": has_embedding,
                    "vision_model": VISION_MODEL,
                    "embedding_model": EMBEDDING_MODEL
                }
            except Exception as e:
                return {
                    "status": "error",
                    "mode": "local",
                    "error": str(e)
                }
    else:
        # Check RunPod serverless endpoint
        result = await _runpod_request({"stats": True}, timeout=30.0)
        if result["status"] == "ok":
            return {
                "status": "ok",
                "mode": "runpod",
                "endpoint": RUNPOD_BASE_URL,
                **result
            }
        else:
            return {
                "status": "error",
                "mode": "runpod",
                "error": result.get("error", "Unknown error")
            }


async def describe_object(images_b64: list[str], timeout: float = 120.0) -> dict:
    """
    Use Gemma 3 27B vision to describe a 3D object from rendered views.

    Args:
        images_b64: List of base64-encoded images (rendered views)
        timeout: Request timeout in seconds

    Returns:
        Dict with parsed JSON description or error
    """
    # Stitch 4 views into a single 2x2 grid image (optimizes token usage)
    # Use views: front (0), side (2), back (4), top (8)
    selected_indices = [0, 2, 4, 8]
    selected_images = [images_b64[i] for i in selected_indices if i < len(images_b64)]

    # Pad with available images if we don't have enough
    while len(selected_images) < 4 and len(selected_images) < len(images_b64):
        selected_images.append(images_b64[len(selected_images)])

    try:
        grid_image = stitch_views_grid(selected_images)
    except Exception as e:
        return {"status": "error", "error": f"Failed to stitch images: {e}"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": VISION_MODEL,
                    "prompt": VISION_PROMPT,
                    "images": [grid_image],  # Single stitched image
                    "stream": False,
                    "format": DESCRIPTION_SCHEMA,  # Strict JSON schema enforcement
                    "options": {
                        "temperature": 0,     # Deterministic output
                        "num_predict": 150    # Limit response length
                    }
                }
            )
            resp.raise_for_status()
            result = resp.json()

            response_text = result.get("response", "")

            # Parse JSON response (should always be valid with schema enforcement)
            try:
                description = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    description = json.loads(json_match.group())
                else:
                    description = {"raw": response_text}

            return {
                "status": "ok",
                "description": description,
                "eval_count": result.get("eval_count", 0),
                "eval_duration_ms": result.get("eval_duration", 0) / 1e6
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


def description_to_text(description: dict) -> str:
    """
    Convert structured description to embedding-friendly text.

    Formats the JSON description as a keyword-rich text string
    optimized for semantic search.
    """
    parts = []

    # Category and subcategory
    if "category" in description:
        parts.append(description["category"])
    if "subcategory" in description:
        parts.append(description["subcategory"])

    # Attributes as comma-separated
    if "attributes" in description and isinstance(description["attributes"], list):
        parts.extend(description["attributes"])

    # Purpose
    if "purpose" in description:
        parts.append(description["purpose"])

    # Similar objects
    if "similar_to" in description and isinstance(description["similar_to"], list):
        parts.extend(description["similar_to"])

    # Fallback to raw if structured parsing failed
    if not parts and "raw" in description:
        return description["raw"][:500]  # Limit length

    # Join with spaces for embedding
    text = " ".join(parts)
    return text if text else "unknown object"


async def embed_text(text: str, timeout: float = 30.0) -> dict:
    """
    Generate embedding for text using EmbeddingGemma.

    Args:
        text: Text to embed
        timeout: Request timeout

    Returns:
        Dict with embedding (768-dim) or error
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text
                }
            )
            resp.raise_for_status()
            result = resp.json()

            # Ollama returns embeddings in "embeddings" array
            embeddings = result.get("embeddings", [])
            if embeddings:
                return {
                    "status": "ok",
                    "embedding": embeddings[0],
                    "dimension": len(embeddings[0])
                }
            else:
                return {
                    "status": "error",
                    "error": "No embedding returned"
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


async def embed_query(query: str, timeout: float = 30.0) -> dict:
    """
    Generate embedding for a search query.

    Uses local Ollama or RunPod serverless based on configuration.
    """
    if USE_LOCAL_OLLAMA:
        # Local Ollama
        return await embed_text(query, timeout)
    else:
        # RunPod serverless
        result = await _runpod_request({"text": query}, timeout=60.0)
        if result["status"] != "ok":
            return result

        return {
            "status": "ok",
            "embedding": result.get("embedding", []),
            "dimension": result.get("dimension", 768)
        }


async def process_3d_model(images_b64: list[str]) -> dict:
    """
    Full pipeline: describe + embed a 3D model from rendered views.

    Uses local Ollama or RunPod serverless based on configuration.

    Args:
        images_b64: List of base64-encoded rendered images

    Returns:
        Dict with description, text, embedding, or error
    """
    if USE_LOCAL_OLLAMA:
        # Local Ollama pipeline
        desc_result = await describe_object(images_b64)
        if desc_result["status"] != "ok":
            return desc_result

        description = desc_result["description"]
        text = description_to_text(description)

        embed_result = await embed_text(text)
        if embed_result["status"] != "ok":
            return embed_result

        return {
            "status": "ok",
            "description": description,
            "text": text,
            "embedding": embed_result["embedding"],
            "dimension": embed_result["dimension"],
            "vision_stats": {
                "eval_count": desc_result.get("eval_count", 0),
                "eval_duration_ms": desc_result.get("eval_duration_ms", 0)
            }
        }
    else:
        # RunPod serverless pipeline
        result = await _runpod_request({"images": images_b64})
        if result["status"] != "ok":
            return result

        return {
            "status": "ok",
            "description": result.get("description", {}),
            "text": result.get("text", ""),
            "embedding": result.get("embedding", []),
            "dimension": result.get("dimension", 768),
            "vision_stats": result.get("stats", {})
        }


# Synchronous wrappers for CLI usage
def check_ollama_sync() -> dict:
    import asyncio
    return asyncio.run(check_ollama())


def process_3d_model_sync(images_b64: list[str]) -> dict:
    import asyncio
    return asyncio.run(process_3d_model(images_b64))


if __name__ == "__main__":
    import asyncio

    # Test Ollama connection
    print("Checking Ollama...")
    result = check_ollama_sync()
    print(json.dumps(result, indent=2))

    if result["status"] == "ok":
        if not result["has_vision"]:
            print(f"\nMissing vision model. Run: ollama pull {VISION_MODEL}")
        if not result["has_embedding"]:
            print(f"\nMissing embedding model. Run: ollama pull {EMBEDDING_MODEL}")
