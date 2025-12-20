"""
RunPod Client - API client for 3D model embedding service.

Supports:
- 3D model embedding (render + embed + average)
- Text embedding
- Image embedding
- System stats
"""
from __future__ import annotations

import httpx
import os
import base64
from pathlib import Path
from typing import Optional, List, Union

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Timeout for 3D model processing (render + embed can take a while on cold start)
MODEL_TIMEOUT = 180.0  # 3 minutes for cold start
IMAGE_TIMEOUT = 180.0  # 3 minutes for batch image embedding (cold start)
DEFAULT_TIMEOUT = 60.0


async def embed_model(
    model_path: Union[str, Path],
    include_stats: bool = False
) -> dict:
    """
    Get embedding for a 3D model.

    Sends the model to RunPod, which renders 12 views and returns
    the averaged embedding.

    Args:
        model_path: Path to 3D model file (GLB, OBJ, STL, PLY, etc.)
        include_stats: Include timing and cost stats in response

    Returns:
        dict with 'embedding' (list[float]) and optionally 'stats'
    """
    model_path = Path(model_path)

    # Read and encode model
    model_bytes = model_path.read_bytes()
    model_b64 = base64.b64encode(model_bytes).decode()

    # Get format from extension
    file_format = model_path.suffix.lstrip('.').lower()
    if file_format == 'gltf':
        file_format = 'glb'  # trimesh handles both

    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "model": model_b64,
                    "format": file_format,
                    "include_stats": include_stats
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_model_bytes(
    model_bytes: bytes,
    file_format: str = "glb",
    include_stats: bool = False
) -> dict:
    """
    Get embedding for a 3D model from bytes.

    Args:
        model_bytes: Raw bytes of the 3D model
        file_format: File format (glb, obj, stl, ply, etc.)
        include_stats: Include timing and cost stats in response

    Returns:
        dict with 'embedding' (list[float]) and optionally 'stats'
    """
    model_b64 = base64.b64encode(model_bytes).decode()

    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "model": model_b64,
                    "format": file_format,
                    "include_stats": include_stats
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_text(
    text: str,
    include_stats: bool = False
) -> dict:
    """
    Get embedding for a text query.

    Args:
        text: Text to embed
        include_stats: Include timing and cost stats

    Returns:
        dict with 'embedding' (list[float]) and optionally 'stats'
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "text": text,
                    "include_stats": include_stats
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_texts(
    texts: List[str],
    include_stats: bool = False
) -> dict:
    """
    Get embeddings for multiple text queries.

    Args:
        texts: List of texts to embed
        include_stats: Include timing and cost stats

    Returns:
        dict with 'embeddings' (list[list[float]]) and optionally 'stats'
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "texts": texts,
                    "include_stats": include_stats
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_images(
    images_b64: List[str],
    include_stats: bool = False
) -> dict:
    """
    Get embeddings for base64-encoded images.

    Args:
        images_b64: List of base64-encoded images
        include_stats: Include timing and cost stats

    Returns:
        dict with 'embeddings' (list[list[float]]) and optionally 'stats'
    """
    print(f"  Calling RunPod embed_images with {len(images_b64)} images...")
    async with httpx.AsyncClient(timeout=IMAGE_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "images": images_b64,
                    "include_stats": include_stats
                }
            }
        )
        print(f"  RunPod response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print(f"  RunPod job status: {result.get('status')}")

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def get_stats() -> dict:
    """
    Get system stats from RunPod endpoint.

    Returns GPU info, model info, pricing, and performance estimates.
    """
    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"stats": True}}
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def health_check() -> dict:
    """Check RunPod endpoint health."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BASE_URL}/health",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        return response.json()
