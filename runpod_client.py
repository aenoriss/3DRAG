"""
RunPod Client - API client for 3D model embedding service.

Supports:
- Process models by UID (download + render + caption + embed on RunPod)
- Text embedding
- System stats
"""
from __future__ import annotations

import httpx
import asyncio
import os
from typing import List

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Timeout for processing (rendering + captioning can take time)
PROCESS_TIMEOUT = 300.0  # 5 minutes for batch processing
DEFAULT_TIMEOUT = 60.0
POLL_INTERVAL = 2.0  # seconds between status checks


async def _poll_for_completion(job_id: str, timeout: float) -> dict:
    """Poll for job completion when status is IN_PROGRESS."""
    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            response = await client.get(
                f"{BASE_URL}/status/{job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            if status == "COMPLETED":
                return result["output"]
            elif status == "FAILED":
                raise Exception(f"Job failed: {result.get('error', 'Unknown error')}")
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                await asyncio.sleep(POLL_INTERVAL)
            else:
                raise Exception(f"Unknown job status: {status}")


async def process_uids(uids: List[str]) -> dict:
    """
    Process a list of Objaverse UIDs on RunPod.

    Downloads, renders (GPU), captions (Florence-2), and embeds (EmbeddingGemma).

    Args:
        uids: List of Objaverse UIDs to process

    Returns:
        dict with 'results' list containing:
        - uid: Model UID
        - name: Model name from annotations
        - caption: Generated caption
        - embedding: 768-dim embedding vector
        - preview: Base64-encoded preview image
    """
    async with httpx.AsyncClient(timeout=PROCESS_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"uids": uids}}
        )
        response.raise_for_status()
        result = response.json()

        status = result.get("status")
        if status == "COMPLETED":
            return result["output"]
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            # Long-running job, poll for completion
            job_id = result.get("id")
            return await _poll_for_completion(job_id, PROCESS_TIMEOUT)
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_text(text: str) -> dict:
    """
    Get embedding for a text query.

    Args:
        text: Text to embed

    Returns:
        dict with 'embedding' (768-dim list[float])
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"text": text}}
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]
        else:
            raise Exception(f"RunPod error: {result}")


async def get_stats() -> dict:
    """
    Get system stats from RunPod endpoint.

    Returns cumulative request stats and cost estimates.
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
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
