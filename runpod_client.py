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

print(f"[RunPod Client] Endpoint: {RUNPOD_ENDPOINT_ID}")

# Timeout for processing (all models in one request)
PROCESS_TIMEOUT = 1800.0  # 30 minutes for large batches
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
    print(f"[RunPod] Sending {len(uids)} UIDs to {BASE_URL}/runsync...")
    async with httpx.AsyncClient(timeout=PROCESS_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"uids": uids}}
        )
        response.raise_for_status()
        result = response.json()
        print(f"[RunPod] Response status: {result.get('status')}")

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


async def process_model_bytes(model_bytes: bytes, file_extension: str, name: str) -> dict:
    """
    Process a single model from bytes on RunPod.

    Args:
        model_bytes: Raw bytes of the 3D model file
        file_extension: File format (glb, obj, fbx, etc.)
        name: Display name for the model

    Returns:
        dict with 'name', 'caption', 'embedding', 'preview'
    """
    import base64
    model_b64 = base64.b64encode(model_bytes).decode()

    print(f"[RunPod] Sending model '{name}' ({len(model_bytes)} bytes) for processing...")
    async with httpx.AsyncClient(timeout=PROCESS_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "model_bytes": model_b64,
                    "file_extension": file_extension,
                    "name": name
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        status = result.get("status")
        if status == "COMPLETED":
            return result["output"]
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = result.get("id")
            return await _poll_for_completion(job_id, PROCESS_TIMEOUT)
        else:
            raise Exception(f"RunPod error: {result}")


async def _send_batch_chunk(chunk: List[dict], chunk_id: int) -> dict:
    """Send a single chunk to RunPod and wait for completion."""
    print(f"[RunPod] Chunk {chunk_id}: Sending {len(chunk)} models...")

    async with httpx.AsyncClient(timeout=PROCESS_TIMEOUT) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={
                "input": {
                    "models_batch": chunk
                }
            }
        )
        response.raise_for_status()
        result = response.json()

        status = result.get("status")
        if status == "COMPLETED":
            print(f"[RunPod] Chunk {chunk_id}: Completed immediately")
            return result["output"]
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = result.get("id")
            print(f"[RunPod] Chunk {chunk_id}: Job {job_id} in progress, polling...")
            return await _poll_for_completion(job_id, PROCESS_TIMEOUT)
        else:
            raise Exception(f"RunPod error: {result}")


async def process_models_batch(models: List[dict], max_per_worker: int = 250) -> dict:
    """
    Process multiple models from bytes on RunPod using multiple workers.

    Intelligently splits the batch across available workers for parallel processing.

    Args:
        models: List of dicts with keys:
            - model_id: Unique identifier
            - name: Display name
            - bytes_b64: Base64-encoded model bytes
            - extension: File format (glb, obj, etc.)
        max_per_worker: Maximum models per worker (default 250)

    Returns:
        dict with 'results' list containing:
        - model_id: Original model ID
        - name: Model name
        - caption: Generated caption
        - embedding: 768-dim embedding vector
        - preview: Base64-encoded preview image
        - error: Error message if processing failed (optional)
    """
    import math
    import time

    total_models = len(models)
    print(f"[RunPod] Processing batch of {total_models} models...")

    # Check worker availability
    try:
        health = await health_check()
        workers_info = health.get("workers", {})
        idle_workers = workers_info.get("idle", 1)
        ready_workers = workers_info.get("ready", 1)
        running_workers = workers_info.get("running", 0)

        print(f"[RunPod] Workers - idle: {idle_workers}, ready: {ready_workers}, running: {running_workers}")

        # Use idle workers, but at least 1
        available_workers = max(1, idle_workers)
    except Exception as e:
        print(f"[RunPod] Health check failed: {e}, assuming 1 worker")
        available_workers = 1

    # Calculate optimal split
    # Workers needed = ceil(total / max_per_worker)
    workers_needed = math.ceil(total_models / max_per_worker)

    # Use available workers, but not more than needed
    num_workers = min(workers_needed, available_workers)

    # Calculate chunk size
    chunk_size = math.ceil(total_models / num_workers)
    chunk_size = min(chunk_size, max_per_worker)  # Safety cap

    print(f"[RunPod] Strategy: {total_models} models -> {num_workers} workers, ~{chunk_size} per worker")

    # Split into chunks
    chunks = []
    for i in range(0, total_models, chunk_size):
        chunk = models[i:i + chunk_size]
        chunks.append(chunk)

    print(f"[RunPod] Split into {len(chunks)} chunks: {[len(c) for c in chunks]}")

    # Send all chunks concurrently
    start_time = time.time()

    tasks = [
        _send_batch_chunk(chunk, i + 1)
        for i, chunk in enumerate(chunks)
    ]

    # Wait for all chunks to complete
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Merge results
    all_results = []
    total_processed = 0
    total_failed = 0

    for i, result in enumerate(chunk_results):
        if isinstance(result, Exception):
            print(f"[RunPod] Chunk {i + 1} failed: {result}")
            total_failed += len(chunks[i])
            # Add error entries for failed chunk
            for model in chunks[i]:
                all_results.append({
                    "model_id": model.get("model_id"),
                    "name": model.get("name"),
                    "error": str(result)
                })
        else:
            chunk_processed = result.get("processed", 0)
            total_processed += chunk_processed
            all_results.extend(result.get("results", []))
            print(f"[RunPod] Chunk {i + 1} completed: {chunk_processed} models")

    print(f"[RunPod] All chunks complete: {total_processed}/{total_models} in {elapsed:.1f}s")

    return {
        "results": all_results,
        "processed": total_processed,
        "requested": total_models,
        "time_sec": round(elapsed, 3),
        "time_per_model": round(elapsed / total_processed, 3) if total_processed else 0,
        "workers_used": len(chunks)
    }


async def health_check() -> dict:
    """Check RunPod endpoint health."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{BASE_URL}/health",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
        )
        return response.json()
