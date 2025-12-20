import httpx
import os
from typing import Optional

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


async def embed_text(text: str) -> list[float]:
    """Get embedding for a single text query."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Use /runsync for synchronous response (waits for result)
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"text": text}}
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]["embedding"]
        else:
            raise Exception(f"RunPod error: {result}")


async def embed_images(images_b64: list[str]) -> list[list[float]]:
    """Get embeddings for multiple base64-encoded images."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/runsync",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": {"images": images_b64}}
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "COMPLETED":
            return result["output"]["embeddings"]
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
