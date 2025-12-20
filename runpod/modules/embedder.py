"""
Text embedding module using Ollama.

Uses EmbeddingGemma for 768-dimensional embeddings.
"""

import time
import requests

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "embeddinggemma"


def wait_for_ollama(timeout: int = 60) -> bool:
    """
    Wait for Ollama server to be ready.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if ready, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ensure_model_loaded():
    """Ensure embedding model is available."""
    resp = requests.get(f"{OLLAMA_URL}/api/tags")
    models = [m["name"] for m in resp.json().get("models", [])]

    if not any(EMBEDDING_MODEL in m for m in models):
        import subprocess
        print(f"Pulling {EMBEDDING_MODEL}...")
        subprocess.run(["ollama", "pull", EMBEDDING_MODEL], check=True)


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        768-dimensional embedding vector
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=30
    )
    resp.raise_for_status()
    result = resp.json()
    embeddings = result.get("embeddings", [])
    return embeddings[0] if embeddings else []


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of 768-dimensional embedding vectors
    """
    if not texts:
        return []

    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120
    )
    resp.raise_for_status()
    result = resp.json()
    return result.get("embeddings", [])
