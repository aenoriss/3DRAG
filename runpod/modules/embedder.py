"""
Text embedding module using sentence-transformers.

Uses all-mpnet-base-v2 for 768-dimensional embeddings.
GPU-accelerated on RunPod.
"""

from sentence_transformers import SentenceTransformer

# Model: 768-dim, high quality, same dim as EmbeddingGemma
MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768

# Global model instance (loaded once)
_model = None


def load_model():
    """Load the embedding model (GPU if available)."""
    global _model
    if _model is None:
        print(f"Loading {MODEL_NAME} on GPU...")
        _model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("Embedding model ready!")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        768-dimensional embedding vector
    """
    model = load_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple texts (GPU-accelerated).

    Args:
        texts: List of texts to embed

    Returns:
        List of 768-dimensional embedding vectors
    """
    if not texts:
        return []

    model = load_model()
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True)
    return embeddings.tolist()


# Legacy functions for compatibility (no longer needed)
def wait_for_ollama(timeout: int = 60) -> bool:
    """Legacy stub - no longer uses Ollama."""
    return True


def ensure_model_loaded():
    """Legacy stub - loads sentence-transformers model."""
    load_model()
