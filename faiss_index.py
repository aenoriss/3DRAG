"""
FAISS Index Manager - Efficient vector indexing and search for 3D models.

Uses IndexHNSWFlat for fast approximate nearest neighbor search.
HNSW is ideal for dynamic datasets with frequent additions.
"""

import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class ModelMetadata:
    """Metadata for an indexed 3D model."""
    id: str
    name: str
    category: Optional[str] = None
    file_path: Optional[str] = None
    indexed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        return cls(**data)


class FAISSIndex:
    """
    Thread-safe FAISS index for 3D model embeddings.

    Uses IndexHNSWFlat for fast approximate nearest neighbor search.
    HNSW (Hierarchical Navigable Small World) is ideal for:
    - Dynamic datasets with frequent additions
    - Fast search with high recall
    - No training required
    """

    EMBEDDING_DIM = 1152  # SigLIP2 so400m embedding dimension
    HNSW_M = 32  # Number of neighbors per node (higher = better recall, more memory)
    HNSW_EF_CONSTRUCTION = 40  # Construction time accuracy (higher = better quality)
    HNSW_EF_SEARCH = 64  # Search time accuracy (higher = better recall, slower)

    def __init__(
        self,
        index_path: str = "models.index",
        metadata_path: str = "metadata.json"
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.metadata: list[ModelMetadata] = []
        self._lock = threading.Lock()

        # Load or create index
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            # Load existing index
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.metadata = [ModelMetadata.from_dict(m) for m in data]
            print(f"Loaded FAISS HNSW index with {self.index.ntotal} vectors")
        else:
            # Create new HNSW index
            self.index = faiss.IndexHNSWFlat(self.EMBEDDING_DIM, self.HNSW_M)
            self.index.hnsw.efConstruction = self.HNSW_EF_CONSTRUCTION
            self.index.hnsw.efSearch = self.HNSW_EF_SEARCH
            self.metadata = []
            print(f"Created new FAISS HNSW index (dim={self.EMBEDDING_DIM}, M={self.HNSW_M})")

    def _save(self):
        """Save index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metadata], f, indent=2)

    def add(
        self,
        embedding: list[float] | np.ndarray,
        model_id: str,
        name: str,
        category: Optional[str] = None,
        file_path: Optional[str] = None,
        save: bool = True
    ) -> int:
        """
        Add a single embedding to the index.

        Args:
            embedding: 1152-dim normalized embedding from SigLIP2
            model_id: Unique identifier for the model
            name: Display name
            category: Optional category
            file_path: Optional path to the original model file
            save: Whether to persist to disk immediately

        Returns:
            Index position of the added embedding
        """
        with self._lock:
            # Convert to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)

            # Ensure correct shape
            embedding = embedding.reshape(1, -1)

            # Verify dimension
            if embedding.shape[1] != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.EMBEDDING_DIM}, "
                    f"got {embedding.shape[1]}"
                )

            # Add to index
            self.index.add(embedding)

            # Add metadata
            meta = ModelMetadata(
                id=model_id,
                name=name,
                category=category,
                file_path=file_path,
                indexed_at=datetime.utcnow().isoformat()
            )
            self.metadata.append(meta)

            idx = self.index.ntotal - 1

            if save:
                self._save()

            return idx

    def add_batch(
        self,
        embeddings: list[list[float]] | np.ndarray,
        metadata_list: list[dict],
        save: bool = True
    ) -> list[int]:
        """
        Add multiple embeddings to the index.

        Args:
            embeddings: List of 1152-dim embeddings
            metadata_list: List of dicts with keys: id, name, category (optional)
            save: Whether to persist to disk

        Returns:
            List of index positions
        """
        with self._lock:
            # Convert to numpy array
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)

            if len(embeddings) != len(metadata_list):
                raise ValueError("Embeddings and metadata must have same length")

            start_idx = self.index.ntotal

            # Add to index
            self.index.add(embeddings)

            # Add metadata
            for meta_dict in metadata_list:
                meta = ModelMetadata(
                    id=meta_dict["id"],
                    name=meta_dict["name"],
                    category=meta_dict.get("category"),
                    file_path=meta_dict.get("file_path"),
                    indexed_at=datetime.utcnow().isoformat()
                )
                self.metadata.append(meta)

            if save:
                self._save()

            return list(range(start_idx, self.index.ntotal))

    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        k: int = 10
    ) -> list[dict]:
        """
        Search for similar models.

        Args:
            query_embedding: 1152-dim query embedding (text or image)
            k: Number of results to return

        Returns:
            List of dicts with 'id', 'name', 'category', 'score'
        """
        with self._lock:
            # Convert to numpy array
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            query_embedding = query_embedding.reshape(1, -1)

            # Clamp k to available vectors
            k = min(k, self.index.ntotal)
            if k == 0:
                return []

            # Search (HNSW returns L2 distances, convert to similarity)
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    # Convert L2 distance to similarity score (1 / (1 + dist))
                    # For normalized vectors, smaller distance = more similar
                    score = 1.0 / (1.0 + dist)
                    results.append({
                        "id": meta.id,
                        "name": meta.name,
                        "category": meta.category,
                        "score": float(score),
                        "distance": float(dist),
                        "file_path": meta.file_path
                    })

            return results

    def get_by_id(self, model_id: str) -> Optional[dict]:
        """Get metadata for a specific model by ID."""
        with self._lock:
            for meta in self.metadata:
                if meta.id == model_id:
                    return meta.to_dict()
            return None

    def list_all(self, skip: int = 0, limit: int = 100) -> list[dict]:
        """List all indexed models."""
        with self._lock:
            return [m.to_dict() for m in self.metadata[skip:skip + limit]]

    @property
    def total(self) -> int:
        """Total number of indexed models."""
        return self.index.ntotal

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_models": self.index.ntotal,
            "embedding_dim": self.EMBEDDING_DIM,
            "index_type": "IndexHNSWFlat",
            "hnsw_m": self.HNSW_M,
            "hnsw_ef_construction": self.HNSW_EF_CONSTRUCTION,
            "hnsw_ef_search": self.HNSW_EF_SEARCH,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "index_size_bytes": self.index_path.stat().st_size if self.index_path.exists() else 0,
            "categories": list(set(m.category for m in self.metadata if m.category))
        }

    def set_search_ef(self, ef: int):
        """
        Adjust search accuracy/speed tradeoff.

        Higher ef = better recall but slower search.
        Default is 64. Range: 16-512 typical.
        """
        with self._lock:
            self.index.hnsw.efSearch = ef


# Global index instance (singleton)
_index: Optional[FAISSIndex] = None


def get_index(
    index_path: str = "models.index",
    metadata_path: str = "metadata.json"
) -> FAISSIndex:
    """Get or create the global FAISS index instance."""
    global _index
    if _index is None:
        _index = FAISSIndex(index_path, metadata_path)
    return _index
