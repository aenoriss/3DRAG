"""
3D Model Search API - FastAPI application for semantic 3D model search.

Features:
- HNSW-based vector search
- Text-based semantic search
- WebSocket for real-time updates
- Dataset generation via RunPod GPU

All heavy processing (download, render, caption, embed) happens on RunPod.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import json
import os
from typing import Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RunPod client and FAISS
from runpod_client import get_stats as get_runpod_stats, health_check
from faiss_index import get_index, FAISSIndex, EMBEDDING_DIM_GEMMA
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = EMBEDDING_DIM_GEMMA  # 768, same as all-mpnet-base-v2

# Local embedding model for search queries (CPU, fast)
EMBEDDING_MODEL = "all-mpnet-base-v2"
_embed_model = None

print(f"Running with sentence-transformers for search ({EMBEDDING_DIM}-dim)")


def get_embed_model():
    """Load embedding model (lazy, singleton)."""
    global _embed_model
    if _embed_model is None:
        print(f"Loading {EMBEDDING_MODEL}...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready!")
    return _embed_model


def embed_text_local(text: str) -> list[float]:
    """Embed text using local sentence-transformers (fast, CPU)."""
    model = get_embed_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        data = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except:
                self.active_connections.discard(connection)


ws_manager = ConnectionManager()


# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize FAISS index
    app.state.index = get_index(embedding_dim=EMBEDDING_DIM)
    app.state.ws_manager = ws_manager
    print(f"FAISS index ready: {app.state.index.total} models indexed")
    yield
    # Shutdown: Save index
    print("Shutting down...")


app = FastAPI(
    title="3D Model Search API",
    description="Search 3D models using natural language",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve preview images statically
from pathlib import Path
DATASET_DIR = Path("dataset")
PREVIEWS_DIR = DATASET_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")



# ============================================================================
# Response Models
# ============================================================================

class SearchResult(BaseModel):
    id: str
    name: str
    score: float
    distance: float
    category: Optional[str] = None
    caption: Optional[str] = None
    file_path: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_indexed: int




class StatsResponse(BaseModel):
    index: dict
    runpod: Optional[dict] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check and basic info."""
    index: FAISSIndex = app.state.index
    return {
        "status": "ok",
        "service": "3D Model Search API",
        "version": "2.0.0",
        "models_indexed": index.total,
        "embedding_dim": EMBEDDING_DIM,
        "index_type": "HNSW"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    index: FAISSIndex = app.state.index

    try:
        runpod_status = await health_check()
    except Exception as e:
        runpod_status = {"error": str(e)}

    return {
        "api": "healthy",
        "mode": "runpod",
        "embedding_dim": EMBEDDING_DIM,
        "index": {
            "status": "healthy",
            "models": index.total
        },
        "runpod": runpod_status,
        "websocket_connections": len(ws_manager.active_connections)
    }


@app.get("/stats", response_model=StatsResponse)
async def stats(include_backend: bool = False):
    """
    Get system statistics.

    - **include_backend**: Also fetch stats from RunPod endpoint
    """
    index: FAISSIndex = app.state.index

    result = {
        "index": index.stats(),
        "mode": "runpod"
    }

    if include_backend:
        try:
            result["runpod"] = await get_runpod_stats()
        except Exception as e:
            result["runpod"] = {"error": str(e)}

    return result


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query (e.g., 'wooden chair', 'armored knight')"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    """
    Search for 3D models using natural language.

    Embeds query locally via Ollama and uses HNSW for fast similarity search.
    """
    index: FAISSIndex = app.state.index

    if index.total == 0:
        return SearchResponse(query=q, results=[], total_indexed=0)

    # Get text embedding from local sentence-transformers (fast, CPU)
    try:
        embedding = embed_text_local(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    # Search FAISS index
    results = index.search(embedding, k)

    return SearchResponse(
        query=q,
        results=[SearchResult(**r) for r in results],
        total_indexed=index.total
    )




@app.get("/models")
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """List all indexed models."""
    index: FAISSIndex = app.state.index
    return {
        "total": index.total,
        "skip": skip,
        "limit": limit,
        "models": index.list_all(skip, limit)
    }


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get metadata for a specific model."""
    index: FAISSIndex = app.state.index
    model = index.get_by_id(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return model


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """
    Delete a model from the index.

    Note: This rebuilds the FAISS index without the deleted vector.
    """
    index: FAISSIndex = app.state.index

    # Check if model exists
    model = index.get_by_id(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")

    # Remove from index
    removed = index.remove_by_id(model_id)
    if not removed:
        raise HTTPException(500, f"Failed to remove model '{model_id}'")

    # Broadcast deletion
    await ws_manager.broadcast({
        "type": "model_deleted",
        "model_id": model_id,
        "total_models": index.total
    })

    return {
        "status": "deleted",
        "model_id": model_id,
        "remaining_models": index.total
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time updates.

    Events:
    - model_processing: When a model starts processing
    - model_added: When a model is successfully added
    - model_error: When processing fails
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial state
        index: FAISSIndex = app.state.index
        await websocket.send_json({
            "type": "connected",
            "total_models": index.total,
            "mode": "runpod"
        })

        # Keep connection alive
        while True:
            try:
                # Wait for messages (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)




# ============================================================================
# Dataset Generation Endpoints
# ============================================================================

from dataset_generator import (
    generate_dataset,
    get_status as get_dataset_status,
    DatasetStatus,
    clear_dataset,
    cancel_generation,
    DATASET_SIZE
)

# Background task for dataset generation
_generation_task: Optional[asyncio.Task] = None


@app.post("/dataset/generate")
async def start_dataset_generation(
    count: int = Query(DATASET_SIZE, ge=10, le=1000, description="Number of models to download")
):
    """
    Start generating a new dataset from Objaverse-XL.

    This will:
    1. Delete the existing dataset and FAISS index
    2. Download `count` GLB models from Objaverse-XL
    3. Render and embed each model
    4. Index them in FAISS

    Progress updates are sent via WebSocket (streaming per-model).
    """
    global _generation_task

    status = get_dataset_status()
    if status.is_generating:
        raise HTTPException(409, "Dataset generation already in progress")

    async def progress_callback(data: dict):
        await ws_manager.broadcast(data)

    async def run_generation():
        try:
            # Reinitialize the index after clearing
            result = await generate_dataset(
                count=count,
                progress_callback=progress_callback
            )
            # Reload the index in app state
            from faiss_index import FAISSIndex
            app.state.index = FAISSIndex(embedding_dim=EMBEDDING_DIM)
            return result
        except Exception as e:
            await ws_manager.broadcast({
                "type": "dataset_error",
                "error": str(e)
            })
            raise

    _generation_task = asyncio.create_task(run_generation())

    return {
        "status": "started",
        "count": count,
        "message": "Dataset generation started. Watch WebSocket for progress."
    }


@app.get("/dataset/status")
async def dataset_generation_status():
    """Get current dataset generation status."""
    status = get_dataset_status()
    return {
        "is_generating": status.is_generating,
        "total": status.total,
        "processed": status.processed,
        "indexed": status.indexed,
        "failed": status.failed,
        "current_model": status.current_model,
        "started_at": status.started_at,
        "error": status.error
    }


@app.delete("/dataset")
async def delete_dataset():
    """Delete the current dataset and clear the FAISS index."""
    status = get_dataset_status()
    if status.is_generating:
        raise HTTPException(409, "Cannot delete while generation is in progress")

    await clear_dataset(clear_index=True)

    # Reinitialize empty index
    from faiss_index import FAISSIndex
    app.state.index = FAISSIndex(embedding_dim=EMBEDDING_DIM)

    await ws_manager.broadcast({
        "type": "dataset_cleared",
        "message": "Dataset deleted"
    })

    return {"status": "deleted", "message": "Dataset and index cleared"}


@app.post("/dataset/cancel")
async def cancel_dataset_generation_endpoint():
    """Cancel ongoing dataset generation."""
    global _generation_task

    status = get_dataset_status()
    if not status.is_generating:
        raise HTTPException(400, "No generation in progress")

    # Set cancellation flag - the loop will check this
    cancel_generation()

    # Wait a moment for the loop to notice
    await asyncio.sleep(0.5)

    if _generation_task:
        _generation_task.cancel()
        _generation_task = None

    # Reset the status
    from dataset_generator import reset_status
    reset_status()

    await ws_manager.broadcast({
        "type": "dataset_cancelled",
        "message": "Generation cancelled"
    })

    return {"status": "cancelled"}


# ============================================================================
# Run with: uvicorn main:app --reload
# ============================================================================
