"""
3D Model Search API - FastAPI application for semantic 3D model search.

Features:
- Upload 3D models (GLB, OBJ, STL, PLY, FBX)
- Automatic rendering + embedding via RunPod or local
- HNSW-based vector search
- Text-based semantic search
- WebSocket for real-time updates
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import asyncio
import json
import os
from typing import Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if we're in local rendering mode
USE_LOCAL_RENDERER = os.getenv("USE_LOCAL_RENDERER", "false").lower() == "true"

# Always import RunPod client for embedding
from runpod_client import (
    embed_text,
    embed_model_bytes,
    embed_images,
    get_stats as get_runpod_stats,
    health_check
)

if USE_LOCAL_RENDERER:
    from local_renderer import render_views
    print("Running in LOCAL RENDER mode (local rendering + RunPod embedding)")
else:
    print("Running in RUNPOD mode (RunPod rendering + embedding)")

from faiss_index import get_index, FAISSIndex


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
    app.state.index = get_index()
    app.state.ws_manager = ws_manager
    print(f"FAISS index ready: {app.state.index.total} models indexed")
    print(f"Mode: {'LOCAL' if USE_LOCAL_RENDERER else 'RUNPOD'}")
    yield
    # Shutdown: Save index
    print("Shutting down...")


app = FastAPI(
    title="3D Model Search API",
    description="Search 3D models using natural language powered by SigLIP2",
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

# Supported 3D formats
SUPPORTED_FORMATS = {
    ".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx", ".dae", ".3ds", ".off"
}


# ============================================================================
# Response Models
# ============================================================================

class SearchResult(BaseModel):
    id: str
    name: str
    score: float
    distance: float
    category: Optional[str] = None
    file_path: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_indexed: int


class ModelResponse(BaseModel):
    status: str
    id: str
    name: str
    embedding_dim: int
    total_models: int
    stats: Optional[dict] = None


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
        "embedding_dim": 1152,
        "index_type": "HNSW"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    index: FAISSIndex = app.state.index

    if USE_LOCAL_RENDERER:
        runpod_status = {"mode": "local", "status": "disabled"}
    else:
        try:
            runpod_status = await health_check()
        except Exception as e:
            runpod_status = {"error": str(e)}

    return {
        "api": "healthy",
        "mode": "local" if USE_LOCAL_RENDERER else "runpod",
        "index": {
            "status": "healthy",
            "models": index.total
        },
        "runpod": runpod_status,
        "websocket_connections": len(ws_manager.active_connections)
    }


@app.get("/stats", response_model=StatsResponse)
async def stats(include_runpod: bool = False):
    """
    Get system statistics.

    - **include_runpod**: Also fetch stats from RunPod (triggers cold start if idle)
    """
    index: FAISSIndex = app.state.index

    result = {"index": index.stats()}

    if include_runpod:
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

    Uses SigLIP2 to embed the query and HNSW for fast similarity search.
    """
    index: FAISSIndex = app.state.index

    if index.total == 0:
        return SearchResponse(query=q, results=[], total_indexed=0)

    # Get text embedding from RunPod
    result = await embed_text(q)
    embedding = result["embedding"]

    # Search FAISS index
    results = index.search(embedding, k)

    return SearchResponse(
        query=q,
        results=[SearchResult(**r) for r in results],
        total_indexed=index.total
    )


@app.post("/models", response_model=ModelResponse)
async def add_model(
    file: UploadFile = File(..., description="3D model file (GLB, OBJ, STL, PLY, FBX)"),
    model_id: str = Form(..., description="Unique identifier for the model"),
    name: str = Form(..., description="Display name"),
    category: Optional[str] = Form(None, description="Optional category"),
    include_stats: bool = Form(False, description="Include processing stats")
):
    """
    Add a new 3D model to the index.

    The model is rendered from 12 camera angles, embedded with SigLIP2,
    and the averaged embedding is indexed in FAISS.
    """
    index: FAISSIndex = app.state.index

    # Validate file extension
    filename = file.filename or "model.glb"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            400,
            f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Check if model_id already exists
    if index.get_by_id(model_id):
        raise HTTPException(400, f"Model with ID '{model_id}' already exists")

    # Read file content
    content = await file.read()

    # Broadcast processing started
    await ws_manager.broadcast({
        "type": "model_processing",
        "model_id": model_id,
        "name": name,
        "status": "processing"
    })

    # Process model (local rendering or full RunPod)
    try:
        if USE_LOCAL_RENDERER:
            # Local rendering + RunPod embedding
            images, images_b64 = render_views(content, ext.lstrip("."))

            # Send images to RunPod for embedding
            embed_result = await embed_images(images_b64, include_stats=include_stats)

            # Average the embeddings
            embeddings = np.array(embed_result["embeddings"], dtype=np.float32)
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            result = {
                "embedding": avg_embedding.tolist(),
                "images_b64": images_b64,
                "views_rendered": len(images)
            }
            if "stats" in embed_result:
                result["stats"] = embed_result["stats"]
        else:
            # Full RunPod processing (render + embed)
            result = await embed_model_bytes(
                model_bytes=content,
                file_format=ext.lstrip("."),
                include_stats=include_stats
            )
    except Exception as e:
        await ws_manager.broadcast({
            "type": "model_error",
            "model_id": model_id,
            "error": str(e)
        })
        raise HTTPException(502, f"Processing error: {str(e)}")

    # Add to FAISS index
    embedding = result["embedding"]
    index.add(
        embedding=embedding,
        model_id=model_id,
        name=name,
        category=category,
        file_path=filename
    )

    # Broadcast model added
    await ws_manager.broadcast({
        "type": "model_added",
        "model_id": model_id,
        "name": name,
        "category": category,
        "total_models": index.total,
        "images_b64": result.get("images_b64", [])  # Only in local mode
    })

    response = ModelResponse(
        status="added",
        id=model_id,
        name=name,
        embedding_dim=len(embedding),
        total_models=index.total
    )

    if include_stats and "stats" in result:
        response.stats = result["stats"]

    return response


@app.post("/models/batch")
async def add_models_batch(
    files: list[UploadFile] = File(..., description="3D model files"),
    model_ids: str = Form(..., description="Comma-separated model IDs"),
    names: str = Form(..., description="Comma-separated display names"),
    categories: Optional[str] = Form(None, description="Comma-separated categories (optional)")
):
    """
    Add multiple 3D models in batch.

    Note: Models are processed sequentially. For large batches,
    consider using the async endpoint (coming soon).
    """
    index: FAISSIndex = app.state.index

    ids = [x.strip() for x in model_ids.split(",")]
    name_list = [x.strip() for x in names.split(",")]
    cat_list = [x.strip() for x in categories.split(",")] if categories else [None] * len(ids)

    if len(files) != len(ids) or len(files) != len(name_list):
        raise HTTPException(400, "Number of files, IDs, and names must match")

    results = []
    for file, mid, mname, cat in zip(files, ids, name_list, cat_list):
        try:
            filename = file.filename or "model.glb"
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            content = await file.read()

            result = await embed_model_bytes(
                model_bytes=content,
                file_format=ext.lstrip(".")
            )

            index.add(
                embedding=result["embedding"],
                model_id=mid,
                name=mname,
                category=cat,
                file_path=filename
            )

            results.append({"id": mid, "status": "added"})
        except Exception as e:
            results.append({"id": mid, "status": "error", "error": str(e)})

    return {
        "processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "added"),
        "results": results,
        "total_models": index.total
    }


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
            "mode": "local" if USE_LOCAL_RENDERER else "runpod"
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
# Local Rendering Endpoint (for testing)
# ============================================================================

@app.post("/render")
async def render_model_preview(
    file: UploadFile = File(..., description="3D model file")
):
    """
    Render a 3D model and return the 12 view images.

    Only available in local render mode. For testing/preview purposes.
    """
    if not USE_LOCAL_RENDERER:
        raise HTTPException(400, "Render endpoint only available in local mode. Set USE_LOCAL_RENDERER=true")

    filename = file.filename or "model.glb"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    content = await file.read()

    try:
        images, images_b64 = render_views(content, ext.lstrip("."))
        return {
            "views": len(images_b64),
            "images_b64": images_b64
        }
    except Exception as e:
        raise HTTPException(500, f"Render error: {str(e)}")


# ============================================================================
# Run with: uvicorn main:app --reload
# Or local mode: USE_LOCAL_RENDERER=true uvicorn main:app --reload
# ============================================================================
