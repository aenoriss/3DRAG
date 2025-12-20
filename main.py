"""
3D Model Search API - FastAPI application for semantic 3D model search.

Features:
- Upload 3D models (GLB, OBJ, STL, PLY, FBX)
- Automatic rendering + embedding via Ollama (default) or RunPod
- HNSW-based vector search
- Text-based semantic search
- WebSocket for real-time updates

Embedding modes:
- ollama (default): Gemma 3 27B vision + EmbeddingGemma (768-dim)
- runpod: SigLIP2 via RunPod (1152-dim)
"""
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import asyncio
import json
import os
from typing import Optional, Set, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Embedding mode: "ollama" (default) or "runpod"
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "ollama").lower()
USE_OLLAMA = EMBEDDING_MODE == "ollama"

# Check if we're in local rendering mode (for RunPod mode)
USE_LOCAL_RENDERER = os.getenv("USE_LOCAL_RENDERER", "false").lower() == "true"

# Import based on mode
if USE_OLLAMA:
    from ollama_client import (
        process_3d_model as ollama_process_3d_model,
        embed_query as ollama_embed_query,
        check_ollama,
        reset_stats as ollama_reset_stats
    )
    from local_renderer import render_views
    from faiss_index import get_index, FAISSIndex, EMBEDDING_DIM_GEMMA
    EMBEDDING_DIM = EMBEDDING_DIM_GEMMA
    print(f"Running in OLLAMA mode (Gemma 3 27B + EmbeddingGemma, {EMBEDDING_DIM}-dim)")
else:
    from runpod_client import (
        embed_text,
        embed_model_bytes,
        embed_images,
        get_stats as get_runpod_stats,
        health_check
    )
    from faiss_index import get_index, FAISSIndex, EMBEDDING_DIM_SIGLIP
    EMBEDDING_DIM = EMBEDDING_DIM_SIGLIP

    if USE_LOCAL_RENDERER:
        from local_renderer import render_views
        print(f"Running in LOCAL RENDER mode (local rendering + RunPod, {EMBEDDING_DIM}-dim)")
    else:
        print(f"Running in RUNPOD mode (RunPod rendering + embedding, {EMBEDDING_DIM}-dim)")


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
    # Startup: Initialize FAISS index with correct dimension for mode
    app.state.index = get_index(embedding_dim=EMBEDDING_DIM)
    app.state.ws_manager = ws_manager
    print(f"FAISS index ready: {app.state.index.total} models indexed")
    print(f"Mode: {'OLLAMA' if USE_OLLAMA else ('LOCAL' if USE_LOCAL_RENDERER else 'RUNPOD')}")
    yield
    # Shutdown: Save index
    print("Shutting down...")


app = FastAPI(
    title="3D Model Search API",
    description="Search 3D models using natural language (Ollama or SigLIP2)",
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

    # Check embedding backend based on mode
    if USE_OLLAMA:
        try:
            ollama_status = await check_ollama()
        except Exception as e:
            ollama_status = {"status": "error", "error": str(e)}
        backend_status = {"ollama": ollama_status}
        mode = "ollama"
    elif USE_LOCAL_RENDERER:
        backend_status = {"runpod": {"mode": "local", "status": "disabled"}}
        mode = "local"
    else:
        try:
            runpod_status = await health_check()
        except Exception as e:
            runpod_status = {"error": str(e)}
        backend_status = {"runpod": runpod_status}
        mode = "runpod"

    return {
        "api": "healthy",
        "mode": mode,
        "embedding_dim": EMBEDDING_DIM,
        "index": {
            "status": "healthy",
            "models": index.total
        },
        **backend_status,
        "websocket_connections": len(ws_manager.active_connections)
    }


@app.get("/stats", response_model=StatsResponse)
async def stats(include_backend: bool = False):
    """
    Get system statistics.

    - **include_backend**: Also fetch stats from embedding backend (Ollama or RunPod)
    """
    index: FAISSIndex = app.state.index

    result = {
        "index": index.stats(),
        "mode": "ollama" if USE_OLLAMA else ("local" if USE_LOCAL_RENDERER else "runpod")
    }

    if include_backend:
        if USE_OLLAMA:
            try:
                result["ollama"] = await check_ollama()
            except Exception as e:
                result["ollama"] = {"error": str(e)}
        elif not USE_LOCAL_RENDERER:
            try:
                result["runpod"] = await get_runpod_stats()
            except Exception as e:
                result["runpod"] = {"error": str(e)}

    return result


@app.post("/stats/reset")
async def reset_stats():
    """Reset cumulative stats on the embedding backend."""
    if USE_OLLAMA:
        try:
            result = await ollama_reset_stats()
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Reset only available in Ollama mode")


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="Search query (e.g., 'wooden chair', 'armored knight')"),
    k: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    """
    Search for 3D models using natural language.

    Uses Ollama (default) or SigLIP2 to embed the query and HNSW for fast similarity search.
    """
    index: FAISSIndex = app.state.index

    if index.total == 0:
        return SearchResponse(query=q, results=[], total_indexed=0)

    # Get text embedding based on mode
    if USE_OLLAMA:
        result = await ollama_embed_query(q)
        if result["status"] != "ok":
            raise HTTPException(status_code=500, detail=f"Ollama error: {result.get('error')}")
        embedding = result["embedding"]
    else:
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

            # Max pooling: keeps strongest features, ignores bad views
            embeddings = np.array(embed_result["embeddings"], dtype=np.float32)
            max_embedding = np.max(embeddings, axis=0)
            max_embedding = max_embedding / np.linalg.norm(max_embedding)

            result = {
                "embedding": max_embedding.tolist(),
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

@app.get("/test-runpod")
async def test_runpod():
    """
    Quick test to verify RunPod embedding is working.
    Sends a simple test image to RunPod and returns the result.
    """
    from PIL import Image
    import io
    import base64

    # Create a simple test image (red square)
    img = Image.new("RGB", (384, 384), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    try:
        result = await embed_images([img_b64], include_stats=True)
        return {
            "status": "ok",
            "message": "RunPod is working!",
            "embedding_dim": len(result["embeddings"][0]) if "embeddings" in result else None,
            "stats": result.get("stats")
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


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
        "downloaded": status.downloaded,
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

    await clear_dataset()

    # Reinitialize empty index
    from faiss_index import FAISSIndex
    app.state.index = FAISSIndex(embedding_dim=EMBEDDING_DIM)

    await ws_manager.broadcast({
        "type": "dataset_cleared",
        "message": "Dataset deleted"
    })

    return {"status": "deleted", "message": "Dataset and index cleared"}


@app.post("/dataset/index-existing")
async def index_existing_models(
    limit: int = Query(100, ge=1, le=1000, description="Max models to index")
):
    """
    Index already downloaded models from ~/.objaverse cache.
    Skips download phase - useful for re-indexing after fixing issues.
    """
    import objaverse
    from pathlib import Path

    index: FAISSIndex = app.state.index

    # Find existing GLB files
    cache_dir = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"
    if not cache_dir.exists():
        raise HTTPException(404, "No cached models found. Run generate first.")

    glb_files = list(cache_dir.rglob("*.glb"))[:limit]
    if not glb_files:
        raise HTTPException(404, "No GLB files found in cache")

    await ws_manager.broadcast({
        "type": "dataset_progress",
        "step": "indexing",
        "message": f"Found {len(glb_files)} cached models, indexing...",
        "total": len(glb_files),
        "downloaded": len(glb_files),
        "indexed": 0,
        "failed": 0
    })

    # Get annotations for names
    uids = [f.stem for f in glb_files]

    def load_ann():
        return objaverse.load_annotations(uids)

    import concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        annotations = await loop.run_in_executor(pool, load_ann)

    indexed = 0
    failed = 0

    for glb_file in glb_files:
        uid = glb_file.stem
        ann = annotations.get(uid, {})
        name = ann.get("name", uid[:20])

        try:
            model_bytes = glb_file.read_bytes()

            if USE_LOCAL_RENDERER:
                import numpy as np
                images, images_b64 = render_views(model_bytes, "glb")
                embed_result = await embed_images(images_b64)
                embeddings = np.array(embed_result["embeddings"], dtype=np.float32)
                # Max pooling: keeps strongest features, ignores bad views
                max_embedding = np.max(embeddings, axis=0)
                max_embedding = max_embedding / np.linalg.norm(max_embedding)
                embedding = max_embedding.tolist()
            else:
                result = await embed_model_bytes(model_bytes, "glb")
                embedding = result["embedding"]

            index.add(
                embedding=embedding,
                model_id=uid,
                name=name,
                category=ann.get("categories", [None])[0] if ann.get("categories") else None,
                file_path=str(glb_file),
                save=False
            )
            indexed += 1

            await ws_manager.broadcast({
                "type": "dataset_progress",
                "step": "indexing",
                "message": f"Indexed {indexed}/{len(glb_files)}",
                "total": len(glb_files),
                "downloaded": len(glb_files),
                "indexed": indexed,
                "failed": failed,
                "current": name
            })

        except Exception as e:
            failed += 1
            print(f"Failed to index {uid}: {e}")

    index._save()

    await ws_manager.broadcast({
        "type": "dataset_complete",
        "indexed": indexed,
        "failed": failed
    })

    return {
        "status": "completed",
        "indexed": indexed,
        "failed": failed,
        "total": len(glb_files)
    }


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
# Or local mode: USE_LOCAL_RENDERER=true uvicorn main:app --reload
# ============================================================================
