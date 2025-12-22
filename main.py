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

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, File, UploadFile, Form
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


@app.post("/models")
async def upload_model(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    name: str = Form(...),
    include_stats: bool = Form(False)
):
    """
    Upload and process a 3D model file.

    Sends the file to RunPod for rendering, captioning, and embedding.
    Then adds the result to the local FAISS index.
    """
    from runpod_client import process_model_bytes
    from pathlib import Path
    import base64

    # Validate file extension
    supported = ['.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.dae', '.3ds']
    file_ext = Path(file.filename).suffix.lower() if file.filename else '.glb'
    if file_ext not in supported:
        raise HTTPException(400, f"Unsupported format: {file_ext}. Supported: {supported}")

    try:
        # Read file bytes
        model_bytes = await file.read()
        print(f"[upload] Processing {name} ({len(model_bytes)} bytes)")

        # Send to RunPod for processing
        result = await process_model_bytes(
            model_bytes=model_bytes,
            file_extension=file_ext.lstrip('.'),
            name=name
        )

        if "error" in result:
            raise HTTPException(500, result["error"])

        # Add to index
        index: FAISSIndex = app.state.index
        index.add(
            embedding=result["embedding"],
            model_id=model_id,
            name=result.get("name", name),
            category=None,
            file_path=None,
            caption=result.get("caption", ""),
            save=True
        )

        # Save preview image
        preview_b64 = result.get("preview")
        if preview_b64:
            from dataset_generator import DATASET_DIR
            previews_dir = DATASET_DIR / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)

            preview_bytes = base64.b64decode(preview_b64)
            preview_path = previews_dir / f"{model_id}.jpg"
            preview_path.write_bytes(preview_bytes)

        # Broadcast update
        await ws_manager.broadcast({
            "type": "model_added",
            "model_id": model_id,
            "name": name,
            "caption": result.get("caption", "")
        })

        return {
            "model_id": model_id,
            "name": name,
            "caption": result.get("caption", ""),
            "time_sec": result.get("time_sec", 0),
            "indexed": True
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[upload] Error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


# Batch upload staging area
batch_staging: dict[str, list] = {}  # session_id -> list of model dicts


@app.post("/models/batch/start")
async def batch_start(
    total: int = Query(..., description="Total number of files to upload"),
    clear: bool = Query(True, description="Clear existing index before processing")
):
    """Start a batch upload session."""
    import uuid
    import shutil
    from dataset_generator import DATASET_DIR

    session_id = str(uuid.uuid4())[:8]
    batch_staging[session_id] = {"total_received": 0, "total_processed": 0}

    # Clear index if requested
    if clear:
        index: FAISSIndex = app.state.index
        index.clear()

        previews_dir = DATASET_DIR / "previews"
        if previews_dir.exists():
            shutil.rmtree(previews_dir)
        previews_dir.mkdir(parents=True, exist_ok=True)

        print(f"[batch] Session {session_id}: Cleared index, expecting {total} files")

    await ws_manager.broadcast({
        "type": "batch_start",
        "session_id": session_id,
        "total": total,
        "clear": clear
    })

    return {"session_id": session_id, "total": total}


@app.post("/models/batch/upload/{session_id}")
async def batch_upload(
    session_id: str,
    files: list[UploadFile] = File(...)
):
    """Upload files to bucket, then process on RunPod via URLs."""
    import base64
    from pathlib import Path
    from runpod_client import process_models_urls
    from dataset_generator import DATASET_DIR
    from storage import upload_file

    if session_id not in batch_staging:
        raise HTTPException(404, "Session not found")

    supported = ['.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.dae', '.3ds']

    # Upload to bucket and collect URLs
    models = []
    for file in files:
        ext = Path(file.filename).suffix.lower() if file.filename else ''
        if ext not in supported:
            continue

        model_bytes = await file.read()
        idx = batch_staging[session_id]["total_received"]
        model_id = f"batch_{idx}_{Path(file.filename).stem}"

        # Upload to bucket
        key = f"models/{session_id}/{model_id}{ext}"
        url = upload_file(model_bytes, key)

        models.append({
            "model_id": model_id,
            "name": Path(file.filename).stem,
            "url": url,
            "extension": ext.lstrip('.')
        })
        batch_staging[session_id]["total_received"] += 1

    if not models:
        return {"received": batch_staging[session_id]["total_received"], "processed": 0}

    # Send URLs to RunPod (much smaller payload!)
    print(f"[batch] Session {session_id}: Processing {len(models)} models via URLs...")
    result = await process_models_urls(models)

    # Add results to index
    index: FAISSIndex = app.state.index
    previews_dir = DATASET_DIR / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for r in result.get("results", []):
        if "error" in r:
            continue
        try:
            index.add(
                embedding=r["embedding"],
                model_id=r["model_id"],
                name=r.get("name", ""),
                category=None,
                file_path=None,
                caption=r.get("caption", ""),
                save=False
            )
            if r.get("preview"):
                preview_bytes = base64.b64decode(r["preview"])
                preview_path = previews_dir / f"{r['model_id']}.jpg"
                preview_path.write_bytes(preview_bytes)
            added += 1
            batch_staging[session_id]["total_processed"] += 1
        except Exception as e:
            print(f"[batch] Error: {e}")

    index.save()

    total_received = batch_staging[session_id]["total_received"]
    total_processed = batch_staging[session_id]["total_processed"]

    await ws_manager.broadcast({
        "type": "batch_progress",
        "session_id": session_id,
        "received": total_received,
        "processed": total_processed
    })

    print(f"[batch] Session {session_id}: {total_processed}/{total_received} processed")
    return {"received": total_received, "processed": total_processed, "added": added}


@app.post("/models/batch/process/{session_id}")
async def batch_process(session_id: str):
    """Process all uploaded files in a batch session."""
    from runpod_client import process_models_batch
    from dataset_generator import DATASET_DIR
    import base64

    if session_id not in batch_staging:
        raise HTTPException(404, "Session not found")

    models = batch_staging.pop(session_id)
    if not models:
        raise HTTPException(400, "No files in session")

    print(f"[batch] Session {session_id}: Processing {len(models)} models...")

    # Send to RunPod
    result = await process_models_batch(models)

    if "error" in result:
        raise HTTPException(500, result["error"])

    # Add results to index
    index: FAISSIndex = app.state.index
    previews_dir = DATASET_DIR / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    failed = 0
    for r in result.get("results", []):
        if "error" in r:
            failed += 1
            continue

        try:
            index.add(
                embedding=r["embedding"],
                model_id=r["model_id"],
                name=r.get("name", ""),
                category=None,
                file_path=None,
                caption=r.get("caption", ""),
                save=False
            )

            if r.get("preview"):
                preview_bytes = base64.b64decode(r["preview"])
                preview_path = previews_dir / f"{r['model_id']}.jpg"
                preview_path.write_bytes(preview_bytes)

            added += 1
        except Exception as e:
            print(f"[batch] Error adding {r.get('model_id')}: {e}")
            failed += 1

    index.save()

    await ws_manager.broadcast({
        "type": "batch_complete",
        "session_id": session_id,
        "added": added,
        "failed": failed,
        "time_sec": result.get("time_sec", 0)
    })

    return {
        "added": added,
        "failed": failed,
        "total": len(models),
        "time_sec": result.get("time_sec", 0)
    }


@app.post("/models/batch")
async def upload_models_batch(
    files: list[UploadFile] = File(...),
    clear: bool = Query(True, description="Clear existing index before processing")
):
    """
    Legacy: Upload and process multiple 3D model files in a single batch.
    For large batches, use /models/batch/start + /upload + /process instead.
    """
    from runpod_client import process_models_batch
    from pathlib import Path
    from dataset_generator import DATASET_DIR
    import base64
    import shutil

    supported = ['.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.dae', '.3ds']

    # Validate files
    valid_files = []
    for file in files:
        ext = Path(file.filename).suffix.lower() if file.filename else ''
        if ext in supported:
            valid_files.append((file, ext))

    if not valid_files:
        raise HTTPException(400, f"No valid 3D files. Supported: {supported}")

    print(f"[batch] Processing {len(valid_files)} files (clear={clear})")

    # Broadcast start
    await ws_manager.broadcast({
        "type": "batch_start",
        "total": len(valid_files),
        "clear": clear
    })

    try:
        # Clear index if requested
        if clear:
            index: FAISSIndex = app.state.index
            index.clear()

            # Clear previews
            previews_dir = DATASET_DIR / "previews"
            if previews_dir.exists():
                shutil.rmtree(previews_dir)
            previews_dir.mkdir(parents=True, exist_ok=True)

            print(f"[batch] Cleared index and previews")

        # Prepare batch payload
        models = []
        for i, (file, ext) in enumerate(valid_files):
            model_bytes = await file.read()
            model_id = f"batch_{i}_{Path(file.filename).stem}"
            model_name = Path(file.filename).stem

            models.append({
                "model_id": model_id,
                "name": model_name,
                "bytes_b64": base64.b64encode(model_bytes).decode(),
                "extension": ext.lstrip('.')
            })

        print(f"[batch] Sending {len(models)} models to RunPod...")

        # Send to RunPod
        result = await process_models_batch(models)

        if "error" in result:
            raise HTTPException(500, result["error"])

        # Add results to index
        index: FAISSIndex = app.state.index
        previews_dir = DATASET_DIR / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        failed = 0
        for r in result.get("results", []):
            if "error" in r:
                failed += 1
                continue

            try:
                index.add(
                    embedding=r["embedding"],
                    model_id=r["model_id"],
                    name=r.get("name", ""),
                    category=None,
                    file_path=None,
                    caption=r.get("caption", ""),
                    save=False  # Save once at the end
                )

                # Save preview
                if r.get("preview"):
                    preview_bytes = base64.b64decode(r["preview"])
                    preview_path = previews_dir / f"{r['model_id']}.jpg"
                    preview_path.write_bytes(preview_bytes)

                added += 1
            except Exception as e:
                print(f"[batch] Error adding {r.get('model_id')}: {e}")
                failed += 1

        # Save index
        index.save()

        # Broadcast complete
        await ws_manager.broadcast({
            "type": "batch_complete",
            "added": added,
            "failed": failed,
            "total": len(valid_files),
            "time_sec": result.get("time_sec", 0)
        })

        print(f"[batch] Complete: {added} added, {failed} failed")

        return {
            "status": "complete",
            "added": added,
            "failed": failed,
            "total": len(valid_files),
            "time_sec": result.get("time_sec", 0),
            "time_per_model": result.get("time_per_model", 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[batch] Error: {e}")
        await ws_manager.broadcast({
            "type": "batch_error",
            "error": str(e)
        })
        raise HTTPException(500, f"Batch processing failed: {str(e)}")


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
