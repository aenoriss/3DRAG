from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import json
import base64
import io
from PIL import Image

from runpod_client import embed_text, embed_images, health_check

app = FastAPI(
    title="3D Model Search API",
    description="Search 3D models using natural language",
    version="1.0.0"
)

# Load FAISS index and metadata at startup
index = faiss.read_index("models.index")
with open("metadata.json") as f:
    metadata = json.load(f)


class SearchResult(BaseModel):
    id: str
    name: str
    score: float
    category: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class AddModelRequest(BaseModel):
    model_id: str
    model_name: str
    category: str | None = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_indexed": index.ntotal,
        "embedding_dim": 1152
    }


@app.get("/health/runpod")
async def runpod_health():
    """Check RunPod endpoint status."""
    try:
        return await health_check()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/search", response_model=SearchResponse)
async def search(q: str, k: int = 10):
    """
    Search for 3D models using natural language.

    - **q**: Search query (e.g., "armored knight", "wooden chair")
    - **k**: Number of results to return (default: 10)
    """
    if k > 100:
        k = 100

    # Get text embedding from RunPod
    embedding = await embed_text(q)
    embedding = np.array(embedding, dtype="float32").reshape(1, -1)

    # Search FAISS index
    scores, ids = index.search(embedding, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < len(metadata):
            m = metadata[idx]
            results.append(SearchResult(
                id=m["id"],
                name=m.get("name", m["id"]),
                score=float(score),
                category=m.get("category")
            ))

    return SearchResponse(query=q, results=results)


@app.post("/models")
async def add_model(
    model_id: str,
    model_name: str,
    category: str | None = None,
    views: list[UploadFile] = File(...)
):
    """
    Add a new 3D model to the index.

    Upload 12 rendered view images (PNG/JPG) of the 3D model.
    """
    if len(views) < 1:
        raise HTTPException(400, "At least 1 view image required")

    # Convert uploaded files to base64
    images_b64 = []
    for view in views:
        content = await view.read()
        # Validate it's an image
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
        except:
            raise HTTPException(400, f"Invalid image: {view.filename}")

        images_b64.append(base64.b64encode(content).decode())

    # Get embeddings from RunPod
    embeddings = await embed_images(images_b64)

    # Average embeddings across views
    avg_embedding = np.mean(embeddings, axis=0).astype("float32").reshape(1, -1)

    # Add to index
    index.add(avg_embedding)
    metadata.append({
        "id": model_id,
        "name": model_name,
        "category": category
    })

    # Persist to disk
    faiss.write_index(index, "models.index")
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "status": "added",
        "id": model_id,
        "views_processed": len(views),
        "total_models": index.ntotal
    }


@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get metadata for a specific model."""
    for m in metadata:
        if m["id"] == model_id:
            return m
    raise HTTPException(404, "Model not found")


@app.get("/models")
async def list_models(skip: int = 0, limit: int = 100):
    """List all indexed models."""
    return {
        "total": len(metadata),
        "models": metadata[skip:skip + limit]
    }
