"""
Dataset Generator - Download and index models from Objaverse.

Creates a mini-dataset of 100 models for testing.
When regenerated, the previous dataset is deleted.
Downloads GLB files directly from Hugging Face for efficiency.

Supports two embedding modes:
- ollama (default): Gemma 3 27B vision + EmbeddingGemma (768-dim)
- runpod: SigLIP2 via RunPod (1152-dim)
"""
from __future__ import annotations

import objaverse
import asyncio
import shutil
import random
import time
import base64
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from PIL import Image

# Number of parallel render processes
RENDER_WORKERS = os.cpu_count() or 4

# Dataset storage
DATASET_DIR = Path("dataset")


def render_single_model(args: tuple) -> dict:
    """
    Render a single model (runs in separate process).

    Args:
        args: (file_id, local_path, name, category)

    Returns:
        Dict with render result or error
    """
    file_id, local_path, name, category = args

    try:
        from local_renderer import render_views

        model_path = Path(local_path)
        if not model_path.exists():
            return {"error": "File not found", "file_id": file_id}

        model_bytes = model_path.read_bytes()
        ext = model_path.suffix.lstrip(".")

        images, images_b64 = render_views(model_bytes, ext, num_views=1)

        # Save files
        renders_dir = DATASET_DIR / "renders" / file_id
        renders_dir.mkdir(parents=True, exist_ok=True)
        images[0].save(renders_dir / f"view_00.png")

        previews_dir = DATASET_DIR / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        images[0].resize((128, 128)).save(previews_dir / f"{file_id}.jpg", "JPEG", quality=85)

        # Delete 3D model file
        try:
            model_path.unlink()
        except Exception:
            pass

        return {
            "file_id": file_id,
            "name": name,
            "category": category,
            "local_path": str(local_path),
            "image_b64": images_b64[0],
        }

    except Exception as e:
        return {"error": str(e), "file_id": file_id}


DATASET_SIZE = 100
BATCH_SIZE = 100  # Process all models in one batch (GPU has 40GB, we use ~8GB)

# Embedding mode: "ollama" (default) or "runpod"
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "ollama").lower()


@dataclass
class DatasetStatus:
    """Current dataset generation status."""
    is_generating: bool = False
    cancelled: bool = False
    total: int = 0
    downloaded: int = 0
    indexed: int = 0
    failed: int = 0
    current_model: Optional[str] = None
    started_at: Optional[str] = None
    error: Optional[str] = None


# Global status
_status = DatasetStatus()


def cancel_generation():
    """Set cancellation flag."""
    global _status
    _status.cancelled = True


def get_status() -> DatasetStatus:
    """Get current dataset generation status."""
    return _status


def reset_status():
    """Reset status for new generation."""
    global _status
    _status = DatasetStatus()
    _status.cancelled = False


async def clear_dataset(clear_index: bool = False):
    """Delete existing dataset directory. Optionally clear FAISS index."""
    # Only clear dataset directory (renders, downloads)
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    DATASET_DIR.mkdir(exist_ok=True)

    # Optionally clear FAISS index
    if clear_index:
        index_path = Path("models.index")
        metadata_path = Path("metadata.json")

        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()


async def generate_dataset(
    count: int = DATASET_SIZE,
    progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict:
    """
    Generate a new dataset from Objaverse-XL.

    Args:
        count: Number of models to download (default 100)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with generation stats
    """
    global _status

    if _status.is_generating:
        raise RuntimeError("Dataset generation already in progress")

    reset_status()
    _status.is_generating = True
    _status.total = count
    _status.started_at = datetime.utcnow().isoformat()

    try:
        # Clear existing dataset
        if progress_callback:
            await progress_callback({
                "type": "dataset_status",
                "status": "clearing",
                "message": "Clearing existing dataset...",
                "total": count,
                "downloaded": 0,
                "indexed": 0,
                "failed": 0
            })

        await clear_dataset()

        # Get UIDs and annotations from Objaverse 1.0 (simpler, direct HF downloads)
        loop = asyncio.get_event_loop()

        if progress_callback:
            await progress_callback({
                "type": "dataset_progress",
                "step": "downloading",
                "message": "Loading Objaverse annotations...",
                "total": count,
                "downloaded": 0,
                "indexed": 0,
                "failed": 0
            })

        # Load UIDs in thread
        def load_uids():
            return objaverse.load_uids()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            all_uids = await loop.run_in_executor(pool, load_uids)

        print(f"Loaded {len(all_uids)} UIDs from Objaverse")

        # Sample random UIDs
        selected_uids = random.sample(all_uids, min(count, len(all_uids)))
        print(f"Selected {len(selected_uids)} objects to download")

        # Load annotations for selected UIDs
        def load_annotations():
            return objaverse.load_annotations(selected_uids)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            annotations = await loop.run_in_executor(pool, load_annotations)

        # Download using objaverse library (handles sharded folder structure)
        print(f"Downloading {len(selected_uids)} GLB files...")

        objects = {}

        # Download in batches for progress updates
        BATCH_SIZE = 5

        for i in range(0, len(selected_uids), BATCH_SIZE):
            if _status.cancelled:
                raise asyncio.CancelledError("Cancelled by user")

            batch_uids = selected_uids[i:i + BATCH_SIZE]

            # Download batch in thread
            def download_batch(uids):
                return objaverse.load_objects(uids, download_processes=4)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                paths = await loop.run_in_executor(pool, download_batch, batch_uids)

            # Add to objects dict
            for uid, path in paths.items():
                ann = annotations.get(uid, {})
                objects[uid] = {
                    "local_path": path,
                    "metadata": {
                        "name": ann.get("name", uid[:20]),
                        "categories": ann.get("categories", []),
                        "tags": ann.get("tags", [])
                    }
                }

            _status.downloaded = len(objects)

            # Stream progress
            if progress_callback:
                await progress_callback({
                    "type": "dataset_progress",
                    "step": "downloading",
                    "message": f"Downloaded {len(objects)}/{count}",
                    "total": count,
                    "downloaded": len(objects),
                    "indexed": 0,
                    "failed": _status.failed
                })

        print(f"Download complete: {len(objects)}/{count} objects")

        if progress_callback:
            await progress_callback({
                "type": "dataset_progress",
                "step": "indexing",
                "message": f"Starting to index {len(objects)} models...",
                "total": count,
                "downloaded": len(objects),
                "indexed": 0,
                "failed": 0
            })

        # Import here to avoid circular imports
        from faiss_index import FAISSIndex, EMBEDDING_DIM_GEMMA, EMBEDDING_DIM_SIGLIP
        from local_renderer import render_views

        # Determine embedding mode
        use_ollama = EMBEDDING_MODE == "ollama"
        use_local = os.getenv("USE_LOCAL_RENDERER", "false").lower() == "true"

        if use_ollama:
            from ollama_client import process_3d_model, description_to_text
            embedding_dim = EMBEDDING_DIM_GEMMA
            print(f"Using Ollama mode (Gemma 3 27B + EmbeddingGemma, {embedding_dim}-dim)")
        else:
            from runpod_client import embed_images
            embedding_dim = EMBEDDING_DIM_SIGLIP
            print(f"Using RunPod mode (SigLIP2, {embedding_dim}-dim)")

        # Create fresh index with correct dimension
        index = FAISSIndex(embedding_dim=embedding_dim)

        # Process in batches for efficiency
        from ollama_client import process_batch

        object_items = list(objects.items())
        total_objects = len(object_items)
        print(f"Starting to index {total_objects} models in batches of {BATCH_SIZE}...")

        for batch_start in range(0, total_objects, BATCH_SIZE):
            if _status.cancelled:
                print("Indexing cancelled by user")
                raise asyncio.CancelledError("Cancelled by user")

            batch_end = min(batch_start + BATCH_SIZE, total_objects)
            batch_items = object_items[batch_start:batch_end]

            print(f"\n=== Batch {batch_start//BATCH_SIZE + 1}: models {batch_start+1}-{batch_end} ===")

            # Step 1: Render all images in parallel
            render_args = []
            for file_id, obj_data in batch_items:
                local_path = obj_data["local_path"]
                metadata = obj_data.get("metadata", {})
                name = metadata.get("name", file_id[:30])
                category = metadata.get("source", None)
                render_args.append((file_id, local_path, name, category))

            print(f"  Rendering {len(render_args)} models with {RENDER_WORKERS} workers...")
            render_start = time.time()

            batch_data = []
            with ProcessPoolExecutor(max_workers=RENDER_WORKERS) as executor:
                results = list(executor.map(render_single_model, render_args))

            for result in results:
                if "error" in result:
                    _status.failed += 1
                    print(f"  Render failed: {result['file_id']}: {result['error']}")
                else:
                    batch_data.append(result)

            render_time = time.time() - render_start
            print(f"  Rendered {len(batch_data)} models in {render_time:.1f}s ({render_time/len(batch_data):.2f}s/model)")

            if not batch_data:
                continue

            # Step 2: Send batch to RunPod
            print(f"  Sending {len(batch_data)} images to RunPod...")

            batch_request = [{"image": item["image_b64"]} for item in batch_data]
            result = await process_batch(batch_request)

            if result["status"] != "ok":
                print(f"  Batch error: {result.get('error', 'unknown')}")
                _status.failed += len(batch_data)
                continue

            batch_results = result.get("results", [])
            stats = result.get("stats", {})
            print(f"  Done: {stats.get('time_sec', 0):.2f}s total, {stats.get('time_per_model', 0):.3f}s/model")

            # Step 3: Add to index
            for i, item in enumerate(batch_data):
                if i < len(batch_results):
                    embed_result = batch_results[i]
                    embedding = embed_result.get("embedding", [])
                    text = embed_result.get("text", "")

                    if embedding:
                        index.add(
                            embedding=embedding,
                            model_id=item["file_id"],
                            name=item["name"],
                            category=item["category"],
                            file_path=item["local_path"],
                            save=False
                        )
                        _status.indexed += 1

                        if progress_callback:
                            # Use the already-saved preview thumbnail
                            preview_path = DATASET_DIR / "previews" / f"{item['file_id']}.jpg"
                            thumb_b64 = ""
                            if preview_path.exists():
                                thumb_b64 = base64.b64encode(preview_path.read_bytes()).decode()
                            await progress_callback({
                                "type": "dataset_progress",
                                "step": "indexing",
                                "message": f"{text[:40]}...",
                                "total": count,
                                "downloaded": total_objects,
                                "indexed": _status.indexed,
                                "failed": _status.failed,
                                "current": item["name"],
                                "model_id": item["file_id"],
                                "images": [thumb_b64] if thumb_b64 else []
                            })
                    else:
                        _status.failed += 1
                else:
                    _status.failed += 1

        # Save index
        index._save()

        _status.is_generating = False
        _status.current_model = None

        result = {
            "status": "completed",
            "total_requested": count,
            "downloaded": _status.downloaded,
            "indexed": _status.indexed,
            "failed": _status.failed,
            "duration_seconds": None  # TODO: calculate
        }

        if progress_callback:
            await progress_callback({
                "type": "dataset_complete",
                **result
            })

        return result

    except asyncio.CancelledError:
        print("Generation cancelled, cleaning up...")
        _status.is_generating = False
        _status.error = "Cancelled"
        # Clean up on cancel
        await clear_dataset()
        raise

    except Exception as e:
        print(f"Generation error: {e}, cleaning up...")
        _status.is_generating = False
        _status.error = str(e)
        # Clean up on error
        await clear_dataset()
        raise


# Synchronous wrapper for CLI usage
def generate_dataset_sync(count: int = DATASET_SIZE) -> Dict:
    """Synchronous wrapper for generate_dataset."""
    return asyncio.run(generate_dataset(count))


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else DATASET_SIZE
    print(f"Generating dataset with {count} models...")

    result = generate_dataset_sync(count)
    print(f"Done! {result}")
