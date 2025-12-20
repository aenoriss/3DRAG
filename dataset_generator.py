"""
Dataset Generator - Download and index models from Objaverse.

Creates a mini-dataset of 100 models for testing.
When regenerated, the previous dataset is deleted.
Downloads GLB files directly from Hugging Face for efficiency.
"""
from __future__ import annotations

import objaverse
import asyncio
import shutil
import random
import time
import base64
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from datetime import datetime

# Dataset storage
DATASET_DIR = Path("dataset")
DATASET_SIZE = 100


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


async def clear_dataset():
    """Delete existing dataset and clear FAISS index."""
    from faiss_index import get_index

    # Clear FAISS index files
    index_path = Path("models.index")
    metadata_path = Path("metadata.json")

    if index_path.exists():
        index_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    # Clear dataset directory
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    DATASET_DIR.mkdir(exist_ok=True)

    # Reinitialize index (will create fresh)
    # Note: This requires restarting the server or reinitializing the singleton


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
        from faiss_index import FAISSIndex
        from runpod_client import embed_images
        from local_renderer import render_views
        import os

        # Check if we're in local mode
        use_local = os.getenv("USE_LOCAL_RENDERER", "false").lower() == "true"

        # Create fresh index
        index = FAISSIndex()

        # Process each model
        print(f"Starting to index {len(objects)} models (use_local={use_local})...")

        for idx, (file_id, obj_data) in enumerate(objects.items()):
            # Check for cancellation
            if _status.cancelled:
                print("Indexing cancelled by user")
                raise asyncio.CancelledError("Cancelled by user")

            _status.current_model = file_id

            try:
                local_path = obj_data["local_path"]
                metadata = obj_data.get("metadata", {})

                # Get name from metadata or use file_id
                name = metadata.get("name", file_id[:30])
                category = metadata.get("source", None)

                print(f"[{idx+1}/{len(objects)}] Processing: {name[:30]}...")

                # Read model file
                model_path = Path(local_path)
                if not model_path.exists():
                    print(f"  File not found: {local_path}")
                    _status.failed += 1
                    continue

                model_bytes = model_path.read_bytes()
                ext = model_path.suffix.lstrip(".")
                print(f"  Read {len(model_bytes)} bytes, ext={ext}")

                rendered_images_b64 = None

                if use_local:
                    # Local rendering + RunPod embedding
                    import numpy as np

                    print(f"  Rendering 12 views...")
                    images, images_b64 = render_views(model_bytes, ext)
                    rendered_images_b64 = images_b64  # Save for progress update
                    print(f"  Rendered {len(images)} views, sending to RunPod...")
                    embed_result = await embed_images(images_b64)
                    print(f"  Got embeddings from RunPod")

                    embeddings = np.array(embed_result["embeddings"], dtype=np.float32)
                    avg_embedding = np.mean(embeddings, axis=0)
                    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                    embedding = avg_embedding.tolist()
                else:
                    # Full RunPod processing
                    from runpod_client import embed_model_bytes
                    result = await embed_model_bytes(model_bytes, ext)
                    embedding = result["embedding"]

                # Add to index
                index.add(
                    embedding=embedding,
                    model_id=file_id,
                    name=name,
                    category=category,
                    file_path=str(local_path),
                    save=False  # Save at end for efficiency
                )

                _status.indexed += 1

                if progress_callback:
                    progress_data = {
                        "type": "dataset_progress",
                        "step": "indexing",
                        "message": f"Indexed {name[:30]}",
                        "total": count,
                        "downloaded": len(objects),
                        "indexed": _status.indexed,
                        "failed": _status.failed,
                        "current": name,
                        "model_id": file_id
                    }
                    # Include small thumbnails (4 views, JPEG compressed)
                    if rendered_images_b64:
                        from PIL import Image
                        import io
                        thumbnails = []
                        # Send 4 views: front, side, top, angled (indices 0, 2, 8, 10)
                        for i in [0, 2, 8, 10]:
                            if i < len(images):
                                # Resize to 96x96 thumbnail and save as JPEG
                                thumb = images[i].resize((96, 96), Image.LANCZOS)
                                buffer = io.BytesIO()
                                thumb.save(buffer, format="JPEG", quality=70)
                                thumbnails.append(base64.b64encode(buffer.getvalue()).decode())
                        progress_data["images"] = thumbnails
                    await progress_callback(progress_data)

            except Exception as e:
                _status.failed += 1
                print(f"Failed to process {file_id}: {e}")

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
