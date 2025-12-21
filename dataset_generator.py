"""
Dataset Generator - Download and index models from Objaverse.

Creates a mini-dataset of 100 models for testing.
When regenerated, the previous dataset is deleted.

All heavy processing (download, render, caption, embed) happens on RunPod GPU.
"""
from __future__ import annotations

import objaverse
import asyncio
import shutil
import random
import base64
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from datetime import datetime

# Dataset storage
DATASET_DIR = Path("dataset")

DATASET_SIZE = 100
# All UIDs sent in one request - RunPod handles internal batching


@dataclass
class DatasetStatus:
    """Current dataset generation status."""
    is_generating: bool = False
    cancelled: bool = False
    total: int = 0
    processed: int = 0
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
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    DATASET_DIR.mkdir(exist_ok=True)

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
    Generate a new dataset from Objaverse.

    All processing happens on RunPod GPU:
    1. Download models from Objaverse
    2. Render views with GPU (EGL)
    3. Caption with Florence-2
    4. Embed with EmbeddingGemma

    Args:
        count: Number of models to process (default 100)
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
                "processed": 0,
                "indexed": 0,
                "failed": 0
            })

        await clear_dataset()

        # Load UIDs from Objaverse
        loop = asyncio.get_event_loop()

        if progress_callback:
            await progress_callback({
                "type": "dataset_progress",
                "step": "loading",
                "message": "Loading Objaverse UIDs...",
                "total": count,
                "processed": 0,
                "indexed": 0,
                "failed": 0
            })

        def load_uids():
            return objaverse.load_uids()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            all_uids = await loop.run_in_executor(pool, load_uids)

        print(f"Loaded {len(all_uids)} UIDs from Objaverse")

        # Sample random UIDs
        selected_uids = random.sample(list(all_uids), min(count, len(all_uids)))
        print(f"Selected {len(selected_uids)} objects to process")

        # Import FAISS index
        from faiss_index import FAISSIndex, EMBEDDING_DIM_GEMMA
        from runpod_client import process_uids

        # Create fresh index
        index = FAISSIndex(embedding_dim=EMBEDDING_DIM_GEMMA)

        # Create previews directory
        previews_dir = DATASET_DIR / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        # Send all UIDs to RunPod in one request (RunPod handles internal batching)
        total_uids = len(selected_uids)
        print(f"Sending {total_uids} models to RunPod...")

        if progress_callback:
            await progress_callback({
                "type": "dataset_progress",
                "step": "processing",
                "message": f"Processing {total_uids} models on RunPod...",
                "total": count,
                "processed": 0,
                "indexed": 0,
                "failed": 0
            })

        try:
            # Send all UIDs to RunPod - it handles batching internally
            result = await process_uids(selected_uids)

            results = result.get("results", [])
            print(f"Received {len(results)} results from RunPod")
            print(f"Time: {result.get('time_sec', 0):.1f}s, {result.get('time_per_model', 0):.2f}s/model")

            for item in results:
                uid = item.get("uid", "")
                name = item.get("name", uid[:20])
                caption = item.get("caption", "")
                embedding = item.get("embedding", [])
                preview_b64 = item.get("preview", "")

                _status.processed += 1

                if not embedding:
                    _status.failed += 1
                    print(f"  No embedding for {uid}")
                    continue

                # Save preview image as jpg
                if preview_b64:
                    try:
                        from PIL import Image
                        import io
                        preview_bytes = base64.b64decode(preview_b64)
                        img = Image.open(io.BytesIO(preview_bytes))
                        preview_path = previews_dir / f"{uid}.jpg"
                        img.convert("RGB").save(preview_path, "JPEG", quality=85)
                    except Exception as e:
                        print(f"  Failed to save preview for {uid}: {e}")

                # Add to index
                index.add(
                    embedding=embedding,
                    model_id=uid,
                    name=name,
                    category=None,
                    file_path=None,
                    caption=caption,
                    save=False
                )
                _status.indexed += 1

                if progress_callback:
                    await progress_callback({
                        "type": "dataset_progress",
                        "step": "indexing",
                        "message": f"{caption[:40]}..." if caption else name,
                        "total": count,
                        "processed": _status.processed,
                        "indexed": _status.indexed,
                        "failed": _status.failed,
                        "current": name,
                        "model_id": uid
                    })

        except Exception as e:
            print(f"RunPod error: {e}")
            _status.failed += total_uids

        # Save index
        index._save()

        _status.is_generating = False
        _status.current_model = None

        result = {
            "status": "completed",
            "total_requested": count,
            "processed": _status.processed,
            "indexed": _status.indexed,
            "failed": _status.failed
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
        await clear_dataset()
        raise

    except Exception as e:
        print(f"Generation error: {e}, cleaning up...")
        _status.is_generating = False
        _status.error = str(e)
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
