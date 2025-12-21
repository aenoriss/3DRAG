"""
RunPod Serverless Handler - Full 3D Model Processing Pipeline

Handles the complete pipeline on GPU:
1. Download models from Objaverse
2. Render views with GPU (EGL)
3. Caption with Florence-2
4. Embed with sentence-transformers (all-mpnet-base-v2)

Input formats:
- {"uids": ["uid1", "uid2", ...]} -> Process specific models
- {"text": "query"}              -> Embed search query
- {"stats": true}                -> Return system stats
"""

# Start Xvfb for pyglet before any imports
import os
import subprocess
import time as time_module

# Start virtual X server
xvfb_process = subprocess.Popen(
    ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time_module.sleep(1)  # Wait for Xvfb to start
os.environ["DISPLAY"] = ":99"
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Use EGL for GPU rendering

import runpod
import time

# Stats tracking
STATS = {
    "total_requests": 0,
    "total_models": 0,
    "total_text_queries": 0,
    "total_time_sec": 0.0,
    "started_at": time.time()
}

GPU_COST_PER_SEC = float(os.getenv("GPU_COST_PER_SEC", "0.00019"))


def startup():
    """Initialize all models and services."""
    print("=== Starting RunPod Worker ===")

    # Load embedding model (GPU)
    from modules.embedder import load_model
    load_model()

    # Load Florence-2 (GPU)
    from modules.captioner import load_florence
    load_florence()

    print("=== Worker Ready ===")


# Initialize on import
startup()


INTERNAL_BATCH_SIZE = 1000  # Process all in one pass


def process_models(uids: list[str]) -> list[dict]:
    """
    Process a list of model UIDs with internal batching.

    Downloads, renders, captions, and embeds in batches of INTERNAL_BATCH_SIZE.

    Args:
        uids: List of Objaverse UIDs (can be thousands)

    Returns:
        List of results with embeddings
    """
    from modules.downloader import download_models, get_annotations
    from modules.renderer import render_models_batch, MAX_RENDER_WORKERS
    from modules.captioner import caption_images_batch
    from modules.embedder import embed_texts_batch

    all_results = []
    total = len(uids)

    print(f"[handler] Processing {total} models in batches of {INTERNAL_BATCH_SIZE}...")

    for batch_start in range(0, total, INTERNAL_BATCH_SIZE):
        batch_end = min(batch_start + INTERNAL_BATCH_SIZE, total)
        batch_uids = uids[batch_start:batch_end]
        batch_num = batch_start // INTERNAL_BATCH_SIZE + 1
        total_batches = (total + INTERNAL_BATCH_SIZE - 1) // INTERNAL_BATCH_SIZE

        print(f"\n=== Batch {batch_num}/{total_batches}: {len(batch_uids)} models ===")

        # Download batch
        print(f"  Downloading {len(batch_uids)} models...")
        paths = download_models(batch_uids, download_processes=4)

        # Load annotations
        annotations = get_annotations(batch_uids)

        # Render batch
        print(f"  Rendering {len(paths)} models ({MAX_RENDER_WORKERS} workers)...")
        models_to_render = [(uid, path) for uid, path in paths.items()]
        render_results = render_models_batch(models_to_render, num_views=1, max_workers=MAX_RENDER_WORKERS)

        # Process render results
        render_data = []
        for result in render_results:
            uid = result["uid"]
            if result["success"]:
                ann = annotations.get(uid, {})
                render_data.append({
                    "uid": uid,
                    "name": ann.get("name", uid[:20]),
                    "image": result["images"][0] if result["images"] else None,
                    "image_b64": result["images_b64"][0] if result["images_b64"] else None
                })
            else:
                print(f"    Render failed: {uid[:20]}...")

            # Clean up downloaded file
            if uid in paths:
                try:
                    os.unlink(paths[uid])
                except Exception:
                    pass

        if not render_data:
            print(f"  No successful renders in batch {batch_num}")
            continue

        # Caption batch
        print(f"  Captioning {len(render_data)} images...")
        images = [d["image"] for d in render_data if d["image"]]
        captions = caption_images_batch(images)

        # Embed batch
        print(f"  Embedding {len(captions)} captions...")
        embeddings = embed_texts_batch(captions)

        # Build batch results
        batch_results = []
        for i, data in enumerate(render_data):
            if i < len(embeddings) and embeddings[i]:
                batch_results.append({
                    "uid": data["uid"],
                    "name": data["name"],
                    "caption": captions[i] if i < len(captions) else "",
                    "embedding": embeddings[i],
                    "preview": data["image_b64"]
                })

        all_results.extend(batch_results)
        print(f"  Batch {batch_num} complete: {len(batch_results)} models processed")

    print(f"\n[handler] Total: {len(all_results)}/{total} models processed successfully")
    return all_results


def handler(event):
    """RunPod serverless handler."""
    try:
        input_data = event.get("input", {})
        print(f"[handler] Received request with keys: {list(input_data.keys())}")

        # Stats endpoint
        if input_data.get("stats"):
            print("[handler] Processing stats request")
            uptime = time.time() - STATS["started_at"]
            avg_time = STATS["total_time_sec"] / STATS["total_requests"] if STATS["total_requests"] > 0 else 0
            estimated_cost = STATS["total_time_sec"] * GPU_COST_PER_SEC
            cost_per_model = estimated_cost / STATS["total_models"] if STATS["total_models"] > 0 else 0

            return {
                "status": "ready",
                "cumulative": {
                    "total_requests": STATS["total_requests"],
                    "total_models": STATS["total_models"],
                    "total_text_queries": STATS["total_text_queries"],
                    "total_time_sec": round(STATS["total_time_sec"], 3),
                    "avg_time_sec": round(avg_time, 3),
                    "uptime_sec": round(uptime, 1),
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "cost_per_model_usd": round(cost_per_model, 6),
                    "gpu_cost_per_sec": GPU_COST_PER_SEC
                }
            }

        # Process specific UIDs
        if "uids" in input_data:
            start = time.time()
            uids = input_data["uids"]
            print(f"[handler] Processing {len(uids)} UIDs...")

            results = process_models(uids)
            print(f"[handler] Processed {len(results)} models successfully")

            elapsed = time.time() - start
            STATS["total_requests"] += 1
            STATS["total_models"] += len(results)
            STATS["total_time_sec"] += elapsed

            return {
                "results": results,
                "processed": len(results),
                "requested": len(uids),
                "time_sec": round(elapsed, 3),
                "time_per_model": round(elapsed / len(results), 3) if results else 0
            }

        # Text query embedding
        if "text" in input_data:
            print(f"[handler] Embedding text query: {input_data['text'][:50]}...")
            start = time.time()
            from modules.embedder import embed_text
            embedding = embed_text(input_data["text"])
            print(f"[handler] Text embedded, dim={len(embedding)}")
            elapsed = time.time() - start

            STATS["total_requests"] += 1
            STATS["total_text_queries"] += 1
            STATS["total_time_sec"] += elapsed

            return {
                "embedding": embedding,
                "dimension": len(embedding),
                "time_sec": round(elapsed, 3)
            }

        return {"error": "No valid input. Use 'uids', 'text', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start worker
runpod.serverless.start({"handler": handler})
