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


INTERNAL_BATCH_SIZE = 250  # Smaller batches to prevent disk full


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

        # Clean up temp/cache files before each batch to prevent disk full
        import tempfile
        import glob
        import shutil
        # Clean temp dir
        for ext in ["*.glb", "*.obj", "*.stl", "*.gltf"]:
            for tmp in glob.glob(os.path.join(tempfile.gettempdir(), ext)):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
        # Clean objaverse cache
        objaverse_cache = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
        if os.path.exists(objaverse_cache):
            try:
                shutil.rmtree(objaverse_cache)
            except Exception:
                pass

        # Timing tracker
        timings = {}
        batch_start_time = time.time()

        # Download batch
        t0 = time.time()
        print(f"  Downloading {len(batch_uids)} models...", flush=True)
        paths = download_models(batch_uids, download_processes=4)
        timings["download"] = time.time() - t0

        # Load annotations
        t0 = time.time()
        annotations = get_annotations(batch_uids)
        timings["annotations"] = time.time() - t0

        # Render batch
        t0 = time.time()
        print(f"  Rendering {len(paths)} models ({MAX_RENDER_WORKERS} workers)...", flush=True)
        models_to_render = [(uid, path) for uid, path in paths.items()]
        render_results = render_models_batch(models_to_render, num_views=1, max_workers=MAX_RENDER_WORKERS)
        timings["render"] = time.time() - t0

        # Process render results
        render_data = []
        failed_count = 0
        error_types = {}
        sample_errors = []

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
                failed_count += 1
                error_type = result.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
                # Keep first 3 sample errors for debugging
                if len(sample_errors) < 3:
                    sample_errors.append(f"{uid[:12]}: {result.get('error', 'unknown')[:50]}")

            # Clean up downloaded file
            if uid in paths:
                try:
                    os.unlink(paths[uid])
                except Exception:
                    pass

        # Log error summary
        if error_types:
            print(f"  Render errors by type: {error_types}", flush=True)
            for err in sample_errors:
                print(f"    Sample: {err}", flush=True)

        if not render_data:
            print(f"  No successful renders in batch {batch_num}")
            continue

        # Caption batch
        t0 = time.time()
        print(f"  Captioning {len(render_data)} images...", flush=True)
        images = [d["image"] for d in render_data if d["image"]]
        captions = caption_images_batch(images)
        timings["caption"] = time.time() - t0

        # Free PIL images from memory (keep only base64 for response)
        del images
        for d in render_data:
            d["image"] = None

        # Embed batch
        t0 = time.time()
        print(f"  Embedding {len(captions)} captions...", flush=True)
        embeddings = embed_texts_batch(captions)
        timings["embed"] = time.time() - t0

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

        # Print timing summary
        batch_total = time.time() - batch_start_time
        print(f"\n  === Batch {batch_num} Timing ===", flush=True)
        print(f"  Download:    {timings['download']:6.1f}s", flush=True)
        print(f"  Annotations: {timings['annotations']:6.1f}s", flush=True)
        print(f"  Render:      {timings['render']:6.1f}s ({len(render_data)} ok, {failed_count} failed)", flush=True)
        print(f"  Caption:     {timings['caption']:6.1f}s", flush=True)
        print(f"  Embed:       {timings['embed']:6.1f}s", flush=True)
        print(f"  TOTAL:       {batch_total:6.1f}s ({len(batch_results)} models)", flush=True)

        # Aggressive cleanup after each batch
        import gc
        del render_results, render_data, captions, embeddings, batch_results
        gc.collect()

        # Clean disk: temp files + objaverse cache
        for ext in ["*.glb", "*.obj", "*.stl", "*.gltf", "*.png", "*.jpg"]:
            for tmp in glob.glob(os.path.join(tempfile.gettempdir(), ext)):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
        objaverse_cache = os.path.expanduser("~/.objaverse")
        if os.path.exists(objaverse_cache):
            try:
                shutil.rmtree(objaverse_cache)
                print(f"  Cleaned objaverse cache", flush=True)
            except Exception:
                pass

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

        # Process single model from bytes (for file uploads)
        if "model_bytes" in input_data:
            import base64
            start = time.time()

            model_b64 = input_data["model_bytes"]
            file_ext = input_data.get("file_extension", "glb")
            model_name = input_data.get("name", "uploaded_model")

            print(f"[handler] Processing uploaded model: {model_name}.{file_ext}")

            try:
                # Decode base64
                model_bytes = base64.b64decode(model_b64)

                # Render
                from modules.renderer import render_model_bytes
                images, images_b64 = render_model_bytes(model_bytes, file_ext, num_views=1)

                if not images:
                    return {"error": "Render failed - no images produced"}

                # Caption
                from modules.captioner import caption_images_batch
                captions = caption_images_batch(images)
                caption = captions[0] if captions else ""

                # Embed
                from modules.embedder import embed_texts_batch
                embeddings = embed_texts_batch([caption])
                embedding = embeddings[0] if embeddings else None

                if not embedding:
                    return {"error": "Embedding failed"}

                elapsed = time.time() - start
                STATS["total_requests"] += 1
                STATS["total_models"] += 1
                STATS["total_time_sec"] += elapsed

                return {
                    "name": model_name,
                    "caption": caption,
                    "embedding": embedding,
                    "preview": images_b64[0] if images_b64 else None,
                    "time_sec": round(elapsed, 3)
                }

            except Exception as e:
                return {"error": f"Processing failed: {str(e)}"}

        # Process batch of models from bytes (for folder uploads)
        if "models_batch" in input_data:
            import base64
            import gc
            start = time.time()

            models = input_data["models_batch"]
            print(f"[handler] Processing batch of {len(models)} uploaded models...")

            from modules.renderer import render_models_bytes_batch, MAX_RENDER_WORKERS
            from modules.captioner import caption_images_batch
            from modules.embedder import embed_texts_batch

            all_results = []
            batch_size = 50  # Process in sub-batches for memory efficiency

            for batch_start in range(0, len(models), batch_size):
                batch_end = min(batch_start + batch_size, len(models))
                batch = models[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(models) + batch_size - 1) // batch_size

                print(f"  Sub-batch {batch_num}/{total_batches}: {len(batch)} models", flush=True)

                # Prepare models for parallel rendering
                render_input = []
                model_names = {}  # model_id -> name mapping
                for model in batch:
                    model_id = model.get("model_id", "unknown")
                    model_names[model_id] = model.get("name", "uploaded")
                    try:
                        model_bytes = base64.b64decode(model["bytes_b64"])
                        render_input.append({
                            "model_id": model_id,
                            "bytes": model_bytes,
                            "extension": model.get("extension", "glb")
                        })
                    except Exception as e:
                        all_results.append({
                            "model_id": model_id,
                            "name": model_names[model_id],
                            "error": f"Decode failed: {str(e)}"
                        })

                if not render_input:
                    continue

                # Parallel render all models in batch (uses multiprocessing + 4-view stitching)
                t0 = time.time()
                render_results = render_models_bytes_batch(
                    render_input,
                    num_views=1,  # Ignored when STITCH_VIEWS=True
                    max_workers=MAX_RENDER_WORKERS
                )
                print(f"    Render: {time.time() - t0:.1f}s ({len(render_results)} models)", flush=True)

                # Collect successful renders
                render_data = []
                for result in render_results:
                    model_id = result["model_id"]
                    if result.get("success") and result.get("images"):
                        render_data.append({
                            "model_id": model_id,
                            "name": model_names.get(model_id, "uploaded"),
                            "image": result["images"][0],
                            "image_b64": result["images_b64"][0] if result.get("images_b64") else None
                        })
                    else:
                        all_results.append({
                            "model_id": model_id,
                            "name": model_names.get(model_id, "uploaded"),
                            "error": result.get("error", "Render failed")
                        })

                if not render_data:
                    continue

                # Caption batch (GPU - runs on main process)
                t0 = time.time()
                images = [d["image"] for d in render_data]
                captions = caption_images_batch(images)
                print(f"    Caption: {time.time() - t0:.1f}s ({len(captions)} images)", flush=True)
                del images

                # Embed batch (GPU - runs on main process)
                t0 = time.time()
                embeddings = embed_texts_batch(captions)
                print(f"    Embed: {time.time() - t0:.1f}s ({len(embeddings)} captions)", flush=True)

                # Build results
                for i, data in enumerate(render_data):
                    if i < len(embeddings) and embeddings[i]:
                        all_results.append({
                            "model_id": data["model_id"],
                            "name": data["name"],
                            "caption": captions[i] if i < len(captions) else "",
                            "embedding": embeddings[i],
                            "preview": data["image_b64"]
                        })
                    else:
                        all_results.append({
                            "model_id": data["model_id"],
                            "name": data["name"],
                            "error": "Embedding failed"
                        })

                # Cleanup
                del render_input, render_results, render_data, captions, embeddings
                gc.collect()

            elapsed = time.time() - start
            successful = len([r for r in all_results if "embedding" in r])

            STATS["total_requests"] += 1
            STATS["total_models"] += successful
            STATS["total_time_sec"] += elapsed

            print(f"[handler] Batch complete: {successful}/{len(models)} successful in {elapsed:.1f}s")

            return {
                "results": all_results,
                "processed": successful,
                "requested": len(models),
                "time_sec": round(elapsed, 3),
                "time_per_model": round(elapsed / successful, 3) if successful else 0
            }

        return {"error": "No valid input. Use 'uids', 'text', 'model_bytes', 'models_batch', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start worker
runpod.serverless.start({"handler": handler})
