"""
RunPod Serverless Handler - SigLIP2 Embedding Service

Supports:
- Image embedding (base64 images from local rendering)
- Text embedding
- 3D model rendering + embedding (optional, requires pyrender)
"""

import runpod
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import base64
import io
import numpy as np
import time

# Pricing ($/second) - update as needed
GPU_COST_PER_SECOND = 0.00019  # L4/A5000/3090

# ============================================================================
# MODEL LOADING (cold start) - GPU Optimized
# ============================================================================

print("Loading SigLIP2 model with GPU optimizations...")

# Load model with FP16 for faster inference
model = AutoModel.from_pretrained(
    "google/siglip2-so400m-patch14-384",
    torch_dtype=torch.float16
).to("cuda")
processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
model.eval()

# Try to compile model for faster inference (PyTorch 2.0+)
try:
    model = torch.compile(model, mode="reduce-overhead")
    print("Model compiled with torch.compile()")
except Exception as e:
    print(f"torch.compile not available: {e}")

# Warmup inference to optimize CUDA kernels
print("Warming up CUDA kernels...")
with torch.inference_mode():
    dummy_input = processor(
        images=[Image.new("RGB", (384, 384))],
        return_tensors="pt"
    ).to("cuda", dtype=torch.float16)
    _ = model.get_image_features(**dummy_input)
    torch.cuda.synchronize()

print(f"Model loaded on {torch.cuda.get_device_name(0)}!")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# EMBEDDING FUNCTIONS - GPU Optimized
# ============================================================================

@torch.inference_mode()
def embed_images_pil(images: list[Image.Image]) -> np.ndarray:
    """Embed PIL images and return normalized embeddings (GPU optimized)."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to("cuda", dtype=torch.float16) if v.dtype == torch.float32 else v.to("cuda")
              for k, v in inputs.items()}

    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


@torch.inference_mode()
def embed_images_b64(images_b64: list[str]) -> np.ndarray:
    """Embed base64-encoded images."""
    images = []
    for img_b64 in images_b64:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)
    return embed_images_pil(images)


@torch.inference_mode()
def embed_text(texts: list[str]) -> np.ndarray:
    """Embed text strings (GPU optimized)."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to("cuda")

    emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


# ============================================================================
# 3D RENDERING (optional - lazy loaded)
# ============================================================================

_rendering_available = None

def _check_rendering():
    """Check if 3D rendering is available."""
    global _rendering_available
    if _rendering_available is None:
        try:
            import os
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender
            import trimesh
            _rendering_available = True
            print("3D rendering available (pyrender + trimesh)")
        except Exception as e:
            _rendering_available = False
            print(f"3D rendering not available: {e}")
    return _rendering_available


def render_and_embed_model(model_bytes: bytes, file_format: str = "glb") -> dict:
    """Render 3D model and return embedding. Requires pyrender."""
    import os
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import pyrender
    import trimesh

    CAMERA_POSITIONS = [
        (0, 0), (0, 45), (0, 90), (0, 135),
        (0, 180), (0, 225), (0, 270), (0, 315),
        (90, 0), (-90, 0),
        (45, 0), (45, 180),
    ]
    RENDER_SIZE = 384
    BACKGROUND_COLOR = [0.5, 0.5, 0.5, 1.0]

    def create_camera_pose(elevation_deg, azimuth_deg, distance):
        elevation = np.radians(elevation_deg)
        azimuth = np.radians(azimuth_deg)
        x = distance * np.cos(elevation) * np.sin(azimuth)
        y = distance * np.sin(elevation)
        z = distance * np.cos(elevation) * np.cos(azimuth)
        camera_pos = np.array([x, y, z])
        forward = -camera_pos / np.linalg.norm(camera_pos)
        if abs(elevation_deg) > 89:
            up = np.array([0, 0, -1 if elevation_deg > 0 else 1])
        else:
            up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        return pose

    # Load mesh
    mesh = trimesh.load(io.BytesIO(model_bytes), file_type=file_format.lower(), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if meshes else None
        if mesh is None:
            raise ValueError("No valid meshes found")

    # Normalize
    mesh.vertices -= mesh.centroid
    mesh.vertices *= 1.0 / np.max(np.abs(mesh.vertices))

    # Create scene
    scene = pyrender.Scene(bg_color=np.array(BACKGROUND_COLOR), ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # Lighting
    for intensity, rot in [(3.0, [[0.707,0,0.707],[0.354,0.866,-0.354],[-0.612,0.5,0.612]]),
                           (1.5, [[-0.707,0,0.707],[-0.183,0.966,-0.183],[-0.683,-0.259,-0.683]]),
                           (2.0, [[-1,0,0],[0,0.866,0.5],[0,-0.5,0.866]])]:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        pose = np.eye(4)
        pose[:3, :3] = np.array(rot)
        scene.add(light, pose=pose)

    # Render views
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_node = scene.add(camera)
    renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)

    images = []
    try:
        for elevation, azimuth in CAMERA_POSITIONS:
            pose = create_camera_pose(elevation, azimuth, 2.5)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)
            images.append(Image.fromarray(color))
    finally:
        renderer.delete()

    # Embed with max pooling (keeps strongest features, ignores bad views)
    embeddings = embed_images_pil(images)
    max_embedding = np.max(embeddings, axis=0)
    max_embedding = max_embedding / np.linalg.norm(max_embedding)

    return {
        "embedding": max_embedding.tolist(),
        "views_rendered": len(images),
        "embedding_dim": len(max_embedding)
    }


# ============================================================================
# MAIN HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"images": ["<base64>", ...]}   → batch image embeddings (for local rendering)
    - {"image": "<base64>"}           → single image embedding
    - {"text": "a red chair"}         → single text embedding
    - {"texts": ["chair", "table"]}   → batch text embeddings
    - {"model": "<base64>", "format": "glb"}  → render + embed 3D model
    - {"stats": true}                 → return system stats
    """
    try:
        input_data = event.get("input", {})
        include_stats = input_data.get("include_stats", False)

        # Stats endpoint
        if input_data.get("stats"):
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            return {
                "status": "ready",
                "gpu": gpu_name,
                "gpu_memory_gb": round(gpu_memory, 1),
                "model": "siglip2-so400m-patch14-384",
                "embedding_dim": 1152,
                "rendering_available": _check_rendering()
            }

        # Batch images (primary use case for local rendering)
        if "images" in input_data:
            start = time.time()
            embeddings = embed_images_b64(input_data["images"])
            elapsed = time.time() - start

            result = {"embeddings": embeddings.tolist()}
            if include_stats:
                result["stats"] = {
                    "time_sec": round(elapsed, 4),
                    "images_count": len(input_data["images"]),
                    "cost_usd": round(elapsed * GPU_COST_PER_SECOND, 8)
                }
            return result

        # Single image
        if "image" in input_data:
            start = time.time()
            embeddings = embed_images_b64([input_data["image"]])
            elapsed = time.time() - start

            result = {"embedding": embeddings[0].tolist()}
            if include_stats:
                result["stats"] = {"time_sec": round(elapsed, 4)}
            return result

        # Single text
        if "text" in input_data:
            start = time.time()
            embeddings = embed_text([input_data["text"]])
            elapsed = time.time() - start

            result = {"embedding": embeddings[0].tolist()}
            if include_stats:
                result["stats"] = {"time_sec": round(elapsed, 4)}
            return result

        # Batch texts
        if "texts" in input_data:
            start = time.time()
            embeddings = embed_text(input_data["texts"])
            elapsed = time.time() - start

            result = {"embeddings": embeddings.tolist()}
            if include_stats:
                result["stats"] = {"time_sec": round(elapsed, 4), "texts_count": len(input_data["texts"])}
            return result

        # 3D Model (optional - requires pyrender)
        if "model" in input_data:
            if not _check_rendering():
                return {"error": "3D rendering not available. Use local rendering mode."}

            start = time.time()
            model_b64 = input_data["model"]
            file_format = input_data.get("format", "glb")
            model_bytes = base64.b64decode(model_b64)

            result = render_and_embed_model(model_bytes, file_format)
            elapsed = time.time() - start

            if include_stats:
                result["stats"] = {
                    "time_sec": round(elapsed, 4),
                    "cost_usd": round(elapsed * GPU_COST_PER_SECOND, 8)
                }
            return result

        return {"error": "No valid input. Use 'images', 'image', 'text', 'texts', 'model', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
