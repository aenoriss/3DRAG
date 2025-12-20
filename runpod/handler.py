"""
RunPod Serverless Handler - Combined 3D Rendering + SigLIP2 Embedding

Accepts 3D models, renders 12 views, embeds with SigLIP2, returns averaged embedding.
"""

import runpod
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import base64
import io
import numpy as np
import trimesh
import time

# Set EGL for GPU rendering
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

# Pricing ($/second) - update as needed
GPU_COST_PER_SECOND = 0.00019  # L4/A5000/3090

# ============================================================================
# MODEL LOADING (cold start) - GPU Optimized
# ============================================================================

print("Loading SigLIP2 model with GPU optimizations...")

# Load model with FP16 for faster inference
model = AutoModel.from_pretrained(
    "google/siglip2-so400m-patch14-384",
    torch_dtype=torch.float16,  # Use FP16 for faster inference
    device_map="cuda"
)
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
# CAMERA CONFIGURATION
# ============================================================================

# 12 camera positions: (elevation_deg, azimuth_deg)
CAMERA_POSITIONS = [
    # 8 around equator (0° elevation, every 45° azimuth)
    (0, 0), (0, 45), (0, 90), (0, 135),
    (0, 180), (0, 225), (0, 270), (0, 315),
    # Top and bottom
    (90, 0), (-90, 0),
    # Angled views (45° elevation)
    (45, 0), (45, 180),
]

RENDER_SIZE = 384  # SigLIP2 input size
BACKGROUND_COLOR = [0.5, 0.5, 0.5, 1.0]  # Neutral gray

# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def create_camera_pose(elevation_deg: float, azimuth_deg: float, distance: float) -> np.ndarray:
    """Create a 4x4 camera pose matrix looking at the origin."""
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


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center and scale mesh to fit in a unit sphere."""
    mesh.vertices -= mesh.centroid
    scale = 1.0 / np.max(np.abs(mesh.vertices))
    mesh.vertices *= scale
    return mesh


def create_scene(mesh: trimesh.Trimesh) -> pyrender.Scene:
    """Create a pyrender scene with mesh and 3-point lighting."""
    scene = pyrender.Scene(
        bg_color=np.array(BACKGROUND_COLOR),
        ambient_light=[0.3, 0.3, 0.3]
    )

    # Convert trimesh to pyrender mesh (preserves textures/colors)
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(py_mesh)

    # 3-point lighting
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    key_pose = np.eye(4)
    key_pose[:3, :3] = np.array([
        [0.707, 0, 0.707],
        [0.354, 0.866, -0.354],
        [-0.612, 0.5, 0.612]
    ])
    scene.add(key_light, pose=key_pose)

    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = np.array([
        [-0.707, 0, 0.707],
        [-0.183, 0.966, -0.183],
        [-0.683, -0.259, -0.683]
    ])
    scene.add(fill_light, pose=fill_pose)

    back_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    back_pose = np.eye(4)
    back_pose[:3, :3] = np.array([
        [-1, 0, 0],
        [0, 0.866, 0.5],
        [0, -0.5, 0.866]
    ])
    scene.add(back_light, pose=back_pose)

    return scene


def render_views(mesh_bytes: bytes, file_extension: str = "glb") -> list[Image.Image]:
    """
    Render 12 views of a 3D model.

    Args:
        mesh_bytes: Raw bytes of the 3D model file
        file_extension: File format (glb, obj, stl, ply, etc.)

    Returns:
        List of 12 PIL Images
    """
    # Load mesh
    mesh = trimesh.load(
        io.BytesIO(mesh_bytes),
        file_type=file_extension.lower(),
        force='mesh'
    )

    # Handle scene files (GLB with multiple meshes)
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for geometry in mesh.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid meshes found in file")

    # Normalize mesh
    mesh = normalize_mesh(mesh)

    # Create scene
    scene = create_scene(mesh)

    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_node = scene.add(camera)

    distance = 2.5
    renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)

    images = []
    try:
        for elevation, azimuth in CAMERA_POSITIONS:
            pose = create_camera_pose(elevation, azimuth, distance)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)
            images.append(Image.fromarray(color))
    finally:
        renderer.delete()

    return images


# ============================================================================
# EMBEDDING FUNCTIONS - GPU Optimized
# ============================================================================

@torch.inference_mode()
def embed_images_pil(images: list[Image.Image]) -> np.ndarray:
    """Embed PIL images and return normalized embeddings (GPU optimized)."""
    inputs = processor(images=images, return_tensors="pt")
    # Move to GPU with FP16
    inputs = {k: v.to("cuda", dtype=torch.float16) if v.dtype == torch.float32 else v.to("cuda")
              for k, v in inputs.items()}

    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()  # Convert back to FP32 for output


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
# MAIN HANDLER
# ============================================================================

def handler(event):
    """
    RunPod serverless handler.

    Input formats:
    - {"model": "<base64>", "format": "glb"}  → render + embed 3D model, return averaged embedding
    - {"text": "a red chair"}                  → single text embedding
    - {"texts": ["chair", "table"]}            → batch text embeddings
    - {"image": "<base64>"}                    → single image embedding
    - {"images": ["<base64>", ...]}            → batch image embeddings
    - {"stats": true}                          → return system stats and GPU info
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
                "render_resolution": RENDER_SIZE,
                "views_per_model": len(CAMERA_POSITIONS),
                "pricing": {
                    "cost_per_second": GPU_COST_PER_SECOND,
                    "estimated_cost_per_model": round(GPU_COST_PER_SECOND * 1.0, 6),
                    "estimated_models_per_dollar": int(1 / (GPU_COST_PER_SECOND * 1.0))
                }
            }

        # 3D Model: render + embed + average
        if "model" in input_data:
            total_start = time.time()

            model_b64 = input_data["model"]
            file_format = input_data.get("format", "glb")

            # Decode model
            decode_start = time.time()
            model_bytes = base64.b64decode(model_b64)
            decode_time = time.time() - decode_start

            # Render 12 views
            render_start = time.time()
            images = render_views(model_bytes, file_format)
            render_time = time.time() - render_start

            # Embed all views
            embed_start = time.time()
            embeddings = embed_images_pil(images)
            embed_time = time.time() - embed_start

            # Average embeddings
            avg_start = time.time()
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Re-normalize
            avg_time = time.time() - avg_start

            total_time = time.time() - total_start
            cost = total_time * GPU_COST_PER_SECOND

            result = {
                "embedding": avg_embedding.tolist(),
                "views_rendered": len(images),
                "embedding_dim": len(avg_embedding)
            }

            if include_stats:
                result["stats"] = {
                    "total_time_sec": round(total_time, 4),
                    "decode_time_sec": round(decode_time, 4),
                    "render_time_sec": round(render_time, 4),
                    "embed_time_sec": round(embed_time, 4),
                    "average_time_sec": round(avg_time, 6),
                    "cost_usd": round(cost, 8),
                    "throughput_models_per_hour": int(3600 / total_time) if total_time > 0 else 0
                }

            return result

        # Single text
        if "text" in input_data:
            start = time.time()
            embeddings = embed_text([input_data["text"]])
            elapsed = time.time() - start

            result = {"embedding": embeddings[0].tolist()}
            if include_stats:
                result["stats"] = {
                    "time_sec": round(elapsed, 4),
                    "cost_usd": round(elapsed * GPU_COST_PER_SECOND, 8)
                }
            return result

        # Batch texts
        if "texts" in input_data:
            start = time.time()
            embeddings = embed_text(input_data["texts"])
            elapsed = time.time() - start

            result = {"embeddings": embeddings.tolist()}
            if include_stats:
                result["stats"] = {
                    "time_sec": round(elapsed, 4),
                    "texts_count": len(input_data["texts"]),
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
                result["stats"] = {
                    "time_sec": round(elapsed, 4),
                    "cost_usd": round(elapsed * GPU_COST_PER_SECOND, 8)
                }
            return result

        # Batch images
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

        return {"error": "No valid input. Use 'model', 'text', 'texts', 'image', 'images', or 'stats'."}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Start the serverless worker
runpod.serverless.start({"handler": handler})
