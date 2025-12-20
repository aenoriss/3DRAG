"""
GPU-accelerated 3D model renderer.

Uses EGL for headless GPU rendering on RunPod.
Supports parallel rendering with ThreadPoolExecutor.
"""

import os
import numpy as np
import trimesh
import io
import base64
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

# Use EGL for GPU rendering (RunPod has NVIDIA GPU)
os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyrender

# Render settings
RENDER_SIZE = 384
BACKGROUND_COLOR = [0.5, 0.5, 0.5, 1.0]

# Parallel rendering settings - aggressive for 80GB+ GPU
# Each EGL context uses ~200-500MB, so 80GB can handle 32+ workers easily
MAX_RENDER_WORKERS = int(os.getenv("RENDER_WORKERS", "32"))


def create_camera_pose(elevation_deg: float, azimuth_deg: float, distance: float) -> np.ndarray:
    """Create a 4x4 camera pose matrix looking at the origin."""
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)

    x = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.sin(elevation)
    z = distance * np.cos(elevation) * np.cos(azimuth)

    camera_pos = np.array([x, y, z])
    forward = -camera_pos / np.linalg.norm(camera_pos)

    world_up = np.array([0, 1, 0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 0, 1])
        right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_pos

    return pose


def render_model(
    model_path: str,
    num_views: int = 1,
    return_pil: bool = False
) -> tuple[list, list]:
    """
    Render views of a 3D model using GPU.

    Args:
        model_path: Path to the 3D model file
        num_views: Number of views to render (1 = front only)
        return_pil: If True, return PIL Images; else return base64 strings

    Returns:
        Tuple of (list of PIL Images or None, list of base64 encoded PNGs)
    """
    # Camera positions (elevation, azimuth)
    CAMERA_POSITIONS = [
        (0, 0),    # Front
        (0, 90),   # Side
        (0, 180),  # Back
        (0, 270),  # Other side
        (45, 45),  # Angled top
    ]

    # Load mesh
    mesh_path = Path(model_path)
    mesh = trimesh.load(str(mesh_path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("No valid mesh found in scene")

    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = 1.0 / max(mesh.extents)
    mesh.vertices *= scale

    # Create pyrender scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(bg_color=BACKGROUND_COLOR)
    scene.add(mesh_pyrender)

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_node = scene.add(camera)

    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=create_camera_pose(45, 45, 2))

    ambient = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(ambient, pose=create_camera_pose(-30, -135, 2))

    # Calculate camera distance
    distance = 2.5

    # Render views
    renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)
    images = []
    images_b64 = []

    try:
        positions = CAMERA_POSITIONS[:num_views]
        for elevation, azimuth in positions:
            pose = create_camera_pose(elevation, azimuth, distance)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)

            img = Image.fromarray(color)
            if return_pil:
                images.append(img)

            # Encode as base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_b64.append(base64.b64encode(buffer.getvalue()).decode())
    finally:
        renderer.delete()

    return images, images_b64


def render_model_bytes(
    model_bytes: bytes,
    file_extension: str = "glb",
    num_views: int = 1
) -> tuple[list, list]:
    """
    Render views from model bytes (for models already in memory).

    Args:
        model_bytes: Raw bytes of the 3D model file
        file_extension: File format (glb, obj, stl, etc.)
        num_views: Number of views to render

    Returns:
        Tuple of (list of PIL Images, list of base64 encoded PNGs)
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as f:
        f.write(model_bytes)
        temp_path = f.name

    try:
        return render_model(temp_path, num_views=num_views, return_pil=True)
    finally:
        os.unlink(temp_path)


def _render_single(args: Tuple[str, str, int]) -> Dict[str, Any]:
    """
    Render a single model (worker function for parallel rendering).

    Args:
        args: (uid, model_path, num_views)

    Returns:
        Dict with uid, images, images_b64, or error
    """
    uid, model_path, num_views = args

    try:
        images, images_b64 = render_model(model_path, num_views=num_views, return_pil=True)
        return {
            "uid": uid,
            "images": images,
            "images_b64": images_b64,
            "success": True
        }
    except Exception as e:
        return {
            "uid": uid,
            "error": str(e),
            "success": False
        }


def render_models_batch(
    models: List[Tuple[str, str]],
    num_views: int = 1,
    max_workers: int = MAX_RENDER_WORKERS
) -> List[Dict[str, Any]]:
    """
    Render multiple models in parallel using GPU.

    Each worker creates its own EGL context for thread-safe rendering.

    Args:
        models: List of (uid, model_path) tuples
        num_views: Number of views per model
        max_workers: Maximum parallel render threads

    Returns:
        List of results with uid, images, images_b64, or error
    """
    if not models:
        return []

    # Prepare args for workers
    render_args = [(uid, path, num_views) for uid, path in models]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_render_single, args): args[0] for args in render_args}

        for future in as_completed(futures):
            uid = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "uid": uid,
                    "error": str(e),
                    "success": False
                })

    # Sort by original order
    uid_order = {uid: i for i, (uid, _) in enumerate(models)}
    results.sort(key=lambda r: uid_order.get(r["uid"], 999))

    return results
