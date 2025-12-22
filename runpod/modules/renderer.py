"""
GPU-accelerated 3D model renderer.

Uses EGL for headless GPU rendering on RunPod.
Supports parallel rendering with multiprocessing (NOT threading).

IMPORTANT: pyrender uses OpenGL which cannot share contexts across threads.
We use multiprocessing.Pool and import pyrender inside each worker process.
"""

import os
import numpy as np
import trimesh
import io
import base64
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, get_context
from typing import List, Tuple, Dict, Any

# Note: pyrender is imported inside worker functions, not here
# This is critical for multiprocessing to work correctly

# Debug: Check OpenGL platform (at import time)
print(f"[renderer] PYOPENGL_PLATFORM = {os.environ.get('PYOPENGL_PLATFORM', 'not set')}", flush=True)
print(f"[renderer] DISPLAY = {os.environ.get('DISPLAY', 'not set')}", flush=True)


def _check_egl():
    """Check EGL availability (called once per process)."""
    try:
        import OpenGL.EGL as egl
        display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        if display != egl.EGL_NO_DISPLAY:
            major, minor = egl.EGLint(), egl.EGLint()
            if egl.eglInitialize(display, major, minor):
                print(f"[renderer] EGL initialized: {major.value}.{minor.value}", flush=True)
                egl.eglTerminate(display)
                return True
        print(f"[renderer] EGL_NO_DISPLAY", flush=True)
    except Exception as e:
        print(f"[renderer] EGL check failed: {e}", flush=True)
    return False

# Render settings
RENDER_SIZE = 384
BACKGROUND_COLOR = [0.5, 0.5, 0.5, 1.0]

# Stitched view settings (3x2 grid of 6 views)
STITCH_VIEWS = True  # Enable 6-view stitching by default
STITCH_GRID_COLS = 3  # 3 columns
STITCH_GRID_ROWS = 2  # 2 rows
STITCH_VIEW_SIZE = RENDER_SIZE  # Each view in the grid

# Parallel rendering settings
# Bottleneck is CPU (mesh loading/processing), not GPU
# Match physical core count for optimal throughput
MAX_RENDER_WORKERS = int(os.getenv("RENDER_WORKERS", "16"))


def stitch_views(images: List[Image.Image], cols: int = 3, rows: int = 2) -> Image.Image:
    """
    Stitch multiple view images into a grid.

    Args:
        images: List of PIL Images to stitch (expects 6 for 3x2 grid)
        cols: Number of columns in grid
        rows: Number of rows in grid

    Returns:
        Single PIL Image with all views stitched together
    """
    if not images:
        raise ValueError("No images to stitch")

    # Ensure we have exactly cols * rows images
    expected = cols * rows
    if len(images) < expected:
        # Duplicate last image to fill grid
        while len(images) < expected:
            images.append(images[-1].copy())
    images = images[:expected]

    # Get dimensions from first image
    w, h = images[0].size

    # Create output image
    output = Image.new('RGB', (w * cols, h * rows))

    # Paste images in grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        output.paste(img.convert('RGB'), (col * w, row * h))

    return output


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
    return_pil: bool = False,
    stitch: bool = None
) -> tuple[list, list]:
    """
    Render views of a 3D model using GPU.

    IMPORTANT: This function imports pyrender locally to support multiprocessing.
    Each process needs its own OpenGL context.

    Args:
        model_path: Path to the 3D model file
        num_views: Number of views to render (1 = front only, ignored if stitch=True)
        return_pil: If True, return PIL Images; else return base64 strings
        stitch: If True, render 4 views and stitch into 2x2 grid (default: STITCH_VIEWS)

    Returns:
        Tuple of (list of PIL Images or None, list of base64 encoded PNGs)
        If stitch=True, returns single stitched image in each list
    """
    # Import pyrender inside function for multiprocessing compatibility
    import pyrender

    # Use global default if not specified
    if stitch is None:
        stitch = STITCH_VIEWS

    # Camera positions (elevation, azimuth) - 6 views for complete coverage
    # Layout in 3x2 grid: [front, right, back] / [left, top, bottom]
    CAMERA_POSITIONS = [
        (0, 0),     # Front
        (0, 90),    # Right side
        (0, 180),   # Back
        (0, 270),   # Left side
        (90, 0),    # Top (looking down)
        (-90, 0),   # Bottom (looking up)
    ]

    # For stitched mode, always render 6 views
    if stitch:
        num_views = 6

    # Load mesh
    mesh_path = Path(model_path)
    if not mesh_path.exists():
        raise ValueError(f"File not found: {model_path}")

    mesh = trimesh.load(str(mesh_path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No geometry in scene")
        mesh = trimesh.util.concatenate(meshes)

    # Validate mesh has triangles
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        raise ValueError("Empty mesh (no vertices)")
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        raise ValueError("Empty mesh (no faces)")

    # Center and normalize
    centroid = mesh.centroid
    if np.isnan(centroid).any():
        raise ValueError("Invalid geometry (NaN centroid)")
    mesh.vertices -= centroid

    extents = mesh.extents
    if max(extents) == 0:
        raise ValueError("Zero-size mesh")
    scale = 1.0 / max(extents)
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

    # Calculate camera distance (smaller = more zoomed in)
    distance = 2.0  # Slightly zoomed in from 2.5

    renderer = pyrender.OffscreenRenderer(RENDER_SIZE, RENDER_SIZE)

    raw_images = []  # Individual view images
    images = []      # Final output (stitched or individual)
    images_b64 = []

    try:
        positions = CAMERA_POSITIONS[:num_views]
        for elevation, azimuth in positions:
            pose = create_camera_pose(elevation, azimuth, distance)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)
            raw_images.append(Image.fromarray(color))

        # Stitch views into 3x2 grid if enabled
        if stitch and len(raw_images) >= 6:
            stitched = stitch_views(raw_images[:6], cols=STITCH_GRID_COLS, rows=STITCH_GRID_ROWS)
            if return_pil:
                images.append(stitched)

            # Encode stitched image as base64 JPEG
            buffer = io.BytesIO()
            stitched.convert("RGB").save(buffer, format="JPEG", quality=85)
            images_b64.append(base64.b64encode(buffer.getvalue()).decode())
        else:
            # Return individual images
            for img in raw_images:
                if return_pil:
                    images.append(img)

                buffer = io.BytesIO()
                img.convert("RGB").save(buffer, format="JPEG", quality=85)
                images_b64.append(base64.b64encode(buffer.getvalue()).decode())
    finally:
        renderer.delete()

    return images, images_b64


def render_model_bytes(
    model_bytes: bytes,
    file_extension: str = "glb",
    num_views: int = 1,
    stitch: bool = None
) -> tuple[list, list]:
    """
    Render views from model bytes (for models already in memory).

    Args:
        model_bytes: Raw bytes of the 3D model file
        file_extension: File format (glb, obj, stl, etc.)
        num_views: Number of views to render (ignored if stitch=True)
        stitch: If True, render 4 views and stitch into 2x2 grid

    Returns:
        Tuple of (list of PIL Images, list of base64 encoded PNGs)
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as f:
        f.write(model_bytes)
        temp_path = f.name

    try:
        return render_model(temp_path, num_views=num_views, return_pil=True, stitch=stitch)
    finally:
        os.unlink(temp_path)


def _render_bytes_single(args: Tuple[str, bytes, str, int]) -> Dict[str, Any]:
    """
    Render a single model from bytes (worker function for parallel rendering).

    Writes bytes to temp file, renders, then cleans up.
    Called in separate process via multiprocessing.Pool.

    Args:
        args: (model_id, model_bytes, file_extension, num_views)

    Returns:
        Dict with model_id, images_b64, or error
    """
    import tempfile

    model_id, model_bytes, file_ext, num_views = args

    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as f:
            f.write(model_bytes)
            temp_path = f.name

        try:
            # Render (pyrender imported inside render_model)
            _, images_b64 = render_model(temp_path, num_views=num_views, return_pil=False)
            return {
                "model_id": model_id,
                "images_b64": images_b64,
                "success": True
            }
        finally:
            os.unlink(temp_path)

    except Exception as e:
        return {
            "model_id": model_id,
            "error": str(e),
            "success": False
        }


def render_models_bytes_batch(
    models: List[Dict[str, Any]],
    num_views: int = 1,
    max_workers: int = MAX_RENDER_WORKERS
) -> List[Dict[str, Any]]:
    """
    Render multiple models from bytes in parallel using multiprocessing.

    Used for batch file uploads from the frontend.

    Args:
        models: List of dicts with keys:
            - model_id: Unique identifier
            - bytes: Raw model bytes (already decoded from base64)
            - extension: File format (glb, obj, etc.)
        num_views: Number of views per model
        max_workers: Maximum parallel render processes

    Returns:
        List of results with model_id, images, images_b64, or error
    """
    if not models:
        return []

    # Prepare args for workers
    render_args = [
        (m["model_id"], m["bytes"], m.get("extension", "glb"), num_views)
        for m in models
    ]

    print(f"[renderer] Parallel rendering {len(models)} uploaded models with {max_workers} processes", flush=True)

    # Use 'spawn' context for clean process isolation
    ctx = get_context('spawn')

    results = []
    with ctx.Pool(processes=max_workers, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_render_bytes_single, render_args):
            results.append(result)

    # Sort by original order
    id_order = {m["model_id"]: i for i, m in enumerate(models)}
    results.sort(key=lambda r: id_order.get(r["model_id"], 999))

    # Convert base64 back to PIL for compatibility with captioning
    for result in results:
        if result.get("success") and result.get("images_b64"):
            images = []
            for b64 in result["images_b64"]:
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
            result["images"] = images

    return results


def _render_single(args: Tuple[str, str, int]) -> Dict[str, Any]:
    """
    Render a single model (worker function for parallel rendering).

    This function is called in a separate process via multiprocessing.Pool.
    pyrender is imported inside render_model() for OpenGL context isolation.

    Args:
        args: (uid, model_path, num_views)

    Returns:
        Dict with uid, images_b64 (no PIL images - not serializable across processes)
    """
    uid, model_path, num_views = args

    try:
        # return_pil=False because PIL Images can't be pickled across processes
        _, images_b64 = render_model(model_path, num_views=num_views, return_pil=False)
        return {
            "uid": uid,
            "images_b64": images_b64,
            "success": True
        }
    except Exception as e:
        error_msg = str(e)
        # Categorize error for analysis
        if "no faces" in error_msg.lower() or "no vertices" in error_msg.lower():
            error_type = "empty_mesh"
        elif "nan" in error_msg.lower():
            error_type = "invalid_geometry"
        elif "zero-size" in error_msg.lower():
            error_type = "zero_size"
        elif "not found" in error_msg.lower():
            error_type = "file_not_found"
        elif "no geometry" in error_msg.lower():
            error_type = "no_geometry"
        else:
            error_type = "other"

        return {
            "uid": uid,
            "error": error_msg,
            "error_type": error_type,
            "success": False
        }


def _init_worker():
    """Initialize worker process with correct environment."""
    # Ensure EGL environment is set in each worker
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # DISPLAY should be inherited from parent


def render_models_batch(
    models: List[Tuple[str, str]],
    num_views: int = 1,
    max_workers: int = MAX_RENDER_WORKERS
) -> List[Dict[str, Any]]:
    """
    Render multiple models in parallel using GPU with multiprocessing.

    Uses multiprocessing.Pool (not threading) because OpenGL contexts
    cannot be shared across threads. Each process imports pyrender fresh
    and creates its own OpenGL/EGL context.

    Args:
        models: List of (uid, model_path) tuples
        num_views: Number of views per model
        max_workers: Maximum parallel render processes

    Returns:
        List of results with uid, images_b64, or error
        Note: PIL images are converted to base64 in workers (not serializable)
    """
    if not models:
        return []

    # Prepare args for workers
    render_args = [(uid, path, num_views) for uid, path in models]

    print(f"[renderer] Parallel rendering {len(models)} models with {max_workers} processes", flush=True)

    # Use 'spawn' context for clean process isolation (required for OpenGL)
    ctx = get_context('spawn')

    results = []
    with ctx.Pool(processes=max_workers, initializer=_init_worker) as pool:
        # Use imap_unordered for better throughput
        for result in pool.imap_unordered(_render_single, render_args):
            results.append(result)

    # Sort by original order
    uid_order = {uid: i for i, (uid, _) in enumerate(models)}
    results.sort(key=lambda r: uid_order.get(r["uid"], 999))

    # Convert base64 back to PIL for compatibility with existing code
    for result in results:
        if result.get("success") and result.get("images_b64"):
            images = []
            for b64 in result["images_b64"]:
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
            result["images"] = images

    return results
