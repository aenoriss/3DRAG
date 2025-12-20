"""
Local Renderer - CPU-based 3D rendering for testing without RunPod.

Uses OSMesa (software rendering) so no GPU required.
"""
from __future__ import annotations

import numpy as np
import trimesh
import io
import base64
from PIL import Image
from typing import Optional, List, Tuple
import os

# Set OSMesa for CPU rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pyrender

# ============================================================================
# CAMERA CONFIGURATION (same as RunPod handler)
# ============================================================================

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
BACKGROUND_COLOR = [0.5, 0.5, 0.5, 1.0]


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


def render_views(
    mesh_bytes: bytes,
    file_extension: str = "glb"
) -> tuple[list[Image.Image], list[str]]:
    """
    Render 12 views of a 3D model.

    Args:
        mesh_bytes: Raw bytes of the 3D model file
        file_extension: File format (glb, obj, stl, ply, etc.)

    Returns:
        Tuple of (list of PIL Images, list of base64 encoded PNGs)
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
    images_b64 = []

    try:
        for elevation, azimuth in CAMERA_POSITIONS:
            pose = create_camera_pose(elevation, azimuth, distance)
            scene.set_pose(camera_node, pose)
            color, _ = renderer.render(scene)

            img = Image.fromarray(color)
            images.append(img)

            # Also encode as base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            images_b64.append(img_b64)
    finally:
        renderer.delete()

    return images, images_b64


def render_model_bytes(
    model_bytes: bytes,
    file_format: str = "glb"
) -> dict:
    """
    Render a 3D model and return views + mock embedding.

    This is a drop-in replacement for the RunPod API for local testing.
    Returns a random embedding since we don't have SigLIP2 locally.

    Args:
        model_bytes: Raw bytes of the 3D model
        file_format: File format

    Returns:
        dict with 'images_b64', 'embedding' (random), 'views_rendered'
    """
    images, images_b64 = render_views(model_bytes, file_format)

    # Generate random normalized embedding (for testing only)
    # In production, this would be from SigLIP2
    embedding = np.random.randn(1152).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    return {
        "images_b64": images_b64,
        "embedding": embedding.tolist(),
        "views_rendered": len(images),
        "embedding_dim": 1152,
        "note": "Using random embedding (local testing mode)"
    }


if __name__ == "__main__":
    # Test with a sample file
    import sys

    if len(sys.argv) < 2:
        print("Usage: python local_renderer.py <path_to_3d_model>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Rendering 12 views of: {model_path}")

    with open(model_path, "rb") as f:
        model_bytes = f.read()

    ext = model_path.rsplit(".", 1)[-1]
    images, _ = render_views(model_bytes, ext)

    # Save test renders
    for i, img in enumerate(images):
        img.save(f"view_{i:02d}.png")
        print(f"Saved view_{i:02d}.png")

    print("Done!")
