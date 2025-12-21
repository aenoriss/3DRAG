"""RunPod modules for 3D model processing pipeline.

Modules are imported lazily to avoid triggering pyrender/pyglet
display initialization before PYOPENGL_PLATFORM is set.
"""

__all__ = [
    "download_models",
    "get_annotations",
    "render_model",
    "render_models_batch",
    "MAX_RENDER_WORKERS",
    "caption_image",
    "caption_images_batch",
    "load_florence",
    "embed_text",
    "embed_texts_batch",
    "load_model",
]
