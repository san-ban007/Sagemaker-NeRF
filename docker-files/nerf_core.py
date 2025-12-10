# /opt/program/nerf_core.py

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from nerfstudio.utils import eval_utils


def load_pipeline(model_dir: str):
    """
    Load a trained Nerfstudio pipeline from the model directory.

    SageMaker will untar your model under /opt/ml/model.
    We search recursively for a config.yml there and let Nerfstudio
    set up the pipeline in 'inference' mode.
    """
    model_root = Path(model_dir)

    config_paths = list(model_root.rglob("config.yml"))
    if not config_paths:
        raise FileNotFoundError(f"No config.yml found under {model_root}")
    config_path = config_paths[0]
    print(f"[nerf_core.load_pipeline] Using config: {config_path}")

    # ----- IMPORTANT: set project_root so relative 'outputs/...' paths work -----
    project_root = None
    for parent in config_path.parents:
        if parent.name == "nerfstudio":
            project_root = parent
            break
    if project_root is None:
        # Fallback heuristic: go up 4 levels: .../nerfstudio/outputs/...
        project_root = config_path.parents[4]

    print(f"[nerf_core.load_pipeline] Using project_root: {project_root}")
    os.chdir(project_root)
    print(f"[nerf_core.load_pipeline] CWD is now: {Path.cwd()}")

    # Call eval_setup; different Nerfstudio versions return tuples of different shapes.
    results = eval_utils.eval_setup(config_path, test_mode="inference")
    print(f"[nerf_core.load_pipeline] eval_setup returned type: {type(results)}")

    # Find the pipeline object inside the result
    pipeline = None
    if isinstance(results, tuple):
        for item in results:
            if hasattr(item, "model") and hasattr(item, "datamanager"):
                pipeline = item
                break
        if pipeline is None:
            raise RuntimeError(
                f"Could not find a pipeline-like object in eval_setup results: {[type(x) for x in results]}"
            )
    else:
        # Some versions may directly return the pipeline
        if hasattr(results, "model") and hasattr(results, "datamanager"):
            pipeline = results
        else:
            raise RuntimeError(
                f"eval_setup returned non-pipeline object of type {type(results)}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    # stash device for later
    pipeline.device = device

    print(f"[nerf_core.load_pipeline] Pipeline type: {type(pipeline)}")
    print(f"[nerf_core.load_pipeline] Using device: {device}")
    return pipeline


@torch.no_grad()
def render_camera(pipeline, camera_index: int = 0) -> np.ndarray:
    """
    Render a single evaluation camera and return an RGB uint8 image [H, W, 3].

    This version:
    - Uses datamanager.eval_dataset.cameras[camera_index]
    - Uses model.get_outputs_for_camera(camera=...) when available
    - Falls back to camera.generate_rays() + model(ray_bundle)
    """

    device = getattr(
        pipeline,
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    dm = pipeline.datamanager

    # 1) Get eval_dataset
    dataset = getattr(dm, "eval_dataset", None)
    if dataset is None:
        raise RuntimeError("datamanager has no 'eval_dataset'; cannot render.")

    # 2) Use dataset.cameras directly
    if not hasattr(dataset, "cameras"):
        raise RuntimeError(
            "eval_dataset has no 'cameras' attribute; cannot obtain camera poses."
        )

    cameras = dataset.cameras
    try:
        num_cams = len(cameras)
    except TypeError:
        num_cams = getattr(cameras, "num_cameras", "unknown")

    print(f"[render_camera] Requested camera_index={camera_index}, num_cameras={num_cams}")

    if isinstance(num_cams, int):
        if camera_index < 0 or camera_index >= num_cams:
            raise IndexError(f"camera_index {camera_index} out of range [0, {num_cams-1}]")

    # Move cameras to device if possible
    if hasattr(cameras, "to"):
        cameras = cameras.to(device)

    cam = cameras[camera_index]
    print(f"[render_camera] Using Cameras[{camera_index}]")

    model = pipeline.model

    if hasattr(model, "get_outputs_for_camera"):
        print("[render_camera] Using model.get_outputs_for_camera(camera=...)")
        outputs = model.get_outputs_for_camera(camera=cam)
    else:
        print("[render_camera] Falling back to camera.generate_rays() + model(ray_bundle)")
        if not hasattr(cam, "generate_rays"):
            raise RuntimeError(
                "Camera object has no 'generate_rays' method and model has no 'get_outputs_for_camera'."
            )
        ray_bundle = cam.generate_rays(device=device)
        outputs = model(ray_bundle)

    if "rgb" not in outputs:
        raise KeyError(f"Model outputs do not contain 'rgb' key. Keys: {list(outputs.keys())}")

    rgb = outputs["rgb"].detach().cpu().numpy()  # [H, W, 3] float32 in [0, 1]
    img_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return img_u8


def render_camera_png_bytes(pipeline, camera_index: int = 0) -> bytes:
    """
    Convenience: render and return PNG bytes.
    """
    img_u8 = render_camera(pipeline, camera_index)
    pil_img = Image.fromarray(img_u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
