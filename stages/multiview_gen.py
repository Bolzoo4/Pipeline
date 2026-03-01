"""
Multi-View Generation — Generate 6 consistent views from a single image.

Uses Zero123++ v1.2 to generate 6 views at known camera angles.
If real multi-view photos are provided, this stage is skipped.

Camera poses (Zero123++ convention):
  View 0: azimuth=30°,  elevation=30°
  View 1: azimuth=90°,  elevation=-20°
  View 2: azimuth=150°, elevation=30°
  View 3: azimuth=210°, elevation=-20°
  View 4: azimuth=270°, elevation=30°
  View 5: azimuth=330°, elevation=-20°
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional

# Known camera poses for Zero123++ v1.2 (6 views)
CAMERA_AZIMUTHS = [30, 90, 150, 210, 270, 330]
CAMERA_ELEVATIONS = [30, -20, 30, -20, 30, -20]
NUM_VIEWS = 6


def generate_multiview(image: np.ndarray, mock: bool = False) -> list[np.ndarray]:
    """
    Generate 6 views from a single input image.

    Args:
        image: Input image (BGR, HxWx3)
        mock: If True, generate synthetic rotated views for testing

    Returns:
        List of 6 view images (RGB, 320x320x3 each)
    """
    if mock:
        return _mock_multiview(image)
    else:
        return _real_multiview(image)


def _mock_multiview(image: np.ndarray) -> list[np.ndarray]:
    """Generate mock multi-view by applying synthetic transforms."""
    h, w = image.shape[:2]
    size = 320
    views = []

    # Resize to square
    img_square = cv2.resize(image, (size, size))
    img_rgb = cv2.cvtColor(img_square, cv2.COLOR_BGR2RGB)

    for i in range(NUM_VIEWS):
        angle = CAMERA_AZIMUTHS[i]
        # Simulate rotation with affine transform
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle - 30, 1.0)
        # Add slight perspective effect based on elevation
        elev = CAMERA_ELEVATIONS[i]
        M[0, 2] += elev * 0.3
        M[1, 2] += elev * 0.5

        rotated = cv2.warpAffine(img_rgb, M, (size, size),
                                  borderMode=cv2.BORDER_REFLECT)
        views.append(rotated)

    return views


def _real_multiview(image: np.ndarray) -> list[np.ndarray]:
    """Generate multi-view using Zero123++ v1.2."""
    try:
        import torch
        from diffusers import DiffusionPipeline
        from PIL import Image as PILImage
    except ImportError:
        raise RuntimeError(
            "diffusers/torch not installed. Use --mock for local testing."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("   ℹ Loading Zero123++ v1.2...")
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=dtype,
    ).to(device)

    # Remove background first for better results
    print("   ℹ Removing background...")
    img_nobg = _remove_background(image)

    # Convert to PIL
    img_rgb = cv2.cvtColor(img_nobg, cv2.COLOR_BGR2RGB)
    pil_input = PILImage.fromarray(img_rgb)

    # Generate 6 views (output is a 3x2 grid, 960x640 or similar)
    print("   ℹ Generating 6 views...")
    result = pipe(
        pil_input,
        num_inference_steps=75,
    ).images[0]

    # Split the grid into individual views
    views = _split_grid(np.array(result))

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return views


def _remove_background(image: np.ndarray) -> np.ndarray:
    """Remove background using rembg."""
    try:
        from rembg import remove
        from PIL import Image as PILImage

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        result = remove(pil_img)

        # Convert back to BGR with white background
        result_np = np.array(result)
        if result_np.shape[2] == 4:
            alpha = result_np[:, :, 3:4].astype(np.float32) / 255.0
            rgb = result_np[:, :, :3].astype(np.float32)
            white_bg = np.ones_like(rgb) * 255
            composited = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            return cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    except ImportError:
        print("   ⚠ rembg not installed, skipping background removal")
        return image


def _split_grid(grid: np.ndarray) -> list[np.ndarray]:
    """
    Split a Zero123++ 3x2 grid image into 6 individual views.
    Grid layout: 3 columns × 2 rows
    """
    h, w = grid.shape[:2]
    rows, cols = 3, 2  # Zero123++ outputs 3 cols × 2 rows
    cell_w = w // cols
    cell_h = h // rows

    views = []
    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            view = grid[y1:y2, x1:x2]
            # Resize to standard 320x320
            view = cv2.resize(view, (320, 320), interpolation=cv2.INTER_LANCZOS4)
            views.append(view)

    return views[:NUM_VIEWS]


def load_real_multiview(image_paths: list[str]) -> list[np.ndarray]:
    """
    Load real multi-view photos provided by merchant.
    Expects 4-8 photos taken at roughly equal angular intervals.
    """
    views = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load: {path}")
        # Convert to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (320, 320), interpolation=cv2.INTER_LANCZOS4)
        views.append(img_resized)

    return views


def get_camera_matrices(num_views: int = 6,
                         distance: float = 2.5,
                         image_size: int = 320) -> list[dict]:
    """
    Get camera projection matrices for the known Zero123++ poses.
    Returns list of dicts with 'extrinsic', 'intrinsic', 'azimuth', 'elevation'.
    """
    cameras = []

    # Simple perspective intrinsic
    focal = image_size * 1.2
    cx, cy = image_size / 2, image_size / 2
    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    for i in range(min(num_views, NUM_VIEWS)):
        az = np.radians(CAMERA_AZIMUTHS[i])
        el = np.radians(CAMERA_ELEVATIONS[i])

        # Camera position in spherical coordinates
        cam_x = distance * np.cos(el) * np.cos(az)
        cam_y = distance * np.cos(el) * np.sin(az)
        cam_z = distance * np.sin(el)

        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Look-at matrix (looking at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1.0, 0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Extrinsic: world → camera
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ cam_pos

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        cameras.append({
            'extrinsic': extrinsic,
            'intrinsic': K,
            'azimuth': CAMERA_AZIMUTHS[i],
            'elevation': CAMERA_ELEVATIONS[i],
            'position': cam_pos,
        })

    return cameras
