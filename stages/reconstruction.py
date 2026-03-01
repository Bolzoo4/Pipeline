"""
3D Reconstruction — Visual hull carving from multi-view silhouettes.

Takes multi-view images and their camera parameters,
creates silhouette masks, carves a voxel grid, then extracts a mesh.

Pipeline:
  1. Create binary silhouette mask for each view (using rembg)
  2. Initialize 3D voxel grid
  3. For each view: project voxels → carve away those outside silhouette
  4. Marching cubes → triangle mesh
"""

import numpy as np
from typing import Optional


def create_silhouettes(views: list[np.ndarray]) -> list[np.ndarray]:
    """
    Create binary silhouette masks from view images.
    Uses rembg for robust background removal, with fallback methods.
    """
    import cv2

    silhouettes = []

    # Try rembg first (best quality)
    rembg_session = None
    try:
        from rembg import remove, new_session
        rembg_session = new_session("u2net")
        print("   ℹ Using rembg for silhouette extraction")
    except ImportError:
        print("   ⚠ rembg not available, using color-based segmentation")

    for i, view in enumerate(views):
        if rembg_session is not None:
            mask = _rembg_silhouette(view, rembg_session)
        else:
            mask = _color_silhouette(view)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        fg_ratio = np.sum(mask > 0) / mask.size
        print(f"   ✓ View {i}: {fg_ratio*100:.1f}% foreground")

        # Sanity check: if >90% foreground, silhouette is bad
        if fg_ratio > 0.90:
            print(f"   ⚠ View {i}: too much foreground, trying adaptive method")
            mask = _adaptive_silhouette(view)
            fg_ratio = np.sum(mask > 0) / mask.size
            print(f"   ✓ View {i} (adaptive): {fg_ratio*100:.1f}% foreground")

        silhouettes.append(mask)

    return silhouettes


def _rembg_silhouette(view: np.ndarray, session) -> np.ndarray:
    """Extract silhouette using rembg (neural background removal)."""
    from PIL import Image as PILImage
    from rembg import remove

    # rembg expects RGB PIL image
    pil_img = PILImage.fromarray(view)
    result = remove(pil_img, session=session)

    # Extract alpha channel as mask
    result_np = np.array(result)
    if result_np.shape[2] == 4:
        alpha = result_np[:, :, 3]
        # Threshold alpha to binary
        _, mask = __import__('cv2').threshold(alpha, 128, 255, __import__('cv2').THRESH_BINARY)
        return mask
    else:
        # No alpha, fall back to color
        return _color_silhouette(view)


def _color_silhouette(view: np.ndarray) -> np.ndarray:
    """Extract silhouette using color-based segmentation."""
    import cv2

    # Convert to multiple color spaces for robust detection
    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(view, cv2.COLOR_RGB2HSV)

    # Method 1: Background is typically the most common color
    # Sample corners to estimate background color
    h, w = view.shape[:2]
    corners = [
        view[0:10, 0:10],      # top-left
        view[0:10, w-10:w],    # top-right
        view[h-10:h, 0:10],    # bottom-left
        view[h-10:h, w-10:w],  # bottom-right
    ]
    bg_color = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

    # Distance from background color
    diff = np.linalg.norm(view.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    # Threshold: pixels far from background = foreground
    threshold = max(30, np.percentile(diff, 30))
    mask = (diff > threshold).astype(np.uint8) * 255

    # Method 2: Saturation-based (objects tend to have higher saturation than gray bg)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)

    # Combine: pixel is foreground if either method says so
    combined = cv2.bitwise_or(mask, sat_mask)

    return combined


def _adaptive_silhouette(view: np.ndarray) -> np.ndarray:
    """Last resort: adaptive thresholding + edge-based segmentation."""
    import cv2

    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)

    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Flood fill from corners (background)
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = dilated.copy()
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        cv2.floodFill(filled, mask, seed, 255)

    # Invert: what wasn't filled = object
    result = cv2.bitwise_not(filled)

    return result


def visual_hull_carving(silhouettes: list[np.ndarray],
                         cameras: list[dict],
                         grid_size: int = 128,
                         volume_extent: float = 1.2) -> np.ndarray:
    """
    Carve a 3D voxel grid using multi-view silhouettes.
    """
    print(f"   ℹ Voxel grid: {grid_size}³ = {grid_size**3:,} voxels")

    coords = np.linspace(-volume_extent, volume_extent, grid_size)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')

    voxel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    num_voxels = voxel_points.shape[0]
    voxel_homo = np.hstack([voxel_points, np.ones((num_voxels, 1))])

    occupancy = np.ones(num_voxels, dtype=bool)

    for i, (sil, cam) in enumerate(zip(silhouettes, cameras)):
        K = cam['intrinsic']
        E = cam['extrinsic']
        P = K @ E[:3]
        projected = (P @ voxel_homo.T).T

        z = projected[:, 2]
        valid_z = z > 0.01
        px = np.zeros(num_voxels)
        py = np.zeros(num_voxels)
        px[valid_z] = projected[valid_z, 0] / z[valid_z]
        py[valid_z] = projected[valid_z, 1] / z[valid_z]

        h, w = sil.shape
        ix = np.round(px).astype(np.int32)
        iy = np.round(py).astype(np.int32)

        in_bounds = valid_z & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        inside_silhouette = np.zeros(num_voxels, dtype=bool)
        inside_silhouette[in_bounds] = sil[iy[in_bounds], ix[in_bounds]] > 0

        # Only carve if this view has meaningful silhouette (not all fg)
        fg_ratio = np.sum(sil > 0) / sil.size
        if fg_ratio < 0.90:
            occupancy &= (inside_silhouette | ~in_bounds)
        else:
            print(f"   ⚠ View {i}: skipping (silhouette too large)")
            continue

        n_remaining = np.sum(occupancy)
        print(f"   ✓ View {i}: {n_remaining:,} voxels remaining")

    voxel_grid = occupancy.reshape(grid_size, grid_size, grid_size)
    print(f"   ✓ Visual hull: {np.sum(voxel_grid):,} occupied voxels")
    return voxel_grid


def extract_mesh(voxel_grid: np.ndarray,
                  volume_extent: float = 1.2) -> tuple:
    """
    Extract triangle mesh from voxel grid using marching cubes.
    """
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(voxel_grid.astype(np.float32), sigma=1.0)

    # Safety: check that a surface exists
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmin >= 0.5 or vmax <= 0.5:
        print(f"   ⚠ Volume range [{vmin:.3f}, {vmax:.3f}], adjusting level...")
        level = (vmin + vmax) / 2
    else:
        level = 0.5

    # Pad the volume with zeros so marching cubes creates a closed surface
    padded = np.pad(smoothed, pad_width=2, mode='constant', constant_values=0)

    vertices, faces, normals, values = marching_cubes(padded, level=level)

    # Adjust for padding offset
    vertices -= 2

    # Scale from grid coords to world coords
    grid_size = voxel_grid.shape[0]
    vertices = (vertices / grid_size - 0.5) * 2 * volume_extent

    print(f"   ✓ Mesh: {len(vertices):,} vertices, {len(faces):,} faces")
    return vertices, faces, normals


def run(views: list[np.ndarray], cameras: list[dict],
        grid_size: int = 128, mock: bool = False) -> dict:
    """
    Full reconstruction pipeline.
    """
    print("   ℹ Creating silhouettes...")
    silhouettes = create_silhouettes(views)

    print("   ℹ Carving visual hull...")
    voxel_grid = visual_hull_carving(silhouettes, cameras, grid_size=grid_size)

    total_occupied = np.sum(voxel_grid)
    total_voxels = voxel_grid.size

    if total_occupied == 0:
        raise ValueError("Visual hull is empty — no object reconstructed.")

    if total_occupied == total_voxels:
        print("   ⚠ All voxels occupied — no carving happened. Using center sphere fallback.")
        # Create a sphere in the center as fallback
        coords = np.linspace(-1, 1, grid_size)
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
        dist = np.sqrt(xx**2 + yy**2 + zz**2)
        voxel_grid = (dist < 0.5).astype(np.float32)

    print("   ℹ Extracting mesh (marching cubes)...")
    vertices, faces, normals = extract_mesh(voxel_grid)

    return {
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'silhouettes': silhouettes,
        'voxel_grid': voxel_grid,
    }
