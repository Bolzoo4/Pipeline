"""
3D Reconstruction — Visual hull carving from multi-view silhouettes.

Takes multi-view images and their camera parameters,
creates silhouette masks, carves a voxel grid, then extracts a mesh.

Pipeline:
  1. Create binary silhouette mask for each view
  2. Initialize 3D voxel grid
  3. For each view: project voxels → carve away those outside silhouette
  4. Marching cubes → triangle mesh
"""

import numpy as np
from typing import Optional


def create_silhouettes(views: list[np.ndarray], threshold: int = 240) -> list[np.ndarray]:
    """
    Create binary silhouette masks from view images.
    Assumes white/light background.
    """
    import cv2

    silhouettes = []
    for view in views:
        # Convert to grayscale
        if view.ndim == 3:
            gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        else:
            gray = view

        # Threshold: dark pixels = object, light pixels = background
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # If less than 5% of pixels are foreground, try Otsu
        fg_ratio = np.sum(mask > 0) / mask.size
        if fg_ratio < 0.05:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        silhouettes.append(mask)

    return silhouettes


def visual_hull_carving(silhouettes: list[np.ndarray],
                         cameras: list[dict],
                         grid_size: int = 128,
                         volume_extent: float = 1.2) -> np.ndarray:
    """
    Carve a 3D voxel grid using multi-view silhouettes.

    Args:
        silhouettes: Binary masks for each view (HxW, 0=bg, 255=fg)
        cameras: Camera dicts with 'extrinsic' and 'intrinsic'
        grid_size: Resolution of the voxel grid (N³)
        volume_extent: Half-extent of the volume in world units

    Returns:
        3D voxel occupancy grid (grid_size³, bool)
    """
    print(f"   ℹ Voxel grid: {grid_size}³ = {grid_size**3:,} voxels")

    # Create 3D grid of world coordinates
    coords = np.linspace(-volume_extent, volume_extent, grid_size)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')

    # Flatten to (N³, 3)
    voxel_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    num_voxels = voxel_points.shape[0]

    # Homogeneous coordinates
    voxel_homo = np.hstack([voxel_points, np.ones((num_voxels, 1))])

    # Start with all voxels occupied
    occupancy = np.ones(num_voxels, dtype=bool)

    for i, (sil, cam) in enumerate(zip(silhouettes, cameras)):
        K = cam['intrinsic']        # 3x3
        E = cam['extrinsic']        # 4x4

        # Project voxels to this view: pixel = K @ E[:3] @ voxel_homo.T
        P = K @ E[:3]               # 3x4 projection matrix
        projected = (P @ voxel_homo.T).T  # (N³, 3)

        # Normalize by z
        z = projected[:, 2]
        valid_z = z > 0.01
        px = np.zeros(num_voxels)
        py = np.zeros(num_voxels)
        px[valid_z] = projected[valid_z, 0] / z[valid_z]
        py[valid_z] = projected[valid_z, 1] / z[valid_z]

        # Round to pixel indices
        h, w = sil.shape
        ix = np.round(px).astype(np.int32)
        iy = np.round(py).astype(np.int32)

        # Check bounds
        in_bounds = valid_z & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        # Check silhouette: voxel is carved if it projects outside the silhouette
        inside_silhouette = np.zeros(num_voxels, dtype=bool)
        inside_silhouette[in_bounds] = sil[iy[in_bounds], ix[in_bounds]] > 0

        # Carve: remove voxels that are NOT inside the silhouette in this view
        occupancy &= (inside_silhouette | ~in_bounds)

        n_remaining = np.sum(occupancy)
        print(f"   ✓ View {i}: {n_remaining:,} voxels remaining "
              f"(silhouette: {np.sum(sil > 0):,} px)")

    # Reshape to 3D grid
    voxel_grid = occupancy.reshape(grid_size, grid_size, grid_size)

    print(f"   ✓ Visual hull: {np.sum(voxel_grid):,} occupied voxels")
    return voxel_grid


def extract_mesh(voxel_grid: np.ndarray,
                  volume_extent: float = 1.2) -> tuple:
    """
    Extract triangle mesh from voxel grid using marching cubes.

    Returns: (vertices, faces, normals) as numpy arrays
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from scipy.ndimage import gaussian_filter
        # Fallback: manual marching cubes via scipy
        raise RuntimeError("scikit-image required for marching cubes. "
                          "Install: pip install scikit-image")

    # Smooth the voxel grid slightly for better mesh
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(voxel_grid.astype(np.float32), sigma=1.0)

    # Marching cubes
    vertices, faces, normals, values = marching_cubes(
        smoothed, level=0.5
    )

    # Scale vertices from grid coordinates to world coordinates
    grid_size = voxel_grid.shape[0]
    vertices = (vertices / grid_size - 0.5) * 2 * volume_extent

    print(f"   ✓ Mesh: {len(vertices):,} vertices, {len(faces):,} faces")

    return vertices, faces, normals


def run(views: list[np.ndarray], cameras: list[dict],
        grid_size: int = 128, mock: bool = False) -> dict:
    """
    Full reconstruction pipeline.

    Args:
        views: List of multi-view images (RGB, HxWx3)
        cameras: Camera parameters for each view
        grid_size: Voxel grid resolution

    Returns:
        dict with 'vertices', 'faces', 'normals'
    """
    print("   ℹ Creating silhouettes...")
    silhouettes = create_silhouettes(views)

    print("   ℹ Carving visual hull...")
    voxel_grid = visual_hull_carving(silhouettes, cameras, grid_size=grid_size)

    if np.sum(voxel_grid) == 0:
        raise ValueError("Visual hull is empty — no object reconstructed. "
                         "Check silhouette masks and camera Parameters.")

    print("   ℹ Extracting mesh (marching cubes)...")
    vertices, faces, normals = extract_mesh(voxel_grid)

    return {
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'silhouettes': silhouettes,
        'voxel_grid': voxel_grid,
    }
