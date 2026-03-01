"""
Mesh Processing — Smooth, simplify, and UV-unwrap the reconstructed mesh.

Pipeline:
  1. Laplacian smoothing → removes voxelization artifacts
  2. Quadric decimation → reduce triangle count for web
  3. UV unwrap with xatlas → automatic UV parameterization
  4. Export ready for texture baking
"""

import numpy as np
from typing import Optional


def smooth_mesh(vertices: np.ndarray, faces: np.ndarray,
                iterations: int = 10, lambda_factor: float = 0.5) -> np.ndarray:
    """
    Laplacian mesh smoothing.

    Moves each vertex toward the average of its neighbors.
    """
    num_verts = len(vertices)

    # Build adjacency list
    adjacency = [set() for _ in range(num_verts)]
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    smoothed = vertices.copy()

    for iteration in range(iterations):
        new_pos = smoothed.copy()
        for v in range(num_verts):
            neighbors = list(adjacency[v])
            if len(neighbors) > 0:
                avg = np.mean(smoothed[neighbors], axis=0)
                new_pos[v] = smoothed[v] + lambda_factor * (avg - smoothed[v])
        smoothed = new_pos

    print(f"   ✓ Smoothed ({iterations} iterations, λ={lambda_factor})")
    return smoothed


def simplify_mesh(vertices: np.ndarray, faces: np.ndarray,
                   target_faces: int = 5000) -> tuple:
    """
    Simplify mesh using vertex clustering (fast, works without Open3D).
    """
    if len(faces) <= target_faces:
        print(f"   ✓ Mesh already has {len(faces)} faces (target: {target_faces})")
        return vertices, faces

    try:
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # Quadric decimation
        simplified = mesh.simplify_quadric_decimation(target_faces)
        new_verts = np.asarray(simplified.vertices)
        new_faces = np.asarray(simplified.triangles)

        print(f"   ✓ Simplified: {len(faces)} → {len(new_faces)} faces")
        return new_verts, new_faces

    except ImportError:
        # Fallback: uniform subsampling (crude but works)
        ratio = target_faces / len(faces)
        if ratio >= 1.0:
            return vertices, faces

        # Keep every Nth face
        step = max(1, int(1 / ratio))
        kept_faces = faces[::step]

        # Reindex vertices
        used_verts = np.unique(kept_faces.ravel())
        vert_map = np.full(len(vertices), -1, dtype=np.int64)
        vert_map[used_verts] = np.arange(len(used_verts))
        new_verts = vertices[used_verts]
        new_faces = vert_map[kept_faces]

        # Remove faces with invalid vertices
        valid = np.all(new_faces >= 0, axis=1)
        new_faces = new_faces[valid]

        print(f"   ✓ Simplified (subsampling): {len(faces)} → {len(new_faces)} faces")
        return new_verts, new_faces


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals."""
    normals = np.zeros_like(vertices)

    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(face_normal)
        if norm > 1e-8:
            face_normal /= norm
        for idx in f:
            normals[idx] += face_normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals /= norms

    return normals


def uv_unwrap(vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """
    Automatic UV unwrapping using xatlas.
    Falls back to spherical projection if xatlas not available.

    Returns: (uv_vertices, uv_coords, uv_faces)
    """
    try:
        import xatlas

        # xatlas expects float32
        verts_f32 = vertices.astype(np.float32)
        faces_u32 = faces.astype(np.uint32)

        atlas = xatlas.Atlas()
        atlas.add_mesh(verts_f32, faces_u32)
        atlas.generate()

        # Get output
        uv_verts, uv_indices, uv_coords = atlas[0]

        print(f"   ✓ UV unwrap (xatlas): {len(uv_verts)} UV vertices, "
              f"{len(uv_indices)} UV faces")
        return uv_verts, uv_coords, uv_indices

    except ImportError:
        print("   ⚠ xatlas not available, using spherical UV projection")
        return _spherical_uv(vertices, faces)


def _spherical_uv(vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """
    Fallback: spherical UV projection.
    Maps vertex positions to UV using spherical coordinates.
    """
    # Center vertices
    center = vertices.mean(axis=0)
    centered = vertices - center

    # Spherical coordinates
    r = np.linalg.norm(centered, axis=1)
    r[r < 1e-8] = 1e-8

    # θ (azimuth) → u
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    u = (theta + np.pi) / (2 * np.pi)

    # φ (elevation) → v
    phi = np.arcsin(np.clip(centered[:, 2] / r, -1, 1))
    v = (phi + np.pi / 2) / np.pi

    uv_coords = np.stack([u, v], axis=1).astype(np.float32)

    print(f"   ✓ UV unwrap (spherical): {len(vertices)} vertices")
    return vertices.astype(np.float32), uv_coords, faces.astype(np.uint32)


def run(vertices: np.ndarray, faces: np.ndarray,
        smooth_iterations: int = 10,
        target_faces: int = 5000) -> dict:
    """
    Complete mesh processing pipeline.

    Returns dict with processed mesh data.
    """
    print("   ℹ Smoothing mesh...")
    smoothed_verts = smooth_mesh(vertices, faces, iterations=smooth_iterations)

    print("   ℹ Simplifying mesh...")
    simple_verts, simple_faces = simplify_mesh(smoothed_verts, faces, target_faces)

    print("   ℹ Computing normals...")
    normals = compute_normals(simple_verts, simple_faces)

    print("   ℹ UV unwrapping...")
    uv_verts, uv_coords, uv_faces = uv_unwrap(simple_verts, simple_faces)

    return {
        'vertices': uv_verts,
        'faces': uv_faces,
        'normals': normals,
        'uv_coords': uv_coords,
        'original_vertices': simple_verts,
        'original_faces': simple_faces,
    }
