"""
Texture Baking — Project multi-view images onto UV-unwrapped mesh.

For each UV texel, finds the best source view and samples the corresponding pixel.
Also generates PBR maps (roughness, metallic) from the baked albedo.

Pipeline:
  1. For each mesh face → determine best view (most front-facing)
  2. For each UV texel → project to best view → sample pixel color
  3. Inpaint any gaps in the texture
  4. Generate roughness/metallic from albedo heuristics
  5. Pack into final texture atlas
"""

import numpy as np
import cv2
from typing import Optional


def bake_albedo(vertices: np.ndarray, faces: np.ndarray,
                uv_coords: np.ndarray, uv_faces: np.ndarray,
                normals: np.ndarray,
                views: list[np.ndarray], cameras: list[dict],
                texture_size: int = 1024) -> np.ndarray:
    """
    Bake albedo texture by projecting multi-view images onto UV map.

    For each face, picks the view where the face is most front-facing,
    then maps UV coordinates to view pixel coordinates.
    """
    tex = np.full((texture_size, texture_size, 3), 200, dtype=np.uint8)  # light gray bg
    tex_mask = np.zeros((texture_size, texture_size), dtype=bool)

    num_faces = len(uv_faces) if uv_faces is not None else len(faces)
    mesh_faces = faces if uv_faces is None else uv_faces

    # Precompute face normals (use original mesh faces)
    face_normals = np.zeros((len(faces), 3))
    for fi, f in enumerate(faces):
        if fi < len(vertices) - 2:
            v0, v1, v2 = vertices[f[0] % len(vertices)], \
                         vertices[f[1] % len(vertices)], \
                         vertices[f[2] % len(vertices)]
            edge1 = v1 - v0
            edge2 = v2 - v0
            fn = np.cross(edge1, edge2)
            norm = np.linalg.norm(fn)
            if norm > 1e-8:
                fn /= norm
            face_normals[fi] = fn

    # For each face, find best view
    for fi in range(min(num_faces, len(faces))):
        f_mesh = faces[fi] if fi < len(faces) else mesh_faces[fi]
        f_uv = mesh_faces[fi] if fi < len(mesh_faces) else f_mesh

        fn = face_normals[fi] if fi < len(face_normals) else np.array([0, 0, 1])

        # Find best view: highest dot product between face normal and view direction
        best_view = 0
        best_score = -1

        for vi, cam in enumerate(cameras):
            cam_pos = cam.get('position', np.array([0, 0, 2.5]))
            # Face center in world space
            face_center = np.mean([vertices[idx % len(vertices)] for idx in f_mesh], axis=0)
            view_dir = cam_pos - face_center
            view_dir_norm = np.linalg.norm(view_dir)
            if view_dir_norm > 1e-8:
                view_dir /= view_dir_norm

            # Dot product: how much the face points toward the camera
            score = np.dot(fn, view_dir)
            if score > best_score:
                best_score = score
                best_view = vi

        if best_score < 0.05:
            continue  # Face not visible from any view

        # Get UV coordinates for this face
        uvs = []
        for idx in f_uv:
            if idx < len(uv_coords):
                uvs.append(uv_coords[idx])
            else:
                uvs.append(np.array([0.0, 0.0]))
        uvs = np.array(uvs, dtype=np.float32)

        # Get 3D vertex positions
        verts_3d = []
        for idx in f_mesh:
            if idx < len(vertices):
                verts_3d.append(vertices[idx])
            else:
                verts_3d.append(np.zeros(3))
        verts_3d = np.array(verts_3d, dtype=np.float64)

        # Project 3D vertices to view pixel coordinates
        cam = cameras[best_view]
        K = cam['intrinsic']
        E = cam['extrinsic']
        P = K @ E[:3]

        # Project vertices
        verts_homo = np.hstack([verts_3d, np.ones((3, 1))])
        projected = (P @ verts_homo.T).T
        z = projected[:, 2]
        if np.any(z <= 0):
            continue

        px = projected[:, 0] / z
        py = projected[:, 1] / z

        view = views[best_view]
        vh, vw = view.shape[:2]

        # Rasterize triangle in UV space
        uv_px = (uvs * texture_size).astype(np.int32)
        view_px = np.stack([px, py], axis=1)

        # Simple scanline rasterization
        _rasterize_triangle(tex, tex_mask, uv_px, view_px, view, texture_size)

    # Inpaint gaps
    if np.sum(~tex_mask) > 0:
        mask_uint8 = (~tex_mask).astype(np.uint8) * 255
        tex = cv2.inpaint(tex, mask_uint8, 3, cv2.INPAINT_TELEA)

    coverage = np.sum(tex_mask) / tex_mask.size * 100
    print(f"   ✓ Albedo baked ({texture_size}×{texture_size}, {coverage:.1f}% coverage)")

    return tex


def _rasterize_triangle(tex: np.ndarray, mask: np.ndarray,
                         uv_tri: np.ndarray, view_tri: np.ndarray,
                         view_img: np.ndarray, tex_size: int):
    """
    Rasterize a single triangle: for each pixel inside the UV triangle,
    compute barycentric coords and sample from the projected view coordinates.
    """
    vh, vw = view_img.shape[:2]

    # Bounding box in UV space
    min_x = max(0, np.min(uv_tri[:, 0]))
    max_x = min(tex_size - 1, np.max(uv_tri[:, 0]))
    min_y = max(0, np.min(uv_tri[:, 1]))
    max_y = min(tex_size - 1, np.max(uv_tri[:, 1]))

    if max_x <= min_x or max_y <= min_y:
        return

    # Precompute barycentric coordinate vectors
    v0 = uv_tri[1].astype(np.float64) - uv_tri[0].astype(np.float64)
    v1 = uv_tri[2].astype(np.float64) - uv_tri[0].astype(np.float64)
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    denom = dot00 * dot11 - dot01 * dot01

    if abs(denom) < 1e-10:
        return

    inv_denom = 1.0 / denom

    for y in range(int(min_y), int(max_y) + 1):
        for x in range(int(min_x), int(max_x) + 1):
            p = np.array([x, y], dtype=np.float64) - uv_tri[0].astype(np.float64)
            dot_v0_p = np.dot(v0, p)
            dot_v1_p = np.dot(v1, p)

            u = (dot11 * dot_v0_p - dot01 * dot_v1_p) * inv_denom
            v = (dot00 * dot_v1_p - dot01 * dot_v0_p) * inv_denom

            if u >= 0 and v >= 0 and u + v <= 1:
                w = 1.0 - u - v
                # Interpolate view coordinates
                sample_x = w * view_tri[0, 0] + u * view_tri[1, 0] + v * view_tri[2, 0]
                sample_y = w * view_tri[0, 1] + u * view_tri[1, 1] + v * view_tri[2, 1]

                sx = int(np.clip(sample_x, 0, vw - 1))
                sy = int(np.clip(sample_y, 0, vh - 1))

                if 0 <= x < tex_size and 0 <= y < tex_size:
                    tex[y, x] = view_img[sy, sx]
                    mask[y, x] = True


def generate_pbr_maps(albedo: np.ndarray) -> tuple:
    """
    Generate roughness and metallic maps from albedo using heuristics.
    """
    gray = cv2.cvtColor(albedo, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Roughness: bright areas (metal) = smooth, dark areas = rough
    # Normalize to 0-1 range
    gray_norm = gray / 255.0

    # Roughness: inverse of brightness (metals are bright and smooth)
    roughness = (1.0 - gray_norm * 0.7) * 255
    roughness = np.clip(roughness, 30, 230).astype(np.uint8)

    # Metallic: bright areas are metallic
    metallic = (gray_norm * 0.9) * 255
    metallic = np.clip(metallic, 0, 255).astype(np.uint8)

    # Add local variance for more detail
    local_var = cv2.GaussianBlur(gray, (0, 0), 3) - cv2.GaussianBlur(gray, (0, 0), 15)
    local_var = np.abs(local_var)
    roughness = np.clip(roughness.astype(np.float32) + local_var * 0.3, 0, 255).astype(np.uint8)

    print(f"   ✓ PBR maps generated ({albedo.shape[1]}×{albedo.shape[0]})")
    return roughness, metallic


def run(mesh_data: dict, views: list[np.ndarray],
        cameras: list[dict], texture_size: int = 1024) -> dict:
    """
    Full texture baking pipeline.

    Returns dict with 'albedo', 'roughness', 'metallic', 'normal' textures.
    """
    print("   ℹ Baking albedo texture...")
    albedo = bake_albedo(
        mesh_data['original_vertices'] if 'original_vertices' in mesh_data else mesh_data['vertices'],
        mesh_data['original_faces'] if 'original_faces' in mesh_data else mesh_data['faces'],
        mesh_data['uv_coords'],
        mesh_data.get('faces'),
        mesh_data.get('normals', np.zeros_like(mesh_data['vertices'])),
        views, cameras,
        texture_size=texture_size,
    )

    print("   ℹ Generating PBR maps...")
    roughness, metallic = generate_pbr_maps(albedo)

    # Normal map: flat (128, 128, 255) — normals come from geometry
    normal = np.full_like(albedo, [128, 128, 255], dtype=np.uint8)

    return {
        'albedo': albedo,
        'roughness': roughness,
        'metallic': metallic,
        'normal': normal,
    }
