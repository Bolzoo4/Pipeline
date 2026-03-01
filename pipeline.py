#!/usr/bin/env python3
"""
Jewelry Asset Pipeline — Multi-View 3D Reconstruction.

Processes a jewelry catalog image (or multi-view set) through a full 3D
reconstruction pipeline to produce a textured .glb model.

Pipeline Stages:
  1. Multi-view generation (Zero123++ from single image, or real photos)
  2. 3D Reconstruction (visual hull carving + marching cubes)
  3. Mesh processing (smooth, simplify, UV unwrap)
  4. Texture baking (project views → UV maps + PBR)
  5. Export .glb

Usage:
    # Single image → AI multiview → 3D
    python pipeline.py -i ring.jpg -o ./bundle/ --real -c ring

    # Real multi-view photos
    python pipeline.py -i views/ -o ./bundle/ --real -c ring --multiview

    # Mock mode (no GPU, synthetic data)
    python pipeline.py -i ring.jpg -o ./bundle/ --mock -c ring
"""

import json
import time
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image

from stages import multiview_gen, reconstruction, mesh_processing, texture_baking


def save_webp(image: np.ndarray, path: Path, quality: int = 90) -> int:
    """Save image as WebP. Returns file size in bytes."""
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode="L")
    elif image.shape[2] == 4:
        pil_img = Image.fromarray(image, mode="RGBA")
    else:
        pil_img = Image.fromarray(image)
    pil_img.save(str(path), "WEBP", quality=quality)
    return path.stat().st_size


def export_glb(mesh_data: dict, textures: dict, output_path: Path) -> int:
    """
    Export mesh + textures as .glb file.
    Uses trimesh if available, otherwise a minimal manual GLB.
    """
    try:
        import trimesh

        vertices = mesh_data.get('original_vertices', mesh_data['vertices'])
        faces = mesh_data.get('original_faces', mesh_data['faces'])
        uv_coords = mesh_data.get('uv_coords')

        # Create trimesh
        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float64),
            faces=faces.astype(np.int64),
        )

        # Add UV coordinates as a visual
        if uv_coords is not None and len(uv_coords) > 0:
            # Create a texture image from the albedo
            albedo = textures.get('albedo')
            if albedo is not None:
                albedo_pil = Image.fromarray(albedo)

                # Create material with the albedo texture
                material = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=albedo_pil,
                    metallicFactor=0.9,
                    roughnessFactor=0.3,
                )

                # Create TextureVisuals
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_coords[:len(vertices)],
                    material=material,
                )

        # Export as GLB
        glb_data = mesh.export(file_type='glb')
        output_path.write_bytes(glb_data)

        size = output_path.stat().st_size
        print(f"   ✓ GLB exported ({size / 1024:.1f}KB, "
              f"{len(vertices):,} verts, {len(faces):,} faces)")
        return size

    except ImportError:
        print("   ⚠ trimesh not available, saving OBJ instead")
        return _export_obj(mesh_data, output_path.with_suffix('.obj'))


def _export_obj(mesh_data: dict, output_path: Path) -> int:
    """Fallback: export as .obj file."""
    vertices = mesh_data.get('original_vertices', mesh_data['vertices'])
    faces = mesh_data.get('original_faces', mesh_data['faces'])

    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    size = output_path.stat().st_size
    print(f"   ✓ OBJ exported ({size / 1024:.1f}KB)")
    return size


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Single image or directory of multi-view photos")
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(),
              help="Output directory for the asset bundle")
@click.option("--category", "-c", default="ring",
              type=click.Choice(["ring", "earring", "necklace", "bracelet", "watch"]),
              help="Jewelry category")
@click.option("--mock/--real", default=True,
              help="Use mock mode (no GPU) or real AI models")
@click.option("--multiview", is_flag=True, default=False,
              help="Input is a directory of multi-view photos")
@click.option("--quality", default=90, type=int,
              help="Texture quality (0-100)")
@click.option("--grid-size", default=128, type=int,
              help="Voxel grid resolution (32-256)")
@click.option("--texture-size", default=1024, type=int,
              help="Output texture resolution")
@click.option("--target-faces", default=5000, type=int,
              help="Target face count for mesh simplification")
def main(input_path: str, output_dir: str, category: str, mock: bool,
         multiview: bool, quality: int, grid_size: int, texture_size: int,
         target_faces: int):
    """Process jewelry images into a textured 3D model (.glb)."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "MOCK" if mock else "REAL (GPU)"
    input_mode = "MULTI-VIEW" if multiview else "SINGLE IMAGE → AI MULTIVIEW"
    click.echo(f"🔧 Pipeline mode: {mode_label}")
    click.echo(f"📸 Input mode: {input_mode}")
    click.echo(f"💎 Category: {category}")
    click.echo(f"📸 Input: {input_path}")
    click.echo(f"📦 Output: {output_dir}")
    click.echo(f"🧊 Grid: {grid_size}³ | 🎨 Tex: {texture_size}px | 🔺 Target: {target_faces} faces")
    click.echo("─" * 60)

    total_start = time.time()

    # ─── Stage 1: Get multi-view images ───
    click.echo("\n📸 Stage 1/5: Multi-View Generation...")
    t = time.time()

    if multiview:
        # Load real photos from directory
        if input_path.is_dir():
            image_files = sorted(
                p for p in input_path.iterdir()
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
            )
            click.echo(f"   ℹ Found {len(image_files)} photos")
            views = multiview_gen.load_real_multiview([str(p) for p in image_files])
        else:
            raise click.ClickException("--multiview requires a directory of images")
    else:
        # Single image → generate multi-view
        image = cv2.imread(str(input_path))
        if image is None:
            raise click.ClickException(f"Failed to load: {input_path}")
        click.echo(f"   📐 Image: {image.shape[1]}×{image.shape[0]}")
        views = multiview_gen.generate_multiview(image, mock=mock)

    num_views = len(views)
    cameras = multiview_gen.get_camera_matrices(num_views)
    click.echo(f"   ✓ {num_views} views generated ({time.time() - t:.2f}s)")

    # Save views for debugging
    views_dir = output_dir / "views"
    views_dir.mkdir(exist_ok=True)
    for i, view in enumerate(views):
        cv2.imwrite(str(views_dir / f"view_{i:02d}.png"),
                    cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

    # ─── Stage 2: 3D Reconstruction ───
    click.echo("\n🧊 Stage 2/5: 3D Reconstruction...")
    t = time.time()
    recon = reconstruction.run(views, cameras, grid_size=grid_size, mock=mock)
    click.echo(f"   ✓ Reconstruction complete ({time.time() - t:.2f}s)")

    # ─── Stage 3: Mesh Processing ───
    click.echo("\n✨ Stage 3/5: Mesh Processing...")
    t = time.time()
    mesh_data = mesh_processing.run(
        recon['vertices'], recon['faces'],
        smooth_iterations=15,
        target_faces=target_faces,
    )
    click.echo(f"   ✓ Mesh processing complete ({time.time() - t:.2f}s)")

    # ─── Stage 4: Texture Baking ───
    click.echo("\n🎨 Stage 4/5: Texture Baking...")
    t = time.time()
    textures = texture_baking.run(mesh_data, views, cameras, texture_size=texture_size)
    click.echo(f"   ✓ Texture baking complete ({time.time() - t:.2f}s)")

    # ─── Stage 5: Export ───
    click.echo("\n💾 Stage 5/5: Export...")
    t = time.time()
    bundle_size = 0

    # Export GLB
    glb_size = export_glb(mesh_data, textures, output_dir / "model.glb")
    bundle_size += glb_size

    # Save textures as WebP (for web viewer fallback)
    for name in ['albedo', 'roughness', 'metallic', 'normal']:
        tex = textures.get(name)
        if tex is not None:
            size = save_webp(tex, output_dir / f"{name}.webp", quality)
            bundle_size += size
            click.echo(f"   ✓ {name}.webp ({size / 1024:.1f}KB)")

    # Save metadata
    metadata = {
        "version": "3.0.0",
        "format": "glb",
        "category": category,
        "source": str(input_path.name),
        "input_mode": "multiview" if multiview else "single_to_multiview",
        "pipeline_mode": "mock" if mock else "real",
        "num_views": num_views,
        "mesh": {
            "vertices": int(len(mesh_data.get('original_vertices', mesh_data['vertices']))),
            "faces": int(len(mesh_data.get('original_faces', mesh_data['faces']))),
        },
        "textures": {
            "size": texture_size,
            "quality": quality,
        },
        "files": {
            "model": "model.glb",
            "albedo": "albedo.webp",
            "roughness": "roughness.webp",
            "metallic": "metallic.webp",
            "normal": "normal.webp",
        },
        "bundle_size_bytes": bundle_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    click.echo(f"   ✓ Export complete ({time.time() - t:.2f}s)")

    total_time = time.time() - total_start
    click.echo("─" * 60)
    click.echo(f"✅ Pipeline complete in {total_time:.2f}s")
    click.echo(f"📦 Bundle size: {bundle_size / 1024:.1f}KB")
    click.echo(f"🧊 Model: model.glb ({glb_size / 1024:.1f}KB)")
    click.echo(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()
