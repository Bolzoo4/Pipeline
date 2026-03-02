#!/usr/bin/env python3
"""
Jewelry Asset Pipeline v4 — InstantMesh 3D Reconstruction.

End-to-end: single image → InstantMesh (Zero123++ + LRM + FlexiCubes) → textured .glb

Usage:
    python pipeline.py -i ring.jpg -o ./bundle/ --real -c ring
    python pipeline.py -i ring.jpg -o ./bundle/ --mock -c ring  # skip AI, test flow
"""

import json
import time
import shutil
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image


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


def create_mock_glb(output_path: Path) -> int:
    """Create a mock GLB for testing (simple torus)."""
    try:
        import trimesh
        mesh = trimesh.creation.torus(major_radius=0.8, minor_radius=0.1, major_sections=48, minor_sections=24)
        # Add a simple gray color
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.tile([180, 180, 180, 255], (len(mesh.vertices), 1)).astype(np.uint8)
        )
        glb_data = mesh.export(file_type='glb')
        output_path.write_bytes(glb_data)
        return output_path.stat().st_size
    except ImportError:
        # Minimal empty GLB
        output_path.write_bytes(b'\x00' * 100)
        return 100


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input image path")
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(),
              help="Output directory for the asset bundle")
@click.option("--category", "-c", default="ring",
              type=click.Choice(["ring", "earring", "necklace", "bracelet", "watch"]),
              help="Jewelry category")
@click.option("--mock/--real", default=True,
              help="Mock mode (no GPU/AI) or real mode")
@click.option("--quality", default=90, type=int,
              help="Texture quality (0-100)")
@click.option("--steps", default=100, type=int,
              help="Diffusion steps for multi-view generation (more = better quality)")
@click.option("--seed", default=42, type=int,
              help="Random seed")
@click.option("--multiview", default="zero123",
              type=click.Choice(["zero123", "nanobanana"]),
              help="Engine to generate multi-view images")
def main(input_path: str, output_dir: str, category: str, mock: bool,
         quality: int, steps: int, seed: int, multiview: str):
    """Jewelry → 3D Model Pipeline (InstantMesh/NanoBanana)."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "MOCK" if mock else f"REAL (GPU) - {multiview.upper()}"
    click.echo(f"🔧 Pipeline mode: {mode_label}")
    click.echo(f"💎 Category: {category}")
    click.echo(f"📸 Input: {input_path}")
    click.echo(f"📦 Output: {output_dir}")
    click.echo("─" * 60)

    total_start = time.time()

    if mock:
        # ─── Mock mode: create a simple torus GLB ───
        click.echo("\n🧊 Mock mode: creating simple 3D model...")
        glb_path = output_dir / "model.glb"
        glb_size = create_mock_glb(glb_path)
        mesh_verts = 1152
        mesh_faces = 2304
        multiview_image = None
        texture_path = None
    else:
        # ─── Stage 1: Preprocess input ───
        click.echo("\n🖼️  Stage 1/4: Preprocessing input...")
        t = time.time()

        from stages.texture_enhance import preprocess_input, upscale_texture, enhance_texture

        processed_path = output_dir / "input_processed.png"
        preprocess_input(str(input_path), str(processed_path), target_size=1024)
        click.echo(f"   ✓ Preprocessed ({time.time() - t:.2f}s)")

        # ─── Stage 2: InstantMesh 3D Reconstruction ───
        click.echo(f"\n🚀 Stage 2/4: 3D Reconstruction ({multiview})...")
        t = time.time()

        from stages.instantmesh_stage import convert_to_glb, ensure_setup

        ensure_setup()
        
        # Result dictionary to hold outputs
        result = {}
        
        if multiview == "nanobanana":
            from stages.nanobanana_multiview import generate_multiview_grid
            import subprocess
            import sys
            import os
            
            grid_path = str(output_dir / "multiview_grid.png")
            click.echo("   [1/2] Generating Multiview Grid with Vertex AI (NanoBanana)...")
            generate_multiview_grid(str(processed_path), grid_path, category=category)
            
            click.echo("   [2/2] Extracting 3D Mesh with LRM...")
            lrm_dir = output_dir / "instantmesh" / "lrm_output"
            lrm_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup InstantMesh env
            INSTANTMESH_DIR = "/workspace/InstantMesh"
            env = os.environ.copy()
            env["PYTHONPATH"] = INSTANTMESH_DIR + ":" + env.get("PYTHONPATH", "")
            
            lrm_script = str(Path("stages/run_lrm_only.py").absolute())
            cmd = [
                sys.executable, lrm_script,
                os.path.join(INSTANTMESH_DIR, "configs", "instant-mesh-large.yaml"),
                grid_path,
                str(lrm_dir)
            ]
            
            proc = subprocess.run(cmd, cwd=INSTANTMESH_DIR, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"   ⚠ LRM failed:\n{proc.stderr}")
                raise RuntimeError(f"LRM Extraction failed (exit {proc.returncode})")
            
            # Find generated mesh (named multiview_grid.obj)
            mesh_path = str(lrm_dir / "multiview_grid.obj")
            tex_path = str(lrm_dir / "multiview_grid.png")
            
            result = {
                "mesh_path": mesh_path,
                "texture_map": tex_path if os.path.exists(tex_path) else None,
                "multiview_image": grid_path
            }
        else:
            # Traditional InstantMesh (Zero123++)
            from stages.instantmesh_stage import run as run_instantmesh
            result = run_instantmesh(
                str(processed_path),
                str(output_dir),
                export_texmap=True,
                diffusion_steps=steps,
                seed=seed,
            )
            
        click.echo(f"   ✓ 3D Reconstruction complete ({time.time() - t:.2f}s)")

        # ─── Stage 3: Texture Enhancement ───
        click.echo("\n✨ Stage 3/4: Texture Enhancement...")
        t = time.time()

        texture_path = result.get("texture_map")
        if texture_path and Path(texture_path).exists():
            # Enhance texture (sharpen, contrast, color)
            enhanced_path = str(Path(texture_path).parent / "texture_enhanced.png")
            enhance_texture(texture_path, enhanced_path)

            # Upscale texture 2x (Real-ESRGAN if available, PIL fallback)
            upscaled_path = str(Path(texture_path).parent / "texture_upscaled.png")
            upscale_texture(enhanced_path, upscaled_path, scale=2, method="auto")

            # Use the upscaled texture for GLB conversion
            result["texture_map"] = upscaled_path
            click.echo(f"   ✓ Texture enhanced + upscaled ({time.time() - t:.2f}s)")
        else:
            click.echo("   ⚠ No texture map to enhance")

        # ─── Stage 4: Export GLB ───
        click.echo("\n💾 Stage 4/4: Export GLB...")
        t = time.time()

        glb_path = output_dir / "model.glb"
        convert_to_glb(result["mesh_path"], str(glb_path), result.get("texture_map"))
        glb_size = glb_path.stat().st_size

        # Copy multiview image to bundle
        multiview_image = result.get("multiview_image")
        if multiview_image and Path(multiview_image).exists():
            shutil.copy2(multiview_image, output_dir / "multiview.png")

        # Copy texture map if present
        texture_path = result.get("texture_map")
        if texture_path and Path(texture_path).exists():
            shutil.copy2(texture_path, output_dir / "texture.png")
            tex_img = cv2.imread(texture_path)
            if tex_img is not None:
                save_webp(cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB),
                         output_dir / "albedo.webp", quality)

        # Get mesh info
        try:
            import trimesh
            mesh = trimesh.load(str(glb_path))
            if isinstance(mesh, trimesh.Scene):
                geom = list(mesh.geometry.values())[0]
                mesh_verts = len(geom.vertices)
                mesh_faces = len(geom.faces)
            else:
                mesh_verts = len(mesh.vertices)
                mesh_faces = len(mesh.faces)
        except Exception:
            mesh_verts = 0
            mesh_faces = 0

        click.echo(f"   ✓ Export complete ({time.time() - t:.2f}s)")

    # ─── Save metadata ───
    bundle_size = glb_size
    for f in output_dir.iterdir():
        if f.suffix in ('.webp', '.png') and f.name != 'multiview.png':
            bundle_size += f.stat().st_size

    metadata = {
        "version": "4.0.0",
        "format": "glb",
        "method": "instantmesh",
        "category": category,
        "source": input_path.name,
        "pipeline_mode": "mock" if mock else "real",
        "mesh": {
            "vertices": mesh_verts,
            "faces": mesh_faces,
        },
        "files": {
            "model": "model.glb",
        },
        "bundle_size_bytes": bundle_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if (output_dir / "albedo.webp").exists():
        metadata["files"]["albedo"] = "albedo.webp"
    if (output_dir / "texture.png").exists():
        metadata["files"]["texture"] = "texture.png"
    if (output_dir / "multiview.png").exists():
        metadata["files"]["multiview"] = "multiview.png"

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    total_time = time.time() - total_start
    click.echo("─" * 60)
    click.echo(f"✅ Pipeline complete in {total_time:.2f}s")
    click.echo(f"📦 Bundle: {bundle_size / 1024:.1f}KB")
    click.echo(f"🧊 Model: {mesh_verts:,} vertices, {mesh_faces:,} faces")
    click.echo(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()
