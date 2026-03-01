#!/usr/bin/env python3
"""
Jewelry Asset Pipeline — CLI orchestrator.

Processes a catalog image through 5 stages to produce a UV-mapped asset bundle:
  1. Segmentation (SAM2 / GrabCut)
  2. Alpha Matting (ViTMatte)
  3. Normal Estimation (Marigold)
  4. PBR Heuristics (Roughness + Metallic)
  5. Toroidal UV Unwrap → final UV-mapped textures

Usage:
    python pipeline.py --input ring.jpg --output ./output_bundle/
    python pipeline.py --input ring.jpg --output ./output_bundle/ --mock
    python pipeline.py --input ring.jpg --output ./output_bundle/ --real -c ring
"""

import json
import time
from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image

from stages import segmentation, alpha_matting, normal_estimation, pbr_heuristics, uv_unwrap


def save_webp(image: np.ndarray, path: Path, quality: int = 90) -> int:
    """Save image as WebP. Returns file size in bytes."""
    if len(image.shape) == 2:
        pil_img = Image.fromarray(image, mode="L")
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    pil_img.save(str(path), "WEBP", quality=quality)
    return path.stat().st_size


@click.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Path to the catalog image (JPEG/PNG)")
@click.option("--output", "-o", "output_dir", required=True, type=click.Path(),
              help="Output directory for the asset bundle")
@click.option("--category", "-c", default="ring",
              type=click.Choice(["ring", "earring", "necklace", "bracelet", "watch"]),
              help="Jewelry category")
@click.option("--mock/--real", default=True,
              help="Use mock mode (no GPU required) or real AI models")
@click.option("--quality", default=90, type=int,
              help="WebP compression quality (0-100)")
@click.option("--uv-width", default=1024, type=int,
              help="UV texture width")
@click.option("--uv-height", default=512, type=int,
              help="UV texture height")
def main(input_path: str, output_dir: str, category: str, mock: bool,
         quality: int, uv_width: int, uv_height: int):
    """Process a jewelry catalog image into a UV-mapped PBR asset bundle."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "MOCK" if mock else "REAL (GPU)"
    click.echo(f"🔧 Pipeline mode: {mode_label}")
    click.echo(f"💎 Category: {category}")
    click.echo(f"📸 Input: {input_path}")
    click.echo(f"📦 Output: {output_dir}")
    click.echo(f"🗺️  UV size: {uv_width}×{uv_height}")
    click.echo("─" * 50)

    # Load input image
    image = cv2.imread(str(input_path))
    if image is None:
        raise click.ClickException(f"Failed to load image: {input_path}")

    h, w = image.shape[:2]
    click.echo(f"📐 Image size: {w}×{h}")

    total_start = time.time()
    bundle_size = 0

    # Stage 1: Segmentation → Trimap
    click.echo("\n🔍 Stage 1/5: Segmentation...")
    t = time.time()
    mask, trimap = segmentation.run(image, mock=mock)
    click.echo(f"   ✓ Segmentation complete ({time.time() - t:.2f}s)")

    # Stage 2: Alpha Matting
    click.echo("🎭 Stage 2/5: Alpha Matting...")
    t = time.time()
    alpha = alpha_matting.run(image, trimap, mock=mock)
    click.echo(f"   ✓ Alpha matting complete ({time.time() - t:.2f}s)")

    # Stage 3: Normal Estimation
    click.echo("🗺️  Stage 3/5: Normal Map...")
    t = time.time()
    normals = normal_estimation.run(image, alpha, mock=mock)
    click.echo(f"   ✓ Normal estimation complete ({time.time() - t:.2f}s)")

    # Stage 4: PBR Heuristics
    click.echo("✨ Stage 4/5: PBR Maps (Roughness + Metallic)...")
    t = time.time()
    roughness, metallic = pbr_heuristics.run(image, alpha)
    click.echo(f"   ✓ PBR heuristics complete ({time.time() - t:.2f}s)")

    # Stage 5: Toroidal UV Unwrap
    click.echo("🔄 Stage 5/5: Toroidal UV Unwrap...")
    t = time.time()
    uv_bundle = uv_unwrap.run(
        image, mask, alpha, normals, roughness, metallic,
        uv_width=uv_width, uv_height=uv_height, mock=mock,
    )
    click.echo(f"   ✓ UV unwrap complete ({time.time() - t:.2f}s)")

    # Save UV-mapped textures
    click.echo("\n💾 Saving UV-mapped bundle...")

    # Albedo (with alpha channel)
    uv_albedo = uv_bundle['albedo']
    uv_alpha = uv_bundle['alpha']
    albedo_rgb = cv2.cvtColor(uv_albedo, cv2.COLOR_BGR2RGB)
    if uv_alpha.ndim == 2:
        albedo_rgba = np.dstack([albedo_rgb, uv_alpha])
    else:
        albedo_rgba = np.dstack([albedo_rgb, uv_alpha[:, :, 0]])
    pil_albedo = Image.fromarray(albedo_rgba, mode="RGBA")
    albedo_path = output_dir / "albedo.webp"
    pil_albedo.save(str(albedo_path), "WEBP", quality=quality)
    bundle_size += albedo_path.stat().st_size
    click.echo(f"   ✓ UV albedo ({albedo_path.stat().st_size / 1024:.1f}KB)")

    # Normal
    size = save_webp(uv_bundle['normal'], output_dir / "normal.webp", quality)
    bundle_size += size
    click.echo(f"   ✓ UV normal ({size / 1024:.1f}KB)")

    # Roughness
    size = save_webp(uv_bundle['roughness'], output_dir / "roughness.webp", quality)
    bundle_size += size
    click.echo(f"   ✓ UV roughness ({size / 1024:.1f}KB)")

    # Metallic
    size = save_webp(uv_bundle['metallic'], output_dir / "metallic.webp", quality)
    bundle_size += size
    click.echo(f"   ✓ UV metallic ({size / 1024:.1f}KB)")

    # Alpha (separate for the renderer)
    size = save_webp(uv_bundle['alpha'], output_dir / "alpha.webp", quality)
    bundle_size += size
    click.echo(f"   ✓ UV alpha ({size / 1024:.1f}KB)")

    # Save metadata
    geom = uv_bundle.get('geometry', {})
    metadata = {
        "version": "2.0.0",
        "uv_mapping": "toroidal",
        "category": category,
        "source_image": input_path.name,
        "dimensions": {"width": w, "height": h},
        "uv_dimensions": {"width": uv_width, "height": uv_height},
        "pipeline_mode": "mock" if mock else "real",
        "quality": quality,
        "files": {
            "albedo": "albedo.webp",
            "alpha": "alpha.webp",
            "normal": "normal.webp",
            "roughness": "roughness.webp",
            "metallic": "metallic.webp",
        },
        "ring_geometry": {
            "center": geom.get('center', [0, 0]),
            "outer_radius": geom.get('outer_radius', 0),
            "inner_radius": geom.get('inner_radius', 0),
        },
        "bundle_size_bytes": bundle_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    total_time = time.time() - total_start
    click.echo("─" * 50)
    click.echo(f"✅ Pipeline complete in {total_time:.2f}s")
    click.echo(f"📦 Bundle size: {bundle_size / 1024:.1f}KB")
    click.echo(f"🔄 UV mapping: toroidal ({uv_width}×{uv_height})")
    click.echo(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()
