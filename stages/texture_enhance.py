"""
Texture Enhancement — Post-processing to improve texture quality.

1. Input image preprocessing (resize, sharpen, enhance)
2. Real-ESRGAN super-resolution on generated texture maps
3. Sharpening and color correction
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional


def preprocess_input(image_path: str, output_path: str,
                      target_size: int = 1024) -> str:
    """
    Preprocess input image for optimal InstantMesh quality.

    - Resize to target resolution (InstantMesh works best with 1024x1024-ish)
    - Center crop to square
    - Enhance contrast/sharpness
    - Save as PNG

    Returns: path to preprocessed image
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    print(f"   ℹ Input: {w}×{h}")

    # Center crop to square
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize to target (InstantMesh internally resizes to 320, but
    # higher input = better rembg + better Zero123++ conditioning)
    if min_dim != target_size:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    # Enhance: slight sharpness boost helps AI understand detail
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)

    # Enhance: slight contrast boost
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)

    # Save
    img.save(output_path, "PNG")
    print(f"   ✓ Preprocessed: {target_size}×{target_size}")
    return output_path


def upscale_texture(texture_path: str, output_path: str,
                     scale: int = 2, method: str = "auto") -> str:
    """
    Upscale texture map using Real-ESRGAN or PIL fallback.

    Args:
        texture_path: Path to texture image
        output_path: Path to save upscaled texture
        scale: Upscale factor (2x or 4x)
        method: 'realesrgan', 'pil', or 'auto' (try realesrgan, fallback to pil)

    Returns: path to upscaled texture
    """
    if not os.path.exists(texture_path):
        print(f"   ⚠ Texture not found: {texture_path}")
        return texture_path

    img = Image.open(texture_path)
    w, h = img.size
    print(f"   ℹ Texture: {w}×{h} → {w*scale}×{h*scale}")

    if method == "auto":
        # Try Real-ESRGAN first
        try:
            return _upscale_realesrgan(texture_path, output_path, scale)
        except Exception as e:
            print(f"   ⚠ Real-ESRGAN unavailable ({e}), using PIL Lanczos")
            return _upscale_pil(texture_path, output_path, scale)
    elif method == "realesrgan":
        return _upscale_realesrgan(texture_path, output_path, scale)
    else:
        return _upscale_pil(texture_path, output_path, scale)


def _upscale_realesrgan(texture_path: str, output_path: str, scale: int) -> str:
    """Upscale using Real-ESRGAN (GPU-accelerated)."""
    import torch
    from PIL import Image

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        raise ImportError("Install: pip install realesrgan basicsr")

    # Choose model based on scale
    if scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        model_name = "RealESRGAN_x2plus"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth",
        model=model,
        half=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    import cv2
    img = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=scale)
    cv2.imwrite(output_path, output)

    size = os.path.getsize(output_path)
    print(f"   ✓ Real-ESRGAN upscaled: {output_path} ({size / 1024:.0f}KB)")
    return output_path


def _upscale_pil(texture_path: str, output_path: str, scale: int) -> str:
    """Upscale using PIL Lanczos (CPU, no extra deps)."""
    img = Image.open(texture_path)
    w, h = img.size
    new_size = (w * scale, h * scale)

    # Lanczos upscale
    upscaled = img.resize(new_size, Image.LANCZOS)

    # Sharpen after upscale to recover detail
    upscaled = upscaled.filter(ImageFilter.SHARPEN)

    upscaled.save(output_path, "PNG")
    size = os.path.getsize(output_path)
    print(f"   ✓ PIL upscaled: {output_path} ({size / 1024:.0f}KB)")
    return output_path


def enhance_texture(texture_path: str, output_path: Optional[str] = None) -> str:
    """
    Post-process texture: sharpen, fix colors, improve contrast.
    """
    if output_path is None:
        output_path = texture_path

    img = Image.open(texture_path).convert("RGB")

    # 1. Sharpen
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)

    # 2. Contrast boost (textures from NeRF tend to be flat)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    # 3. Color saturation (slight boost for metals/gems)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)

    # 4. Unsharp mask for crisp detail
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))

    img.save(output_path, "PNG")
    print(f"   ✓ Texture enhanced: {output_path}")
    return output_path
