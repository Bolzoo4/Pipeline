"""
Segmentation stage — SAM2-based jewelry segmentation + trimap generation.

In MOCK mode: generates a synthetic circular mask centered on the image.
In REAL mode: uses SAM2 for precise jewelry segmentation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional


def generate_trimap(mask: np.ndarray, dilate_px: int = 15, erode_px: int = 10) -> np.ndarray:
    """
    Convert a binary mask into a trimap with three regions:
    - 255 (white): definite foreground
    - 128 (gray): uncertain region (border)
    - 0 (black): definite background
    """
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))

    dilated = cv2.dilate(mask, kernel_d, iterations=1)
    eroded = cv2.erode(mask, kernel_e, iterations=1)

    trimap = np.zeros_like(mask)
    trimap[dilated > 127] = 128  # uncertain
    trimap[eroded > 127] = 255   # definite foreground

    return trimap


def segment_mock(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mock segmentation: creates an elliptical mask centered on the image.
    Simulates a ring-shaped jewelry item.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a ring-like elliptical mask in the center
    center = (w // 2, h // 2)
    axes_outer = (min(w, h) // 3, min(w, h) // 4)
    axes_inner = (min(w, h) // 5, min(w, h) // 7)

    cv2.ellipse(mask, center, axes_outer, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, center, axes_inner, 0, 0, 360, 0, -1)

    trimap = generate_trimap(mask)
    return mask, trimap


def segment_real(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Real segmentation using SAM2.
    Uses automatic mask generation, then picks the most central/largest mask.
    """
    import torch

    try:
        # SAM2 has multiple install paths depending on version
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            HAS_SAM2 = True
        except ImportError:
            HAS_SAM2 = False

        if not HAS_SAM2:
            # Fallback: use HuggingFace transformers pipeline
            from transformers import pipeline
            print("   ℹ Using HuggingFace SAM2 pipeline (transformers)")
            
            segmentor = pipeline(
                "mask-generation",
                model="facebook/sam2-hiera-large",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            # Convert BGR to RGB for HF pipeline
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_rgb)
            
            outputs = segmentor(pil_img, points_per_batch=64)
            masks = outputs["masks"]
            
            if not masks:
                raise ValueError("SAM2 found no objects in the image")
            
            # Score masks: prefer large, central masks
            h, w = image.shape[:2]
            center_x, center_y = w / 2, h / 2
            best_mask = None
            best_score = -1
            
            for mask_pil in masks:
                mask_np = np.array(mask_pil).astype(np.uint8) * 255
                area = np.sum(mask_np > 0)
                
                # Skip very small or very large masks
                total_px = h * w
                if area < total_px * 0.01 or area > total_px * 0.9:
                    continue
                
                # Centrality score
                ys, xs = np.where(mask_np > 0)
                if len(xs) == 0:
                    continue
                cx, cy = xs.mean(), ys.mean()
                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
                centrality = 1.0 - dist
                
                # Combined score: area * centrality
                score = (area / total_px) * centrality
                if score > best_score:
                    best_score = score
                    best_mask = mask_np
            
            if best_mask is None:
                raise ValueError("No suitable jewelry mask found. Try a cleaner background image.")
            
            binary_mask = best_mask
        else:
            # Native SAM2 package
            print("   ℹ Using native SAM2 package")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            sam2 = build_sam2(
                "sam2_hiera_l",
                "sam2_hiera_large.pt",
                device=device,
            )
            mask_generator = SAM2AutomaticMaskGenerator(sam2)
            
            # SAM2 expects RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image_rgb)
            
            if not masks:
                raise ValueError("SAM2 found no objects in the image")
            
            # Take the largest mask
            largest = max(masks, key=lambda m: m["area"])
            binary_mask = (largest["segmentation"].astype(np.uint8)) * 255

    except Exception as e:
        print(f"   ⚠ SAM2 failed: {e}")
        print("   ℹ Falling back to GrabCut segmentation...")
        binary_mask = _grabcut_fallback(image)

    trimap = generate_trimap(binary_mask)
    return binary_mask, trimap


def _grabcut_fallback(image: np.ndarray) -> np.ndarray:
    """
    Fallback segmentation using OpenCV GrabCut.
    Less accurate than SAM2 but works without GPU or model downloads.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # Assume jewelry is centered — init rect in center 60%
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Convert GrabCut mask to binary
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return binary


def run(image: np.ndarray, mock: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Run segmentation stage.
    Returns (binary_mask, trimap).
    """
    if mock:
        return segment_mock(image)
    return segment_real(image)
