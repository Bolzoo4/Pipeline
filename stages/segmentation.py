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
    Real segmentation using SAM2 via HuggingFace transformers.
    Uses AutoModel directly instead of pipeline() to avoid type mismatch bugs.
    """
    import torch

    try:
        from transformers import AutoModelForMaskGeneration, AutoProcessor
        from PIL import Image as PILImage

        print("   ℹ Using SAM2 (AutoModel)")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained("facebook/sam2-hiera-large")
        model = AutoModelForMaskGeneration.from_pretrained(
            "facebook/sam2-hiera-large",
        ).to(device).eval()

        # Convert BGR→RGB PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(image_rgb)
        h, w = image.shape[:2]

        # Use a center point prompt (jewelry is typically centered in catalog photos)
        # SAM2 needs 4 levels: [image_level, object_level, point_level, coordinates]
        input_points = [[[[w // 2, h // 2]]]]
        input_labels = [[[1]]]  # foreground

        inputs = processor(
            images=pil_img,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get masks and scores
        masks = processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )

        if not masks or len(masks[0]) == 0:
            raise ValueError("SAM2 found no masks")

        # Take the mask with highest IoU score
        scores = outputs.iou_scores[0].cpu().numpy()
        best_idx = scores.argmax()
        binary_mask = masks[0][best_idx].squeeze().cpu().numpy().astype(np.uint8) * 255

        # Ensure it's 2D
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]

        print(f"   ✓ SAM2 mask: {np.sum(binary_mask > 0)} foreground pixels (score: {scores.flatten()[best_idx]:.3f})")

        del model, processor
        torch.cuda.empty_cache()

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
