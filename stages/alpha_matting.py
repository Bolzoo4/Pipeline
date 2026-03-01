"""
Alpha Matting stage — ViTMatte-based high-fidelity alpha extraction.

In MOCK mode: generates smooth alpha from trimap using bilateral filter.
In REAL mode: uses ViTMatte for sub-pixel accurate alpha estimation.
"""

import numpy as np
import cv2


def alpha_from_trimap_mock(image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    """
    Mock alpha matting: smooths the trimap into a continuous alpha map.
    Uses guided filter to create realistic-looking soft edges.
    """
    alpha = trimap.astype(np.float32) / 255.0

    alpha_smooth = cv2.bilateralFilter(alpha, d=9, sigmaColor=0.3, sigmaSpace=15)

    # Ensure definite foreground/background regions stay sharp
    alpha_smooth[trimap == 255] = 1.0
    alpha_smooth[trimap == 0] = 0.0

    return (alpha_smooth * 255).astype(np.uint8)


def alpha_from_trimap_real(image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    """
    Real alpha matting using ViTMatte.
    Computes continuous alpha values in the uncertain trimap region.
    """
    try:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        import torch
        from PIL import Image as PILImage
    except ImportError:
        raise RuntimeError(
            "transformers/torch not installed. "
            "Use --mock flag for local testing without GPU."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
    model = VitMatteForImageMatting.from_pretrained(
        "hustvl/vitmatte-small-composition-1k",
        torch_dtype=dtype,
    ).to(device)

    # Convert BGR→RGB PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)
    pil_trimap = PILImage.fromarray(trimap)

    # Process
    inputs = processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract alpha
    alpha = outputs.alphas.squeeze().cpu().numpy()
    alpha = (alpha * 255).clip(0, 255).astype(np.uint8)

    # Resize back to original if needed
    h, w = image.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    return alpha


def run(image: np.ndarray, trimap: np.ndarray, mock: bool = True) -> np.ndarray:
    """
    Run alpha matting stage.
    Returns alpha map as uint8 [0, 255].
    """
    if mock:
        return alpha_from_trimap_mock(image, trimap)
    return alpha_from_trimap_real(image, trimap)
