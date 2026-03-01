"""
Normal Map Estimation stage — Marigold-based monocular normal estimation.

In MOCK mode: generates normals from image gradients (Sobel-based).
In REAL mode: uses Marigold diffusion model for high-quality normal estimation.
"""

import numpy as np
import cv2


def normals_from_gradients_mock(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Mock normal estimation using Sobel gradients.
    Creates a plausible normal map from image luminance gradients.
    Output is in tangent space: RGB = (N_xyz + 1) / 2 * 255
    """
    # Convert to grayscale for gradient computation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gray = image.astype(np.float32) / 255.0

    # Compute gradients (approximate surface normals)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Scale gradients for visible normal variation
    strength = 2.0
    grad_x *= strength
    grad_y *= strength

    # Construct normal vectors: N = normalize(-dI/dx, -dI/dy, 1)
    nx = -grad_x
    ny = -grad_y
    nz = np.ones_like(nx)

    # Normalize
    magnitude = np.sqrt(nx**2 + ny**2 + nz**2)
    magnitude = np.maximum(magnitude, 1e-8)
    nx /= magnitude
    ny /= magnitude
    nz /= magnitude

    # Convert from [-1, 1] to [0, 255] (tangent space encoding)
    normal_map = np.stack([
        ((nx + 1.0) * 0.5 * 255).astype(np.uint8),
        ((ny + 1.0) * 0.5 * 255).astype(np.uint8),
        ((nz + 1.0) * 0.5 * 255).astype(np.uint8),
    ], axis=-1)

    # Mask out background
    alpha_mask = (alpha > 10).astype(np.uint8)
    flat_normal = np.array([128, 128, 255], dtype=np.uint8)  # pointing straight up
    for c in range(3):
        normal_map[:, :, c] = normal_map[:, :, c] * alpha_mask + flat_normal[c] * (1 - alpha_mask)

    return normal_map


def normals_from_marigold_real(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Real normal estimation using Marigold diffusion model.
    """
    try:
        from diffusers import MarigoldNormalsPipeline
        import torch
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "diffusers/torch not installed. "
            "Use --mock flag for local testing without GPU."
        )

    # Marigold expects PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    pipe = MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1",
        variant="fp16",
        torch_dtype=torch.float16,
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    output = pipe(pil_image, num_inference_steps=4)
    normal_np = output.prediction[0]  # [H, W, 3] in [-1, 1]

    # Convert to tangent space encoding [0, 255]
    normal_map = ((normal_np + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

    # Convert RGB to BGR for consistency
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)

    return normal_map


def run(image: np.ndarray, alpha: np.ndarray, mock: bool = True) -> np.ndarray:
    """
    Run normal estimation stage.
    Returns normal map as uint8 BGR image [0, 255].
    """
    if mock:
        return normals_from_gradients_mock(image, alpha)
    return normals_from_marigold_real(image, alpha)
