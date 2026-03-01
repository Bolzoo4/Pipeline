"""
PBR Heuristics stage — Deterministic roughness and metallic map generation.

No AI models required. Works identically in mock and real modes.
"""

import numpy as np
import cv2


def compute_metallic(alpha: np.ndarray, metallic_value: float = 0.95) -> np.ndarray:
    """
    Generate metallic map.
    Jewelry is predominantly metal, so metallic is high (0.95) wherever alpha > 0.
    
    Args:
        alpha: Alpha map [0, 255]
        metallic_value: Metallic value for foreground pixels (0.0 - 1.0)
    
    Returns:
        Metallic map as uint8 [0, 255]
    """
    metallic = np.zeros_like(alpha, dtype=np.uint8)
    foreground = alpha > 10  # threshold for foreground
    metallic[foreground] = int(metallic_value * 255)
    return metallic


def compute_roughness(
    image: np.ndarray,
    alpha: np.ndarray,
    gamma: float = 0.4,
    min_roughness: float = 0.05,
    max_roughness: float = 0.45,
) -> np.ndarray:
    """
    Generate roughness map from inverted luma with gamma correction.
    
    - Specular highlights (bright) → low roughness (smooth/shiny)
    - Shadow areas (dark) → higher roughness (matte)
    
    Args:
        image: BGR input image
        alpha: Alpha map [0, 255]
        gamma: Gamma correction exponent
        min_roughness: Minimum roughness value (for specular highlights)
        max_roughness: Maximum roughness value (for shadow areas)
    
    Returns:
        Roughness map as uint8 [0, 255]
    """
    # Convert to grayscale (luma)
    if len(image.shape) == 3:
        luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        luma = image.astype(np.float32) / 255.0

    # Invert: bright areas → low roughness, dark areas → high roughness
    inverted = 1.0 - luma

    # Apply gamma correction for non-linear mapping
    roughness = np.power(inverted, gamma)

    # Scale to desired range
    roughness = min_roughness + roughness * (max_roughness - min_roughness)

    # Convert to uint8
    roughness_map = (roughness * 255).clip(0, 255).astype(np.uint8)

    # Zero out background
    roughness_map[alpha <= 10] = 0

    return roughness_map


def run(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run PBR heuristics stage.
    Returns (roughness_map, metallic_map) as uint8 [0, 255].
    """
    roughness = compute_roughness(image, alpha)
    metallic = compute_metallic(alpha)
    return roughness, metallic
