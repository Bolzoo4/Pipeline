"""
Toroidal UV Unwrap — Creates UV-mapped textures for ring jewelry.

Converts a front-view ring photo into a toroidal UV texture map that
wraps correctly on a Three.js TorusGeometry.

Approach:
  1. Detect ring geometry from segmentation mask (center, inner/outer radius)
  2. Polar-to-Cartesian unwrap: ring photo → rectangular strip
  3. Separate band (repeating metal) from setting (diamond/stone)
  4. Fill unseen portions (back half, inner tube) with band texture
  5. Output: complete UV texture maps (albedo, normal, roughness, metallic)

UV Convention (matches Three.js TorusGeometry):
  u ∈ [0, 1] = φ / 2π = position around the ring loop (major circle)
  v ∈ [0, 1] = θ / 2π = position around the tube cross-section (minor circle)

From a front-view photo we see roughly:
  u ∈ [0.25, 0.75] (front half of the ring)
  v ∈ [0.25, 0.75] (outward-facing half of the tube)
"""

import numpy as np
import cv2
from typing import Optional


# ─── Ring Geometry Detection ───

def detect_ring_geometry(mask: np.ndarray) -> dict:
    """
    Detect ring center, inner/outer radius, and orientation from the mask.
    Returns dict with: center, inner_radius, outer_radius, angle, axes.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in ring mask")

    # Sort by area — largest is outer boundary, second is inner hole
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Outer contour → fit ellipse
    outer = contours[0]
    if len(outer) < 5:
        # Fallback: use bounding circle
        (cx, cy), radius = cv2.minEnclosingCircle(outer)
        return {
            'center': (int(cx), int(cy)),
            'outer_radius': int(radius),
            'inner_radius': int(radius * 0.6),
            'angle': 0,
            'outer_axes': (int(radius), int(radius)),
            'inner_axes': (int(radius * 0.6), int(radius * 0.6)),
        }

    outer_ellipse = cv2.fitEllipse(outer)
    (cx, cy), (w_outer, h_outer), angle = outer_ellipse

    # Inner contour (hole) → if exists
    inner_radius_est = min(w_outer, h_outer) * 0.3  # default estimate
    inner_axes = (int(inner_radius_est), int(inner_radius_est))

    if len(contours) > 1 and len(contours[1]) >= 5:
        inner_ellipse = cv2.fitEllipse(contours[1])
        _, (w_inner, h_inner), _ = inner_ellipse
        inner_axes = (int(w_inner / 2), int(h_inner / 2))
        inner_radius_est = (w_inner + h_inner) / 4

    return {
        'center': (int(cx), int(cy)),
        'outer_radius': int((w_outer + h_outer) / 4),
        'inner_radius': int(inner_radius_est),
        'angle': angle,
        'outer_axes': (int(w_outer / 2), int(h_outer / 2)),
        'inner_axes': inner_axes,
    }


# ─── Band / Setting Separation ───

def separate_band_setting(image: np.ndarray, mask: np.ndarray,
                          geom: dict, setting_angle_range: float = 60.0
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate the ring into band (repeating metal) and setting (stone/decoration).

    The setting is typically at the top of the ring (12 o'clock position).
    We detect it by looking for the region with highest color variance.

    Returns: (band_mask, setting_mask, band_color)
    """
    cx, cy = geom['center']
    h, w = mask.shape[:2]

    # Create angular map
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        band_color = np.array([180, 170, 130], dtype=np.uint8)  # default gold
        return mask, np.zeros_like(mask), band_color

    angles = np.degrees(np.arctan2(ys - cy, xs - cx))  # -180 to 180

    # Find the angle with highest local color variance → that's the setting
    angle_bins = np.linspace(-180, 180, 36)
    variances = []
    for i in range(len(angle_bins) - 1):
        in_bin = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
        if np.sum(in_bin) < 10:
            variances.append(0)
            continue
        bin_pixels = image[ys[in_bin], xs[in_bin]]
        variances.append(np.mean(np.var(bin_pixels.astype(np.float32), axis=0)))

    variances = np.array(variances)

    # Setting is at the angle with peak variance
    if variances.max() > 0:
        peak_bin = np.argmax(variances)
        setting_center_angle = (angle_bins[peak_bin] + angle_bins[peak_bin + 1]) / 2
    else:
        setting_center_angle = -90  # default: top of ring

    # Create masks
    setting_mask = np.zeros_like(mask)
    band_mask = np.zeros_like(mask)

    half_range = setting_angle_range / 2
    for idx in range(len(xs)):
        angle = angles[idx]
        # Handle wraparound
        diff = abs(angle - setting_center_angle)
        if diff > 180:
            diff = 360 - diff
        if diff < half_range:
            setting_mask[ys[idx], xs[idx]] = 255
        else:
            band_mask[ys[idx], xs[idx]] = 255

    # Extract band color: median of band pixels
    band_pixels = image[band_mask > 0]
    if len(band_pixels) > 0:
        band_color = np.median(band_pixels, axis=0).astype(np.uint8)
    else:
        band_color = np.array([180, 170, 130], dtype=np.uint8)

    return band_mask, setting_mask, band_color


# ─── Polar Unwrap ───

def polar_unwrap(image: np.ndarray, mask: np.ndarray, geom: dict,
                 uv_width: int = 1024, uv_height: int = 256
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap the ring photo using polar coordinates.

    Converts from (x, y) image space to (φ, r) polar space centered on the ring.
    φ = angle around the ring → maps to u (horizontal in UV texture)
    r = radial position across the band → maps to v (vertical in UV texture)

    Returns: (unwrapped_strip, validity_mask)
    """
    cx, cy = geom['center']
    inner_r = geom['inner_radius']
    outer_r = geom['outer_radius']

    # The "tube" of the ring spans from inner_r to outer_r
    # Add some padding
    r_min = max(0, inner_r - 5)
    r_max = outer_r + 5
    r_range = r_max - r_min

    strip = np.zeros((uv_height, uv_width, 3), dtype=np.uint8)
    validity = np.zeros((uv_height, uv_width), dtype=np.uint8)

    h, w = image.shape[:2]

    for j in range(uv_height):
        # v: position across the band (0 = inner, 1 = outer)
        r = r_min + (j / uv_height) * r_range

        for i in range(uv_width):
            # u: angle around the ring (-π to π)
            angle = ((i / uv_width) * 2 * np.pi) - np.pi

            # Convert polar → image coords
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))

            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                strip[j, i] = image[y, x]
                validity[j, i] = 255

    return strip, validity


def polar_unwrap_fast(image: np.ndarray, mask: np.ndarray, geom: dict,
                      uv_width: int = 1024, uv_height: int = 256
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of polar_unwrap (much faster).
    """
    cx, cy = geom['center']
    inner_r = geom['inner_radius']
    outer_r = geom['outer_radius']

    r_min = max(0, inner_r - 5)
    r_max = outer_r + 5

    # Create coordinate grids
    u_coords = np.linspace(0, 2 * np.pi, uv_width, endpoint=False) - np.pi  # angles
    v_coords = np.linspace(r_min, r_max, uv_height)  # radii

    angles_grid, radii_grid = np.meshgrid(u_coords, v_coords)

    # Polar → image coordinates
    x_coords = (cx + radii_grid * np.cos(angles_grid)).astype(np.int32)
    y_coords = (cy + radii_grid * np.sin(angles_grid)).astype(np.int32)

    h, w = image.shape[:2]

    # Clip to image bounds
    x_clipped = np.clip(x_coords, 0, w - 1)
    y_clipped = np.clip(y_coords, 0, h - 1)

    # Sample image
    strip = image[y_clipped, x_clipped]

    # Validity: in bounds AND in mask
    in_bounds = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    in_mask = mask[y_clipped, x_clipped] > 0
    validity = (in_bounds & in_mask).astype(np.uint8) * 255

    return strip, validity


# ─── Create Full Toroidal UV Map ───

def create_toroidal_uv(strip: np.ndarray, validity: np.ndarray,
                       band_color: np.ndarray,
                       uv_width: int = 1024, uv_height: int = 512
                       ) -> np.ndarray:
    """
    Create a full toroidal UV texture from the polar-unwrapped strip.

    The strip covers the visible portion of the ring (front half).
    We fill the invisible portions with band color/texture.

    UV texture layout:
      u (horizontal) = φ, position around the ring (0 → 2π)
      v (vertical) = θ, position around the tube cross-section (0 → 2π)

    The polar strip maps to the outward-facing part of the tube (v ≈ 0.25-0.75).
    Inner tube and back of ring get filled with band.
    """
    strip_h, strip_w = strip.shape[:2]

    # Create full UV texture filled with band color
    uv_tex = np.full((uv_height, uv_width, 3), band_color, dtype=np.uint8)

    # The strip represents the outward-facing part of the tube
    # In the full UV, this occupies roughly v ∈ [0.2, 0.8] (the outer half of the tube)
    v_start = int(uv_height * 0.15)
    v_end = int(uv_height * 0.85)
    v_span = v_end - v_start

    # Resize strip to fit the visible tube portion
    strip_resized = cv2.resize(strip, (uv_width, v_span), interpolation=cv2.INTER_LINEAR)
    validity_resized = cv2.resize(validity, (uv_width, v_span), interpolation=cv2.INTER_NEAREST)

    # Paste strip into UV texture where valid
    for j in range(v_span):
        for i in range(uv_width):
            if validity_resized[j, i] > 0:
                uv_tex[v_start + j, i] = strip_resized[j, i]

    # Apply gaussian blur to blend edges between strip and band fill
    alpha_mask = np.zeros(uv_tex.shape[:2], dtype=np.float32)
    alpha_mask[v_start:v_end] = validity_resized.astype(np.float32) / 255.0
    alpha_mask = cv2.GaussianBlur(alpha_mask, (15, 15), 3)

    # Blend: combine UV data with band fill using alpha
    band_fill = np.full_like(uv_tex, band_color)
    for c in range(3):
        uv_tex[:, :, c] = (uv_tex[:, :, c].astype(np.float32) * alpha_mask +
                           band_fill[:, :, c].astype(np.float32) * (1 - alpha_mask)).astype(np.uint8)

    return uv_tex


def create_toroidal_uv_fast(strip: np.ndarray, validity: np.ndarray,
                            band_color: np.ndarray,
                            uv_width: int = 1024, uv_height: int = 512
                            ) -> np.ndarray:
    """
    Vectorized version of create_toroidal_uv.
    """
    strip_h, strip_w = strip.shape[:2]

    # Fill with band color
    uv_tex = np.full((uv_height, uv_width, 3), band_color, dtype=np.uint8)

    # Place strip in the outer tube region
    v_start = int(uv_height * 0.15)
    v_end = int(uv_height * 0.85)
    v_span = v_end - v_start

    strip_resized = cv2.resize(strip, (uv_width, v_span), interpolation=cv2.INTER_LINEAR)
    validity_resized = cv2.resize(validity, (uv_width, v_span), interpolation=cv2.INTER_NEAREST)

    # Create alpha mask for blending
    alpha = np.zeros((uv_height, uv_width), dtype=np.float32)
    alpha[v_start:v_end] = validity_resized.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (21, 21), 5)

    # Place strip data
    uv_data = uv_tex.copy()
    uv_data[v_start:v_end] = np.where(
        validity_resized[:, :, np.newaxis] > 0,
        strip_resized,
        uv_data[v_start:v_end]
    )

    # Blend with band color
    alpha_3d = alpha[:, :, np.newaxis]
    band_fill = np.full_like(uv_tex, band_color, dtype=np.float32)
    result = (uv_data.astype(np.float32) * alpha_3d +
              band_fill * (1.0 - alpha_3d))

    return result.clip(0, 255).astype(np.uint8)


# ─── Process All Maps ───

def unwrap_map(texture_map: np.ndarray, mask: np.ndarray, geom: dict,
               fill_value: np.ndarray,
               uv_width: int = 1024, uv_height: int = 512,
               is_grayscale: bool = False) -> np.ndarray:
    """
    Unwrap any texture map (normal, roughness, metallic) into toroidal UV space.
    """
    if is_grayscale and texture_map.ndim == 2:
        # Convert to 3-channel for uniform processing
        texture_map = cv2.cvtColor(texture_map, cv2.COLOR_GRAY2BGR)
        fill_value = np.array([fill_value[0]] * 3, dtype=np.uint8)

    strip, validity = polar_unwrap_fast(texture_map, mask, geom,
                                         uv_width=uv_width, uv_height=uv_height // 2)
    uv_map = create_toroidal_uv_fast(strip, validity, fill_value,
                                      uv_width=uv_width, uv_height=uv_height)

    if is_grayscale:
        uv_map = cv2.cvtColor(uv_map, cv2.COLOR_BGR2GRAY)

    return uv_map


# ─── Main Entry Point ───

def run(image: np.ndarray, mask: np.ndarray, alpha: np.ndarray,
        normals: np.ndarray, roughness: np.ndarray, metallic: np.ndarray,
        uv_width: int = 1024, uv_height: int = 512,
        mock: bool = False) -> dict:
    """
    Create toroidal UV-mapped texture bundle from ring photo.

    Returns dict of UV-mapped textures:
      'albedo', 'alpha', 'normal', 'roughness', 'metallic'
    """
    print("   ℹ Detecting ring geometry...")
    geom = detect_ring_geometry(mask)
    print(f"   ✓ Ring center: {geom['center']}, outer_r: {geom['outer_radius']}, inner_r: {geom['inner_radius']}")

    print("   ℹ Separating band / setting...")
    band_mask, setting_mask, band_color = separate_band_setting(image, mask, geom)
    print(f"   ✓ Band color: BGR({band_color[0]}, {band_color[1]}, {band_color[2]})")

    print("   ℹ Creating toroidal UV maps...")

    # Albedo UV
    uv_albedo = unwrap_map(image, mask, geom, band_color,
                           uv_width=uv_width, uv_height=uv_height)

    # Alpha UV
    uv_alpha = unwrap_map(alpha, mask, geom,
                          fill_value=np.array([255, 255, 255], dtype=np.uint8),
                          uv_width=uv_width, uv_height=uv_height,
                          is_grayscale=True)

    # Normal UV — fill with flat normal (128, 128, 255)
    uv_normal = unwrap_map(normals, mask, geom,
                           fill_value=np.array([128, 128, 255], dtype=np.uint8),
                           uv_width=uv_width, uv_height=uv_height)

    # Roughness UV — fill with band roughness estimate
    band_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    band_roughness = np.array([int(255 - np.median(band_gray[mask > 0]))],
                               dtype=np.uint8) if np.any(mask > 0) else np.array([80], dtype=np.uint8)
    uv_roughness = unwrap_map(roughness, mask, geom,
                              fill_value=np.array([band_roughness[0]] * 3, dtype=np.uint8),
                              uv_width=uv_width, uv_height=uv_height,
                              is_grayscale=True)

    # Metallic UV — fill with high metallic for band
    uv_metallic = unwrap_map(metallic, mask, geom,
                             fill_value=np.array([230, 230, 230], dtype=np.uint8),
                             uv_width=uv_width, uv_height=uv_height,
                             is_grayscale=True)

    print(f"   ✓ UV textures generated ({uv_width}×{uv_height})")

    return {
        'albedo': uv_albedo,
        'alpha': uv_alpha,
        'normal': uv_normal,
        'roughness': uv_roughness,
        'metallic': uv_metallic,
        'geometry': geom,
    }
