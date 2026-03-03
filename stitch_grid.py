#!/usr/bin/env python3
import os
import sys
from PIL import Image

def stitch_6_views(image_paths, output_path):
    """
    Stitches 6 images into a 3x2 grid (640x960).
    Expected order: 0, 1, 2, 3, 4, 5
    Each image will be resized to 320x320.
    """
    if len(image_paths) != 6:
        raise ValueError(f"Expected exactly 6 images, got {len(image_paths)}")

    grid_img = Image.new('RGB', (640, 960), (255, 255, 255))
    
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img = img.resize((320, 320), Image.LANCZOS)
        
        row = idx // 2
        col = idx % 2
        grid_img.paste(img, (col * 320, row * 320))
        
    grid_img.save(output_path, "PNG")
    print(f"✅ Grid saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python stitch_grid.py <out.png> <img0> <img1> <img2> <img3> <img4> <img5>")
        sys.exit(1)
        
    out = sys.argv[1]
    imgs = sys.argv[2:]
    stitch_6_views(imgs, out)
