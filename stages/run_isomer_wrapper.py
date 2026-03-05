import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Unique3D imports (must be run with PYTHONPATH pointing to Unique3D)
try:
    from app.utils import set_seed, do_resize_content
    from app.normal_model import NormalModel
    from app.isomer import Isomer
except ImportError:
    print("❌ Error: Unique3D modules not found. Ensure PYTHONPATH includes /workspace/Unique3D")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_grid", type=str, required=True, help="Path to 2x2 RGB grid")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ckpt_path", type=str, default="/workspace/Unique3D/ckpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Models
    print("   [Unique3D] Loading models...")
    # These paths are based on typical Unique3D setup
    normal_model = NormalModel(device=device, ckpt_path=os.path.join(args.ckpt_path, "img_to_mv.pth"))
    isomer = Isomer(device=device, ckpt_path=os.path.join(args.ckpt_path, "mv_to_mesh.pth"))

    # 2. Load Input RGB Grid
    print(f"   [Unique3D] Loading RGB grid: {args.input_grid}")
    rgb_grid = Image.open(args.input_grid).convert("RGB")
    
    # 3. Estimate Normals
    # Unique3D expects 4 views. If grid is 2x2, we might need to split it
    # or ensure normal_model.predict takes the grid directly.
    # Usually, normal_model takes the multiview images.
    print("   [Unique3D] Estimating normal maps...")
    normals_grid = normal_model.predict(rgb_grid)
    normal_path = os.path.join(args.output_dir, "normals_grid.png")
    normals_grid.save(normal_path)

    # 4. Run ISOMER Reconstruction
    print("   [Unique3D] Running ISOMER...")
    # isomer.reconstruct typically takes RGB and Normals
    mesh_path = os.path.join(args.output_dir, "mesh.obj")
    isomer.reconstruct(rgb_grid, normals_grid, mesh_path)

    print(f"   ✅ Reconstruction complete: {mesh_path}")

if __name__ == "__main__":
    main()
