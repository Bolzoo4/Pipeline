"""
Unique3D ISOMER Wrapper — runs inside /workspace/Unique3D.

This script is called as a subprocess by unique3d_stage.py.
It loads the Unique3D models and runs the full pipeline:
  Single Image → Multiview Generation → Normal Estimation → ISOMER Mesh
"""

import os
import sys
import argparse
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to single input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for mesh")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Try to use Unique3D's built-in pipeline ───
    # The exact imports depend on the repo structure.
    # We try two known layouts:
    
    # Layout A: AiuniAI/Unique3D (main repo with app/ directory)
    try:
        from app.inference import run_pipeline
        print("   [Unique3D] Using app.inference.run_pipeline...")
        run_pipeline(
            input_image=args.input_image,
            output_dir=args.output_dir,
            device=device,
        )
        print(f"   ✅ Reconstruction complete → {args.output_dir}")
        return
    except ImportError:
        pass

    # Layout B: Gradio-based (app/gradio_local.py exposes functions)
    try:
        from app.gradio_local import generate3d
        print("   [Unique3D] Using Gradio generate3d function...")
        
        input_img = Image.open(args.input_image).convert("RGB")
        mesh_path = os.path.join(args.output_dir, "mesh.obj")
        
        generate3d(input_img, mesh_path)
        print(f"   ✅ Reconstruction complete → {mesh_path}")
        return
    except ImportError:
        pass

    # Layout C: Manual pipeline assembly
    try:
        print("   [Unique3D] Assembling pipeline manually...")
        
        # 1. Load input
        input_img = Image.open(args.input_image).convert("RGB")
        
        # 2. Generate multiview (Unique3D's own diffusion model)
        from scripts.generate_views import generate_multiview
        multiview_imgs = generate_multiview(input_img, device=device)
        print(f"   [Unique3D] Generated {len(multiview_imgs)} views")
        
        # 3. Estimate normals
        from scripts.generate_normals import estimate_normals
        normal_imgs = estimate_normals(multiview_imgs, device=device)
        print(f"   [Unique3D] Estimated {len(normal_imgs)} normal maps")
        
        # 4. ISOMER reconstruction
        from mesh_reconstruction.isomer import reconstruct_mesh
        mesh_path = os.path.join(args.output_dir, "mesh.obj")
        tex_path = os.path.join(args.output_dir, "mesh.png")
        
        reconstruct_mesh(
            multiview_imgs, normal_imgs,
            mesh_path=mesh_path,
            texture_path=tex_path,
            device=device,
        )
        print(f"   ✅ Reconstruction complete → {mesh_path}")
        return
    except ImportError as e:
        print(f"   ⚠ Manual assembly failed: {e}")

    # Fallback: use the CLI if available
    print("   [Unique3D] Trying CLI fallback...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "app/gradio_local.py", 
         "--input", args.input_image,
         "--output", args.output_dir,
         "--no-gui"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"   ✅ CLI reconstruction complete → {args.output_dir}")
    else:
        print(f"   ❌ All methods failed. stderr:\n{result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
