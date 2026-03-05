"""
Unique3D Wrapper — runs inside /workspace/Unique3D.

Calls the actual Unique3D pipeline functions directly (no Gradio needed).
We mock the gradio module to avoid import errors in headless mode.

Pipeline:
  1. model_zoo.init_models()       → load all checkpoints
  2. run_sr_fast()                 → upscale input if small
  3. run_mvprediction()            → generate 4 multiview images
  4. geo_reconstruct()             → ISOMER mesh reconstruction
  5. save_glb_and_video()          → export GLB
"""

import os
import sys
import types
import argparse

# ─── Mock gradio BEFORE any Unique3D imports ───
# Unique3D's internal modules import gradio at module level (gr.Error, gr.Warning, etc.)
# but we don't need any UI — this avoids the huggingface_hub incompatibility.
_gr = types.ModuleType('gradio')
_gr.Error = type('GradioError', (Exception,), {})
_gr.Warning = type('GradioWarning', (UserWarning,), {})
_gr.Info = lambda *a, **kw: None
_gr.Progress = lambda *a, **kw: (lambda f: f)
_gr.update = lambda **kw: kw
sys.modules['gradio'] = _gr
# Also mock gradio sub-modules that might get imported
for sub in ['gr', 'gradio.utils', 'gradio.networking']:
    sys.modules[sub] = _gr

# ─── Environment setup ───
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['DIFFUSERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

from PIL import Image
from pytorch3d.structures import Meshes

from app.all_models import model_zoo
from app.custom_models.mvimg_prediction import run_mvprediction
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.multiview_inference import geo_reconstruct
from scripts.utils import save_glb_and_video


def main():
    parser = argparse.ArgumentParser(description="Unique3D CLI inference")
    parser.add_argument("--input_image", type=str, required=False, help="Input single image")
    parser.add_argument("--grid", type=str, required=False, help="Pre-generated 2x2 multiview grid")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_remove_bg", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    if not args.input_image and not args.grid:
        raise ValueError("Must provide either --input_image or --grid")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load all models
    print("   [Unique3D] Loading models...")
    model_zoo.init_models()
    print("   [Unique3D] Models loaded ✓")

    remove_bg = not args.no_remove_bg

    if args.grid:
        print(f"   [Unique3D] Loading custom 2x2 multiview grid: {args.grid}")
        grid_img = Image.open(args.grid).convert("RGBA")
        
        # Cut grid into 4 views
        w, h = grid_img.size
        w2, h2 = w // 2, h // 2
        rgb_pils_rgba = [
            grid_img.crop((0, 0, w2, h2)),         # front (top-left)
            grid_img.crop((w2, 0, w, h2)),         # right (top-right)
            grid_img.crop((0, h2, w2, h)),         # back (bottom-left)
            grid_img.crop((w2, h2, w, h))          # left (bottom-right)
        ]
        
        # Flatten onto white background (ISOMER uses white BG if no alpha provided)
        rgb_pils = []
        for img in rgb_pils_rgba:
            white_bg = Image.new("RGB", img.size, (255, 255, 255))
            white_bg.paste(img, mask=img.split()[3])
            rgb_pils.append(white_bg)
            
        front_pil = rgb_pils[0]
        print(f"   [Unique3D] Split grid into 4 views ✓")

    else:
        # 2. Load & upscale input image
        print(f"   [Unique3D] Loading input: {args.input_image}")
        preview_img = Image.open(args.input_image).convert("RGB")
        
        if preview_img.size[0] <= 512:
            print("   [Unique3D] Upscaling input (<=512px)...")
            preview_img = run_sr_fast([preview_img])[0]
    
        # 3. Generate multiview images
        print("   [Unique3D] Generating multiview images...")
        rgb_pils, front_pil = run_mvprediction(preview_img, remove_bg=remove_bg, seed=args.seed)
        print(f"   [Unique3D] Generated {len(rgb_pils)} views ✓")

    # 4. Reconstruct mesh with ISOMER
    print("   [Unique3D] Running ISOMER reconstruction...")
    new_meshes = geo_reconstruct(
        rgb_pils, None, front_pil,
        do_refine=True,
        predict_normal=True,
        expansion_weight=0.1,
        init_type="std"
    )

    # 5. Post-process vertices (from gradio_3dgen.py)
    vertices = new_meshes.verts_packed()
    vertices = vertices / 2 * 1.35
    vertices[..., [0, 2]] = -vertices[..., [0, 2]]
    new_meshes = Meshes(
        verts=[vertices],
        faces=new_meshes.faces_list(),
        textures=new_meshes.textures,
    )

    # 6. Export GLB
    print("   [Unique3D] Exporting GLB...")
    ret_mesh, _ = save_glb_and_video(
        args.output_dir, new_meshes,
        with_timestamp=False,
        dist=3.5,
        fov_in_degrees=2 / 1.35,
        cam_type="ortho",
        export_video=False,
    )
    
    print(f"   ✅ Mesh saved: {ret_mesh}")


if __name__ == "__main__":
    main()
