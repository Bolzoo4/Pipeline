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
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_remove_bg", action="store_true", help="Skip background removal")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load all models
    print("   [Unique3D] Loading models...")
    model_zoo.init_models()
    print("   [Unique3D] Models loaded ✓")

    # 2. Load & upscale input image
    print(f"   [Unique3D] Loading input: {args.input_image}")
    preview_img = Image.open(args.input_image).convert("RGB")
    
    if preview_img.size[0] <= 512:
        print("   [Unique3D] Upscaling input (<=512px)...")
        preview_img = run_sr_fast([preview_img])[0]

    # 3. Generate multiview images
    print("   [Unique3D] Generating multiview images...")
    remove_bg = not args.no_remove_bg
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
