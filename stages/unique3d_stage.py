import os
import sys
import subprocess
import click
from pathlib import Path

UNIQUE3D_DIR = "/workspace/Unique3D"


def ensure_setup():
    """Verify Unique3D is cloned and ready."""
    if not os.path.isdir(UNIQUE3D_DIR):
        raise RuntimeError(
            f"Unique3D not found at {UNIQUE3D_DIR}. "
            "Run setup_runpod.sh first."
        )
    # Check checkpoints exist
    ckpt_dir = os.path.join(UNIQUE3D_DIR, "ckpt")
    if not os.path.isdir(ckpt_dir) or len(os.listdir(ckpt_dir)) < 3:
        raise RuntimeError(
            "Unique3D checkpoints not found. Re-run setup_runpod.sh "
            "or download manually from HuggingFace."
        )
    click.echo("   ✓ Unique3D already set up")


def run_unique3d(input_image_path: str, output_dir: str, seed: int = 42):
    """
    Runs Unique3D end-to-end on a SINGLE input image.
    
    Unique3D internally:
      1. Loads all models (multiview diffusion, normal estimation, ISOMER)
      2. Upscales input if needed
      3. Generates multiview images
      4. Reconstructs mesh with ISOMER
      5. Exports GLB
    """
    ensure_setup()
    
    isomer_dir = Path(output_dir) / "unique3d_output"
    isomer_dir.mkdir(parents=True, exist_ok=True)
    
    wrapper_script = str(Path(__file__).parent / "run_unique3d_wrapper.py")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = UNIQUE3D_DIR + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, wrapper_script,
        "--input_image", str(input_image_path),
        "--output_dir", str(isomer_dir),
        "--seed", str(seed),
    ]
    
    click.echo(f"   [Unique3D] Running full pipeline (multiview + normals + ISOMER)...")
    
    # Don't capture output — let it stream to console for progress visibility
    proc = subprocess.run(cmd, cwd=UNIQUE3D_DIR, env=env)
    
    if proc.returncode != 0:
        raise RuntimeError(f"Unique3D failed (exit {proc.returncode})")
    
    # Find generated mesh — save_glb_and_video outputs .glb files
    glb_files = list(isomer_dir.glob("*.glb"))
    obj_files = list(isomer_dir.glob("*.obj"))
    
    mesh_path = None
    if glb_files:
        mesh_path = str(glb_files[0])
    elif obj_files:
        mesh_path = str(obj_files[0])
    else:
        raise FileNotFoundError(f"Unique3D finished but no mesh found in {isomer_dir}")
    
    # Look for texture
    tex_candidates = list(isomer_dir.glob("*.png"))
    tex_path = None
    for t in tex_candidates:
        if "normal" not in t.name.lower():
            tex_path = str(t)
            break
    
    return {
        "mesh_path": mesh_path,
        "texture_map": tex_path,
    }
