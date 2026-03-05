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


def run_unique3d(input_image_path: str = None, grid_path: str = None, output_dir: str = "", seed: int = 42):
    """
    Runs Unique3D on a SINGLE input image OR a pre-generated 2x2 grid.
    
    Unique3D internally:
      1. Loads all models (multiview diffusion, normal estimation, ISOMER)
      2. Generates multiview images (skipped if grid_path provided)
      3. Reconstructs mesh with ISOMER
      4. Exports GLB
    """
    ensure_setup()
    
    isomer_dir = Path(output_dir) / "unique3d_output"
    isomer_dir.mkdir(parents=True, exist_ok=True)
    
    wrapper_script = str(Path(__file__).parent / "run_unique3d_wrapper.py")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = UNIQUE3D_DIR + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, wrapper_script,
        "--output_dir", str(isomer_dir),
        "--seed", str(seed),
    ]
    
    if grid_path:
        cmd.extend(["--grid", str(grid_path)])
    elif input_image_path:
        cmd.extend(["--input_image", str(input_image_path)])
    else:
        raise ValueError("Must provide either input_image_path or grid_path to Unique3D")
    
    click.echo(f"   [Unique3D] Running full pipeline (multiview + normals + ISOMER)...")
    
    # Don't capture output — let it stream to console for progress visibility
    proc = subprocess.run(cmd, cwd=UNIQUE3D_DIR, env=env)
    
    if proc.returncode != 0:
        raise RuntimeError(f"Unique3D failed (exit {proc.returncode})")
    
    # Unique3D's save_glb_and_video uses the provided path as a *base name*, 
    # so it creates `unique3d_output.glb` instead of putting it inside a directory.
    expected_glb = Path(output_dir) / "unique3d_output.glb"
    
    if not expected_glb.exists():
        raise FileNotFoundError(f"Unique3D finished but no mesh found at {expected_glb}")
        
    mesh_path = str(expected_glb)
    
    # Unique3D bakes the texture directly into the vertex colors or GLB texture,
    # so there is no separate texture file to return.
    tex_path = None
    
    return {
        "mesh_path": mesh_path,
        "texture_map": tex_path,
    }
