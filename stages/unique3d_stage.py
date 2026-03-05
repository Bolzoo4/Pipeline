import os
import sys
import subprocess
import click
from pathlib import Path

UNIQUE3D_DIR = "/workspace/Unique3D"

def run_unique3d_isomer(grid_path, output_dir):
    """
    Runs Unique3D ISOMER reconstruction on a 4-view grid.
    Expected grid: 2x2 or 4x1 with Front, Right, Back, Left views.
    """
    lrm_dir = Path(output_dir) / "unique3d" / "isomer_output"
    lrm_dir.mkdir(parents=True, exist_ok=True)
    
    # We need a wrapper script inside Unique3D to call their internal APIs
    wrapper_script = Path(__file__).parent / "run_isomer_wrapper.py"
    
    # We'll create this wrapper script below
    env = os.environ.copy()
    env["PYTHONPATH"] = UNIQUE3D_DIR + ":" + env.get("PYTHONPATH", "")
    
    cmd = [
        sys.executable, str(wrapper_script),
        "--input_grid", str(grid_path),
        "--output_dir", str(lrm_dir),
        "--ckpt_path", "/workspace/Unique3D/ckpt"
    ]
    
    click.echo(f"   [Unique3D] Running ISOMER reconstruction...")
    proc = subprocess.run(cmd, cwd=UNIQUE3D_DIR, env=env, capture_output=True, text=True)
    
    if proc.returncode != 0:
        click.echo(f"   ⚠ Unique3D failed:\n{proc.stderr}")
        raise RuntimeError(f"Unique3D Reconstruction failed (exit {proc.returncode})")
    
    # Unique3D usually saves as 'mesh.obj' or named after input
    # We'll find it in the output dir
    obj_files = list(lrm_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError("Unique3D finished but no .obj was found!")
    
    mesh_path = str(obj_files[0])
    tex_path = str(mesh_path.replace(".obj", ".png"))
    
    return {
        "mesh_path": mesh_path,
        "texture_map": tex_path if os.path.exists(tex_path) else None
    }
