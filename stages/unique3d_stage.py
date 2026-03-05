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
    click.echo("   ✓ Unique3D already set up")


def run_unique3d(input_image_path: str, output_dir: str):
    """
    Runs Unique3D end-to-end on a SINGLE input image.
    
    Unique3D internally:
      1. Generates 4 orthographic multiview images
      2. Estimates normal maps
      3. Runs ISOMER reconstruction → OBJ + texture
    
    Args:
        input_image_path: Path to a single product photo (any angle, white bg preferred)
        output_dir: Where to save the mesh output
    
    Returns:
        dict with mesh_path, texture_map
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
    ]
    
    click.echo(f"   [Unique3D] Running full pipeline (multiview + normals + ISOMER)...")
    proc = subprocess.run(cmd, cwd=UNIQUE3D_DIR, env=env, capture_output=True, text=True)
    
    if proc.returncode != 0:
        click.echo(f"   ⚠ Unique3D failed:\n{proc.stderr}")
        raise RuntimeError(f"Unique3D failed (exit {proc.returncode})")
    
    # Find generated mesh
    obj_files = list(isomer_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"Unique3D finished but no .obj found in {isomer_dir}")
    
    mesh_path = str(obj_files[0])
    
    # Look for texture (PNG next to OBJ, or _albedo, or texture_*)
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
