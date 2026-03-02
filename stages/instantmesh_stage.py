"""
InstantMesh Stage — End-to-end single image → textured 3D mesh.

Uses TencentARC/InstantMesh which combines:
  1. Zero123++ (with white-bg UNet) → 6 multi-view images
  2. Large Reconstruction Model (LRM) → triplane features
  3. FlexiCubes → textured mesh extraction

This replaces: multiview_gen + reconstruction + texture_baking
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional

INSTANTMESH_DIR = "/workspace/InstantMesh"
INSTANTMESH_REPO = "https://github.com/TencentARC/InstantMesh.git"
CONFIG = "instant-mesh-large"


def ensure_setup():
    """Clone and setup InstantMesh if not already present."""
    if os.path.exists(os.path.join(INSTANTMESH_DIR, "run.py")):
        print("   ✓ InstantMesh already set up")
        return

    print("   ℹ Cloning InstantMesh...")
    subprocess.run(
        ["git", "clone", INSTANTMESH_REPO, INSTANTMESH_DIR],
        check=True,
        capture_output=True,
    )

    # Install only missing deps (avoid reinstalling torch etc)
    print("   ℹ Installing InstantMesh dependencies...")
    req_file = os.path.join(INSTANTMESH_DIR, "requirements.txt")
    subprocess.run(
        ["pip", "install", "--no-deps", "-r", req_file],
        capture_output=True,
    )

    # Install key deps that might be missing
    subprocess.run(
        ["pip", "install", "einops", "omegaconf", "rembg",
         "huggingface_hub", "pytorch-lightning"],
        capture_output=True,
    )

    print("   ✓ InstantMesh setup complete")


def run(image_path: str, output_dir: str, export_texmap: bool = True,
        diffusion_steps: int = 75, seed: int = 42) -> dict:
    """
    Run InstantMesh end-to-end on a single image.

    Args:
        image_path: Path to input image
        output_dir: Where to save outputs
        export_texmap: If True, export mesh with UV texture map
        diffusion_steps: Number of diffusion steps for Zero123++
        seed: Random seed

    Returns:
        dict with paths to generated files
    """
    ensure_setup()

    output_path = os.path.join(output_dir, "instantmesh")
    os.makedirs(output_path, exist_ok=True)

    # Build command
    config_file = os.path.join(INSTANTMESH_DIR, "configs", f"{CONFIG}.yaml")
    cmd = [
        sys.executable, os.path.join(INSTANTMESH_DIR, "run.py"),
        config_file,
        image_path,
        "--output_path", output_path,
        "--diffusion_steps", str(diffusion_steps),
        "--seed", str(seed),
    ]

    if export_texmap:
        cmd.append("--export_texmap")

    print(f"   ℹ Running InstantMesh ({CONFIG})...")
    print(f"   ℹ Command: {' '.join(cmd)}")

    # Run with InstantMesh dir as CWD so relative paths work
    env = os.environ.copy()
    env["PYTHONPATH"] = INSTANTMESH_DIR + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        cwd=INSTANTMESH_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"   ⚠ InstantMesh stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"InstantMesh failed (exit {result.returncode})")

    # Print stdout for debugging
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            print(f"   │ {line.strip()}")

    # Find output files
    mesh_dir = os.path.join(output_path, CONFIG, "meshes")
    image_dir = os.path.join(output_path, CONFIG, "images")

    # Look for mesh file
    mesh_files = list(Path(mesh_dir).glob("*.obj")) if os.path.exists(mesh_dir) else []

    if not mesh_files:
        # Try looking in other locations
        mesh_files = list(Path(output_path).rglob("*.obj"))

    if not mesh_files:
        raise RuntimeError(
            f"No mesh generated. Check output at {output_path}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    mesh_path = str(mesh_files[0])
    print(f"   ✓ Mesh generated: {mesh_path}")

    # Find multiview image
    mv_images = list(Path(image_dir).glob("*.png")) if os.path.exists(image_dir) else []

    # Find texture map if exported
    tex_files = list(Path(mesh_dir).glob("*.png"))

    # Find MTL file
    mtl_files = list(Path(mesh_dir).glob("*.mtl"))

    output = {
        "mesh_path": mesh_path,
        "mesh_dir": mesh_dir,
        "multiview_image": str(mv_images[0]) if mv_images else None,
        "texture_map": str(tex_files[0]) if tex_files else None,
        "mtl_file": str(mtl_files[0]) if mtl_files else None,
    }

    # Log file sizes
    mesh_size = os.path.getsize(mesh_path)
    print(f"   ✓ Mesh size: {mesh_size / 1024:.1f}KB")
    if output["texture_map"]:
        tex_size = os.path.getsize(output["texture_map"])
        print(f"   ✓ Texture map: {tex_size / 1024:.1f}KB")

    return output


def convert_to_glb(mesh_path: str, output_path: str,
                    texture_path: Optional[str] = None) -> str:
    """
    Convert OBJ mesh to GLB format using trimesh.
    """
    import trimesh

    print(f"   ℹ Converting OBJ → GLB...")

    # Load mesh (trimesh will read the MTL and textures automatically)
    mesh = trimesh.load(mesh_path, process=False)

    # If it's a Scene, merge it
    if isinstance(mesh, trimesh.Scene):
        # Get the first mesh from the scene
        geometries = list(mesh.geometry.values())
        if geometries:
            mesh = geometries[0]
        else:
            raise RuntimeError("No geometry found in mesh file")

    # Export as GLB
    glb_data = mesh.export(file_type='glb')
    Path(output_path).write_bytes(glb_data)

    size = os.path.getsize(output_path)
    print(f"   ✓ GLB exported: {output_path} ({size / 1024:.1f}KB)")
    return output_path
