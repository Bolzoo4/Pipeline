#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v5.0 — RunPod One-Shot Setup (Unique3D)
#
# Run this ONCE after starting a fresh RunPod instance:
#   cd /workspace && bash pipeline/setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.4.1 template (Python 3.11, CUDA 12.x)
# Requires: ~15GB disk for models+deps, ≥24GB VRAM GPU (A6000/A100)
# Total setup time: ~15-20 minutes
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v5.0 (Unique3D) — Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. GPU check ───
echo ""
echo "🔍 [1/10] GPU check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU found!"

# ─── 2. System deps ───
echo ""
echo "📦 [2/10] System deps..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libopengl0 > /dev/null 2>&1
echo "  ✅ System deps OK"

# ─── 3. Clone Unique3D ───
echo ""
echo "📥 [3/10] Unique3D repo..."
if [ -d "/workspace/Unique3D" ]; then
    echo "  ✓ Already cloned"
else
    cd /workspace
    git clone https://github.com/AiuniAI/Unique3D.git
    echo "  ✅ Cloned"
fi
cd /workspace

# ─── 4. Install pip packages (NO compilation, pure Python only) ───
#    We install everything that doesn't need to compile against torch FIRST.
#    This is fast (~2 min).
echo ""
echo "📦 [4/10] Python packages (pip-only, no compilation)..."
pip install --upgrade pip 2>&1 | tail -1
pip install \
    'accelerate>=0.28' \
    'datasets' \
    'diffusers>=0.26.3' \
    'fire' \
    'jaxtyping' \
    'numba' \
    'numpy' \
    'omegaconf>=2.3.0' \
    'opencv-python-headless' \
    'peft' \
    'Pillow' \
    'pygltflib' \
    'pymeshlab>=2023.12' \
    'rembg' \
    'tqdm' \
    'transformers' \
    'trimesh' \
    'typeguard' \
    'wandb' \
    'xatlas' \
    'click' \
    'google-genai' \
    2>&1 | tail -5
echo "  ✅ Python packages OK"

# ─── 5. Check torch version (Unique3D deps may have upgraded it) ───
echo ""
echo "🔍 [5/10] Checking torch version..."
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
echo "  torch=${TORCH_VERSION}  cuda=${CUDA_VERSION}"

# ─── 6. Install ninja (needed for compiling C++ extensions) ───
echo ""
echo "📦 [6/10] Ninja build system..."
pip install ninja 2>&1 | tail -1
echo "  ✅ ninja OK"

# ─── 7. Compile nvdiffrast (against current torch, ~1 min) ───
echo ""
echo "🔨 [7/10] Compiling nvdiffrast... (1-2 min)"
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation --force-reinstall
echo "  ✅ nvdiffrast OK"

# ─── 8. Compile pytorch3d (against current torch, ~5 min) ───
echo ""
echo "🔨 [8/10] Compiling pytorch3d... (5-8 min, please wait)"
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
echo "  ✅ pytorch3d OK"

# ─── 9. Compile torch_scatter (against current torch, ~2 min) ───
echo ""
echo "🔨 [9/10] Compiling torch_scatter... (1-2 min)"
pip install torch_scatter --no-build-isolation
echo "  ✅ torch_scatter OK"

# ─── 10. Vertex AI Environment ───
echo ""
echo "🌐 [10/10] Vertex AI environment..."
export GOOGLE_CLOUD_PROJECT="gen-lang-client-0752039042"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"
echo "  ✅ Set"

# ─── Verify everything ───
echo ""
echo "🧪 Verifying all imports..."
cd /workspace
python3 -c "
import sys
errors = []

try:
    import torch
    print(f'  ✓ torch {torch.__version__}  (CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
except Exception as e:
    errors.append(f'torch: {e}')

for mod in ['diffusers', 'transformers', 'accelerate', 'peft']:
    try:
        m = __import__(mod)
        print(f'  ✓ {mod} {m.__version__}')
    except Exception as e:
        errors.append(f'{mod}: {e}')

for mod in ['nvdiffrast', 'pytorch3d', 'torch_scatter', 'trimesh', 'xatlas', 'rembg']:
    try:
        __import__(mod)
        print(f'  ✓ {mod}')
    except Exception as e:
        errors.append(f'{mod}: {e}')

if errors:
    print()
    print('  ⚠ FAILED imports:')
    for e in errors:
        print(f'    ❌ {e}')
    sys.exit(1)
else:
    print('  ✅ All imports OK!')
"

# ─── Download models ───
echo ""
read -p "📥 Download Unique3D models (~8GB)? [Y/n] " answer
answer=${answer:-Y}
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading from HuggingFace Spaces (3-5 min)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

ckpt_dir = '/workspace/Unique3D/ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

print('  Downloading checkpoints: img2mvimg, image2normal, controlnet-tile...')
try:
    snapshot_download(
        repo_id='Wuvin/Unique3D',
        repo_type='space',
        allow_patterns='ckpt/*',
        local_dir='/workspace/Unique3D',
    )
    print('  ✅ Checkpoints downloaded!')
except Exception as e:
    print(f'  ⚠ Download failed: {e}')
    print('  Manual: https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt')

print('  Downloading rembg model...')
from rembg import new_session
s = new_session('u2net')
del s
print('  ✅ All models ready!')
"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "📋 Run the pipeline:"
echo "  cd /workspace/pipeline"
echo "  python3 pipeline.py -i test_ring.webp -o /workspace/output --real -c ring"
echo "═══════════════════════════════════════════════"
