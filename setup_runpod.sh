#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v5.0 — RunPod One-Shot Setup (Unique3D)
#
# IMPORTANT: This script keeps the RunPod template's torch version
# (2.4.1+cu124) and pins all other packages to compatible versions.
#
# Run: cd /workspace && bash pipeline/setup_runpod.sh
# Requires: ≥24GB VRAM GPU (A6000/A100), ~15GB disk
# Time: ~15-20 minutes
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
echo "  ✅ OK"

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

# ─── 4. Check base torch (DO NOT UPGRADE) ───
echo ""
echo "🔍 [4/10] Base torch version (keeping RunPod template)..."
python3 -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.version.cuda}')"

# ─── 5. Install ALL Python packages with PINNED versions ───
#    These versions are tested to work together with torch 2.4.1
echo ""
echo "📦 [5/10] Installing pinned Python packages (~3 min)..."
pip install --upgrade pip 2>&1 | tail -1

# Core diffusion stack — pinned for torch 2.4.x compatibility
pip install \
    'diffusers==0.27.2' \
    'transformers==4.44.0' \
    'huggingface_hub==0.25.2' \
    'accelerate==0.33.0' \
    'tokenizers>=0.19,<1.0' \
    'datasets>=2.18,<3.0' \
    'peft==0.12.0' \
    'safetensors>=0.4' \
    2>&1 | tail -3

# Other Unique3D deps (pure Python, no compilation)
pip install \
    'omegaconf>=2.3.0' \
    'fire' \
    'jaxtyping' \
    'numba' \
    'numpy' \
    'opencv-python-headless' \
    'Pillow' \
    'pygltflib' \
    'pymeshlab>=2023.12' \
    'rembg' \
    'onnxruntime-gpu' \
    'tqdm' \
    'trimesh' \
    'typeguard' \
    'wandb' \
    'xatlas' \
    2>&1 | tail -3

# Pipeline deps
pip install 'click' 'google-genai' 2>&1 | tail -1

# Fix blinker
pip install --force-reinstall 'blinker>=1.6' 2>&1 | tail -1

echo "  ✅ Python packages OK"

# ─── 6. Ninja ───
echo ""
echo "📦 [6/10] Ninja..."
pip install ninja 2>&1 | tail -1
echo "  ✅ OK"

# ─── 7. Compile nvdiffrast (against torch 2.4.1) ───
echo ""
echo "🔨 [7/10] Compiling nvdiffrast... (1-2 min)"
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation --force-reinstall
echo "  ✅ nvdiffrast OK"

# ─── 8. Compile pytorch3d (against torch 2.4.1) ───
echo ""
echo "🔨 [8/10] Compiling pytorch3d... (5-8 min, this is the slowest step)"
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
echo "  ✅ pytorch3d OK"

# ─── 9. Compile torch_scatter (against torch 2.4.1) ───
echo ""
echo "🔨 [9/10] Compiling torch_scatter... (1-3 min)"
pip install torch_scatter --no-build-isolation
echo "  ✅ torch_scatter OK"

# ─── 10. Vertex AI Environment ───
echo ""
echo "🌐 [10/10] Vertex AI environment..."
export GOOGLE_CLOUD_PROJECT="gen-lang-client-0752039042"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"
echo "  ✅ Set"

# ─── Verify ALL imports ───
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

for mod, ver in [('diffusers', True), ('transformers', True), ('accelerate', True), ('peft', True), ('huggingface_hub', True)]:
    try:
        m = __import__(mod)
        print(f'  ✓ {mod} {m.__version__}')
    except Exception as e:
        errors.append(f'{mod}: {e}')

for mod in ['nvdiffrast', 'pytorch3d', 'torch_scatter', 'trimesh', 'xatlas', 'rembg', 'omegaconf']:
    try:
        __import__(mod)
        print(f'  ✓ {mod}')
    except Exception as e:
        errors.append(f'{mod}: {e}')

# Test diffusers can actually import controlnet (the critical pipeline)
try:
    from diffusers import StableDiffusionControlNetPipeline
    print(f'  ✓ diffusers.StableDiffusionControlNetPipeline')
except Exception as e:
    errors.append(f'diffusers controlnet import: {e}')

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

print('  Downloading: img2mvimg, image2normal, controlnet-tile...')
try:
    snapshot_download(
        repo_id='Wuvin/Unique3D',
        repo_type='space',
        allow_patterns='ckpt/*',
        local_dir='/workspace/Unique3D',
    )
    print('  ✅ Checkpoints downloaded!')
except Exception as e:
    print(f'  ⚠ Failed: {e}')
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
echo "📋 Run:"
echo "  cd /workspace/pipeline"
echo "  python3 pipeline.py -i test_ring.webp -o /workspace/output --real -c ring"
echo "═══════════════════════════════════════════════"
