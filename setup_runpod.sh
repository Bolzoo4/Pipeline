#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v5.0 — RunPod One-Shot Setup (Unique3D)
#
# Run this ONCE after starting a fresh RunPod instance:
#   cd /workspace && bash pipeline/setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.4.1 template (Python 3.11, CUDA 12.x)
# Requires: ~15GB disk for models+deps, ≥24GB VRAM GPU (A6000/A100)
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v5.0 (Unique3D) — Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. GPU check ───
echo ""
echo "🔍 GPU check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU found!"

# ─── 2. System deps ───
echo ""
echo "📦 System deps..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libopengl0 > /dev/null 2>&1
echo "  ✅ System deps OK"

# ─── 3. Fix blinker ───
echo ""
echo "🔧 Fixing blinker..."
pip install --force-reinstall 'blinker>=1.6' 2>&1 | tail -1
echo "  ✅ blinker cleaned"

# ─── 4. Clone Unique3D ───
echo ""
if [ -d "/workspace/Unique3D" ]; then
    echo "✓ Unique3D already cloned"
else
    echo "📥 Cloning Unique3D..."
    cd /workspace
    git clone https://github.com/AiuniAI/Unique3D.git
    echo "  ✅ Unique3D cloned"
fi
cd /workspace

# ─── 5. Install Unique3D requirements (this upgrades torch!) ───
echo ""
echo "📦 Installing Unique3D Python deps (this upgrades torch to 2.10+)..."
cd /workspace/Unique3D
# Install everything from requirements.txt except ninja/nvdiffrast (we handle those separately)
grep -vE "nvdiffrast|ninja" requirements.txt | pip install -r /dev/stdin 2>&1 | tail -5
echo "  ✅ Unique3D Python deps OK"

# ─── 6. Install pipeline-specific deps ───
echo ""
echo "📦 Installing pipeline deps..."
pip install click opencv-python-headless google-genai pygltflib xatlas trimesh rembg 2>&1 | tail -3
echo "  ✅ Pipeline deps OK"

# ─── 7. Compile ninja (needed by nvdiffrast + pytorch3d) ───
echo ""
echo "📦 Installing ninja..."
pip install ninja 2>&1 | tail -1
echo "  ✅ ninja OK"

# ─── 8. Compile nvdiffrast AGAINST NEW TORCH ───
echo ""
echo "📦 Compiling nvdiffrast (against current torch)..."
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation --force-reinstall 2>&1 | tail -3
echo "  ✅ nvdiffrast OK"

# ─── 9. Compile pytorch3d AGAINST NEW TORCH ───
echo ""
echo "📦 Compiling pytorch3d (takes ~5 min)..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation 2>&1 | tail -3
echo "  ✅ pytorch3d OK"

# ─── 10. Install torch_scatter AGAINST NEW TORCH ───
echo ""
echo "📦 Installing torch_scatter..."
pip install torch_scatter --no-build-isolation 2>&1 | tail -3
echo "  ✅ torch_scatter OK"

# ─── 11. Vertex AI Environment ───
echo ""
echo "🌐 Setting Vertex AI environment..."
export GOOGLE_CLOUD_PROJECT="gen-lang-client-0752039042"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"
echo "  ✅ Vertex AI configured"

# ─── 12. Verify imports ───
echo ""
echo "🧪 Verifying core imports..."
cd /workspace
python3 -c "
import torch
import diffusers
import transformers
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  diffusers:    {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')

import nvdiffrast
print(f'  nvdiffrast:   OK')

import pytorch3d
print(f'  pytorch3d:    OK')

import torch_scatter
print(f'  torch_scatter: OK')

print('  ✅ All imports OK')
"

# ─── 13. Pre-download Unique3D models ───
echo ""
read -p "📥 Pre-download Unique3D models (~8GB)? [Y/n] " answer
answer=${answer:-Y}
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading Unique3D models from HuggingFace Spaces..."
    python3 -c "
from huggingface_hub import snapshot_download
import os

ckpt_dir = '/workspace/Unique3D/ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

print('  📥 Downloading all Unique3D checkpoints...')
print('     (img2mvimg, image2normal, controlnet-tile, realesrgan-x4.onnx)')
try:
    snapshot_download(
        repo_id='Wuvin/Unique3D',
        repo_type='space',
        allow_patterns='ckpt/*',
        local_dir='/workspace/Unique3D',
    )
    print('  ✅ Unique3D checkpoints downloaded!')
except Exception as e:
    print(f'  ⚠ Download failed: {e}')
    print('  ℹ You can manually download from:')
    print('    https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt')

print('  📥 rembg u2net...')
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
echo "📋 Quick start:"
echo "  cd /workspace/pipeline"
echo "  python3 pipeline.py -i test_ring.webp -o /workspace/output --real -c ring"
echo ""
echo "📋 Options:"
echo "  --seed 42        Random seed"
echo "  --mock           Skip GPU, test flow only"
echo "═══════════════════════════════════════════════"
