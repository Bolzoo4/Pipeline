#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v4 — RunPod GPU Setup (InstantMesh)
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v4 — InstantMesh Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. Check GPU ───
echo ""
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ─── 2. System deps ───
echo "📦 Installing system deps..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "✅ System deps installed"

# ─── 3. Clone InstantMesh ───
echo ""
if [ -d "/workspace/InstantMesh" ]; then
    echo "✅ InstantMesh already cloned"
else
    echo "📥 Cloning InstantMesh..."
    cd /workspace
    git clone https://github.com/TencentARC/InstantMesh.git
    echo "✅ InstantMesh cloned"
fi

# ─── 4. Install dependencies ───
echo ""
echo "📦 Installing dependencies..."

# Pipeline base deps
pip install click trimesh pygltflib Pillow opencv-python-headless numpy 2>&1 | tail -1

# InstantMesh deps
cd /workspace/InstantMesh
pip install -r requirements.txt 2>&1 | tail -5

# Fix blinker if needed
rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true

echo "✅ Dependencies installed"

# ─── 5. Pre-download models ───
echo ""
echo "📥 Pre-downloading models (this takes a few minutes)..."

python3 -c "
from huggingface_hub import hf_hub_download

print('  📥 Downloading InstantMesh UNet weights...')
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='diffusion_pytorch_model.bin', repo_type='model')

print('  📥 Downloading InstantMesh LRM weights...')
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='instant_mesh_large.ckpt', repo_type='model')

print('  📥 Downloading Zero123++ model...')
from diffusers import DiffusionPipeline
import torch
pipe = DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2', custom_pipeline='zero123plus', torch_dtype=torch.float16)
del pipe

print('  📥 Downloading rembg model...')
from rembg import new_session
session = new_session('u2net')
del session

print()
print('🎉 All models downloaded!')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  python pipeline.py -i photo.jpg -o ./bundle/ --real -c ring"
echo ""
echo "  # Mock mode (no GPU)"
echo "  python pipeline.py -i photo.jpg -o ./bundle/ --mock -c ring"
echo "═══════════════════════════════════════════════"
