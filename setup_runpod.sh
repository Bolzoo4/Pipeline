#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline — RunPod GPU Setup (Multi-View 3D)
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v3 — Multi-View 3D Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. Check GPU ───
echo ""
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ─── 2. Install system dependencies ───
echo "📦 Installing system deps..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "✅ System deps installed"

# ─── 3. Install Python dependencies ───
echo "📦 Installing Python dependencies..."
pip install -r requirements-gpu.txt
echo "✅ Python deps installed"

# ─── 4. Pre-download models ───
echo ""
echo "📥 Pre-downloading AI models..."

python3 -c "
import torch

print('  📥 Downloading Zero123++ v1.2 (multi-view generation)...')
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    'sudo-ai/zero123plus-v1.2',
    custom_pipeline='sudo-ai/zero123plus-pipeline',
    torch_dtype=torch.float16,
)
del pipe
print('  ✅ Zero123++ ready')

print('  📥 Downloading rembg model (background removal)...')
from rembg import new_session
session = new_session('u2net')
del session
print('  ✅ rembg ready')

print()
print('🎉 All models downloaded!')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "Usage:"
echo "  # Single image → AI multiview → 3D"
echo "  python pipeline.py -i photo.jpg -o ./bundle/ --real -c ring"
echo ""
echo "  # Real multi-view photos"
echo "  python pipeline.py -i views_dir/ -o ./bundle/ --real -c ring --multiview"
echo ""
echo "  # Mock mode (no GPU)"
echo "  python pipeline.py -i photo.jpg -o ./bundle/ --mock -c ring"
echo "═══════════════════════════════════════════════"
