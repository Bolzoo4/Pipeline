#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline — RunPod GPU Setup Script
# ═══════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline — RunPod GPU Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. Check GPU ───
echo ""
echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ─── 2. Install dependencies via requirements file ───
echo "📦 Installing Python dependencies..."
pip install -r requirements-gpu.txt

echo "✅ Dependencies installed"

# ─── 3. Pre-download models ───
echo ""
echo "📥 Pre-downloading AI models (this takes ~5-10 min first time)..."

python3 -c "
from transformers import pipeline, VitMatteForImageMatting, VitMatteImageProcessor
from diffusers import MarigoldNormalsPipeline
import torch

print('  📥 Downloading SAM2 (segmentation)...')
seg = pipeline('mask-generation', model='facebook/sam2-hiera-large', device='cpu', torch_dtype=torch.float32)
del seg
print('  ✅ SAM2 ready')

print('  📥 Downloading ViTMatte (alpha matting)...')
proc = VitMatteImageProcessor.from_pretrained('hustvl/vitmatte-small-composition-1k')
model = VitMatteForImageMatting.from_pretrained('hustvl/vitmatte-small-composition-1k')
del proc, model
print('  ✅ ViTMatte ready')

print('  📥 Downloading Marigold (normal estimation)...')
pipe = MarigoldNormalsPipeline.from_pretrained('prs-eth/marigold-normals-lcm-v0-1', variant='fp16', torch_dtype=torch.float16)
del pipe
print('  ✅ Marigold ready')

print()
print('🎉 All models downloaded and cached!')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup complete! Run the pipeline with:"
echo ""
echo "  python pipeline.py -i PHOTO.jpg -o ./bundle/ --real -c ring"
echo ""
echo "═══════════════════════════════════════════════"
