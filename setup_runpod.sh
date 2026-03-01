#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline — RunPod GPU Setup Script
# ═══════════════════════════════════════════════════════════════
#
# How to use:
# 1. Go to runpod.io → Pods → Create Pod
# 2. Select a GPU: RTX A4000 ($0.20/hr) or RTX 3090 ($0.31/hr) 
#    or A100 ($1.64/hr) — any NVIDIA GPU with ≥16GB VRAM works
# 3. Select template: "RunPod Pytorch 2.1" (or any with CUDA 12.x)
# 4. Start the pod
# 5. Connect via SSH or Web Terminal
# 6. Upload this script + your jewelry images
# 7. Run: bash setup_runpod.sh
# 8. Run: python pipeline.py --input YOUR_IMAGE.jpg --output ./bundle/ --real
#
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

# ─── 2. Install dependencies ───
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    transformers>=4.40.0 \
    diffusers>=0.28.0 \
    accelerate>=0.28.0 \
    safetensors>=0.4.0 \
    opencv-python-headless>=4.9.0 \
    Pillow>=10.2.0 \
    numpy>=1.26.0 \
    click>=8.1.0

echo "✅ Dependencies installed"

# ─── 3. Pre-download models (so pipeline doesn't wait) ───
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
echo "  python pipeline.py --input PHOTO.jpg --output ./bundle/ --real -c ring"
echo ""
echo "Options:"
echo "  -c ring|earring|necklace|bracelet|watch"
echo "  --quality 90   (WebP quality, 0-100)"
echo ""
echo "Example with multiple images:"
echo "  for img in images/*.jpg; do"
echo "    name=\$(basename \$img .jpg)"
echo "    python pipeline.py -i \$img -o ./bundles/\$name/ --real -c ring"
echo "  done"
echo "═══════════════════════════════════════════════"
