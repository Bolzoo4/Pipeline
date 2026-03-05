#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v5.0 — RunPod One-Shot Setup (Unique3D)
#
# Run this ONCE after starting a fresh RunPod instance:
#   cd /workspace && bash pipeline/setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.4.1 template (Python 3.11, CUDA 12.4)
# Requires: ~6GB disk space for models, ≥24GB VRAM GPU (A6000/A100)
# ═══════════════════════════════════════════════════════════════

# Nano Banana Pro (Gemini 3) / Vertex AI Environment
export GOOGLE_CLOUD_PROJECT="virtual-try-on-488619"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v5.0 (Unique3D) — Setup"
echo "═══════════════════════════════════════════════"

# ─── 1. GPU check ───
echo ""
echo "🔍 GPU check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ─── 2. System deps ───
echo "📦 System deps..."
apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1
echo "  ✅ System deps OK"

# ─── 3. Fix blinker (breaks pip on some RunPod images) ───
echo ""
echo "🔧 Fixing blinker..."
rm -rf /usr/lib/python3/dist-packages/blinker* /usr/lib/python3.11/dist-packages/blinker* 2>/dev/null || true
echo "  ✅ blinker cleaned"

# ─── 4. Upgrade pip ───
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "  ✅ pip upgraded"

# ─── 5. Pipeline core deps ───
echo ""
echo "📦 Pipeline core deps..."
pip install click Pillow numpy opencv-python-headless trimesh pygltflib onnxruntime-gpu google-genai 2>&1 | tail -1
echo "  ✅ Pipeline deps OK"

# ─── 6. Clone Unique3D ───
echo ""
if [ -d "/workspace/Unique3D" ]; then
    echo "✅ Unique3D already cloned"
else
    echo "📥 Cloning Unique3D..."
    cd /workspace
    git clone https://github.com/AiuniAI/Unique3D.git
    echo "  ✅ Unique3D cloned"
fi
cd /workspace

# ─── 7. Install ninja (needed for nvdiffrast) ───
echo ""
echo "📦 Installing ninja..."
pip install ninja 2>&1 | tail -1
echo "  ✅ ninja OK"

# ─── 8. Install nvdiffrast (needs special flag) ───
echo ""
echo "📦 Installing nvdiffrast..."
pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation 2>&1 | tail -1
echo "  ✅ nvdiffrast OK"

# ─── 9. Install Unique3D deps ───
echo ""
echo "📦 Installing Unique3D deps..."
cd /workspace/Unique3D
# Skip some problematic or already installed versions
grep -vE "nvdiffrast|ninja|torch|torchvision" requirements.txt | pip install -r /dev/stdin 2>&1 | tail -3
pip install xatlas trimesh rembg[gpu,pillow] 2>&1 | tail -1
echo "  ✅ Unique3D deps OK"

# ─── 10. Pin compatible versions ───
echo ""
echo "📦 Pinning compatible versions..."
pip install \
    'huggingface_hub==0.21.4' \
    'transformers==4.37.2' \
    'diffusers==0.27.2' \
    'accelerate==0.28.0' \
    'onnxruntime-gpu' \
    2>&1 | tail -1
echo "  ✅ Versions pinned"

# ─── 11. Verify imports ───
echo ""
echo "🧪 Verifying imports..."
cd /workspace
python3 -c "
import torch, rembg, diffusers, xatlas, trimesh
import nvdiffrast
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
print(f'  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  diffusers:  {diffusers.__version__}')
print('  ✅ All imports OK')
"

# ─── 12. Pre-download models (Unique3D) ───
echo ""
read -p "📥 Pre-download Unique3D models (~4GB)? [Y/n] " answer
answer=${answer:-Y}
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading models (this takes 3-5 minutes)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os

def download_ckpt(repo, filename, local_dir):
    print(f'  📥 Downloading {filename}...')
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    return path

# Unique3D models
ckpt_dir = '/workspace/Unique3D/ckpt'
os.makedirs(ckpt_dir, exist_ok=True)

# Note: These are example checkpoints, actual repo might have different ones
# We'll try to download the core ones needed for ISOMER
try:
    download_ckpt('Wanshui/Unique3D', 'img_to_mv.pth', ckpt_dir)
    download_ckpt('Wanshui/Unique3D', 'mv_to_mesh.pth', ckpt_dir)
except Exception as e:
    print(f'  ⚠ Some models failed to download: {e}')
    print('  ℹ You might need to download them manually or via the official unique3d script.')

print('  📥 rembg u2net...')
from rembg import new_session
s = new_session('u2net')
del s

print('  ✅ All models downloaded!')
"
fi

# ─── Done ───
echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "📋 Quick start (with manual grid):"
echo "  cd /workspace/pipeline"
echo "  python pipeline.py -i test_ring.jpg -o /workspace/output/ --real --grid griglia_corretta.png -c ring"
echo ""
echo "📋 Quick start (auto-generate views with Gemini):"
echo "  cd /workspace/pipeline"
echo "  python pipeline.py -i test_ring.jpg -o /workspace/output/ --real -c ring"
echo ""
echo "📋 Options:"
echo "  --seed 42        Random seed"
echo "  --mock           Skip GPU, test flow only"
echo "  --grid FILE      Use pre-made 2x2 multiview grid"
echo "═══════════════════════════════════════════════"
