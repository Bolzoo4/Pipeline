#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gioielli Pipeline v4.1 — RunPod One-Shot Setup
#
# Run this ONCE after starting a fresh RunPod instance:
#   cd /workspace && bash pipeline/setup_runpod.sh
#
# Tested on: RunPod PyTorch 2.4.1 template (Python 3.11, CUDA 12.4)
# Requires: ~6GB disk space for models
# ═══════════════════════════════════════════════════════════════

# Nano Banana Pro (Gemini 3) / Vertex AI Environment
export GOOGLE_CLOUD_PROJECT="virtual-try-on-488619"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="True"

set -e

echo "═══════════════════════════════════════════════"
echo "💎 Gioielli Pipeline v4.1 — Setup"
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

# ─── 6. Clone InstantMesh ───
echo ""
if [ -d "/workspace/InstantMesh" ]; then
    echo "✅ InstantMesh already cloned"
else
    echo "📥 Cloning InstantMesh..."
    cd /workspace
    git clone https://github.com/TencentARC/InstantMesh.git
    echo "  ✅ InstantMesh cloned"
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

# ─── 9. Install InstantMesh deps (skip nvdiffrast line) ───
echo ""
echo "📦 Installing InstantMesh deps..."
cd /workspace/InstantMesh
grep -v nvdiffrast requirements.txt | pip install -r /dev/stdin 2>&1 | tail -3
echo "  ✅ InstantMesh deps OK"

# ─── 10. Pin compatible versions (avoid huggingface/transformers conflicts) ───
echo ""
echo "📦 Pinning compatible versions..."
pip install \
    'huggingface_hub==0.21.4' \
    'transformers==4.37.2' \
    'diffusers==0.27.2' \
    'accelerate==0.28.0' \
    2>&1 | tail -1
echo "  ✅ Versions pinned"

# ─── 11. Verify imports ───
echo ""
echo "🧪 Verifying imports..."
cd /workspace
python3 -c "
import torch, rembg, diffusers, xatlas, trimesh
import nvdiffrast
from einops import rearrange
from omegaconf import OmegaConf
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.cuda.is_available()}')
print(f'  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'  diffusers:  {diffusers.__version__}')
print('  ✅ All imports OK')
"

# ─── 12. Pre-download models (optional, saves time on first run) ───
echo ""
read -p "📥 Pre-download models (~4GB)? [Y/n] " answer
answer=${answer:-Y}
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "📥 Downloading models (this takes 3-5 minutes)..."
    python3 -c "
from huggingface_hub import hf_hub_download

print('  📥 InstantMesh UNet...')
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='diffusion_pytorch_model.bin', repo_type='model')

print('  📥 InstantMesh LRM (large)...')
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='instant_mesh_large.ckpt', repo_type='model')

print('  📥 Zero123++ v1.2...')
try:
    from diffusers import DiffusionPipeline
    import torch
    pipe = DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2', custom_pipeline='zero123plus', torch_dtype=torch.float16)
    del pipe
except Exception as e:
    print(f'  ⚠ Zero123++ download skipped (not critical for NanoBanana): {e}')

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
echo "📋 Quick start:"
echo "  cd /workspace/pipeline"
echo "  python pipeline.py -i /workspace/test_ring.jpg -o /workspace/output/ --real -c ring"
echo ""
echo "📋 Options:"
echo "  --steps 100     Diffusion steps (default 100, more = better)"
echo "  --seed 42       Random seed"
echo "  --mock           Skip GPU, test flow only"
echo "═══════════════════════════════════════════════"
