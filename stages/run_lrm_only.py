import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import v2
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

# This script must be run from inside the InstantMesh directory
# so these local imports work
sys.path.append(os.getcwd())

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_obj_with_mtl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('input_grid', type=str, help='Path to the 640x960 stitched multiview image.')
    parser.add_argument('output_path', type=str, help='Output directory for the mesh.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

    # Load LRM model
    print('   [LRM] Loading reconstruction model...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
    
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()

    # Load input grid image (640x960)
    print(f'   [LRM] Loading multiview grid: {args.input_grid}')
    input_image = Image.open(args.input_grid).convert("RGB")
    images_np = np.asarray(input_image, dtype=np.float32) / 255.0
    
    # Process into (6, 3, 320, 320)
    images_tensor = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float() # (3, 960, 640)
    images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    images_tensor = images_tensor.unsqueeze(0).to(device) # (1, 6, 3, 320, 320)
    
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    
    # Ensure outputs dir exists
    os.makedirs(args.output_path, exist_ok=True)
    obj_name = os.path.basename(args.input_grid).split('.')[0]
    mesh_path = os.path.join(args.output_path, f'{obj_name}.obj')

    # Reconstruct
    print('   [LRM] Running 3D reconstruction...')
    with torch.no_grad():
        planes = model.forward_planes(images_tensor, input_cameras)
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=True,
            **infer_config,
        )
        
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        save_obj_with_mtl(
            vertices.data.cpu().numpy(),
            uvs.data.cpu().numpy(),
            faces.data.cpu().numpy(),
            mesh_tex_idx.data.cpu().numpy(),
            tex_map.permute(1, 2, 0).data.cpu().numpy(),
            mesh_path,
        )
        
    print(f'   [LRM] Mesh saved to {mesh_path}')

if __name__ == "__main__":
    main()
