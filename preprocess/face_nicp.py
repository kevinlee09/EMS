import os
import sys
import json
import torch
import argparse
import numpy as np

from ext_utils import io3d
from ext_utils.transform import *
sys.path.append(os.path.join(os.path.dirname(__file__), "external/bfmface_nicp"))

import render
from bfm_model import load_bfm_model
from landmark import get_mesh_landmark
from nicp import non_rigid_icp_mesh2mesh
from icp_utils import normalize_mesh, batch_vertex_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input dir')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2, use -1 for CPU')
    args = parser.parse_args()

    if int(args.gpu_ids) >= 0:
        device = torch.device(f'cuda:{args.gpu_ids}')
    
    for data_name in os.listdir(args.input):
        meshes = io3d.load_obj_as_mesh(f'{args.input}/{data_name}/mesh/{data_name}.obj', device = device)
        
        with torch.no_grad():
            norm_meshes, norm_param = normalize_mesh(meshes)
            dummy_render = render.create_dummy_render([1, 0, 0], device = device)
            target_lm_index, lm_mask = get_mesh_landmark(norm_meshes, dummy_render)
            bfm_meshes, bfm_lm_index = load_bfm_model(device)
            lm_mask = torch.all(lm_mask, dim = 0)
            bfm_lm_index_m = bfm_lm_index[:, lm_mask]
            target_lm_index_m = target_lm_index[:, lm_mask]

        fine_config = json.load(open('external/bfmface_nicp/config/config.json'))
        registered_mesh = non_rigid_icp_mesh2mesh(bfm_meshes, norm_meshes, bfm_lm_index_m, target_lm_index_m, fine_config, device=device)

            
        orig_mesh = inv_transform_bfm_mesh(registered_mesh, norm_param)
        io3d.save_meshes_as_objs([f'{args.input}/{data_name}/mesh/{data_name}_bfm.obj'], orig_mesh, save_textures = False)

        lmk_3d = batch_vertex_sample(bfm_lm_index, orig_mesh.verts_padded())
        lmk_3d = lmk_3d[0].detach().cpu().numpy()
        print("lmk_3d: ", lmk_3d.shape)

        
        lmk_path = os.path.join(args.input, data_name, "landmark")
        if not os.path.exists(lmk_path):
            os.makedirs(lmk_path)
        
        np.save(f"{lmk_path}/{data_name}_lmk3d.npy", lmk_3d)
