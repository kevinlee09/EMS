import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "external/Detailed3DFace"))

from options import Options
from pix2pixHD.models import create_model
from bilinear_model import BilinearModel
from utils import color_transfer, tensor2im, subdiv, dpmap2verts

def main():
    opt = Options().parse()
    img_dirs = []
    img_names = []
    for data_name in os.listdir(opt.input):
        for name in os.listdir(os.path.join(opt.input, data_name)):
            img_dirs.append(os.path.join(opt.input, data_name, name))
        
    for img_dir in img_dirs:
        if os.path.basename(img_dir) != "image":
            continue
        for name in os.listdir(img_dir):
            if any(name.endswith(extension) for extension in
                ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']):
                img_names.append(os.path.join(img_dir, name))
    
    predef_dir = os.path.join("assets/face_recon", opt.predef_dir)
    print("predef_dir: ", predef_dir)
    bilinear_model = BilinearModel(predef_dir)

    if opt.render:
        from renderer import MeshRenderer
        renderer = MeshRenderer()

    if opt.name == 'dpmap_rig':
        opt.input_nc = 6
        pos_maps = np.load(f'{predef_dir}/posmaps.npz')
        pos_maps = pos_maps.f.arr_0
        pos_maps = torch.from_numpy(pos_maps).unsqueeze(0)

    dpmap_model = create_model(opt)

    for img_name in img_names:
        base_name = os.path.splitext(img_name)[0].split("/")[-1]
        if not os.path.exists(f'{opt.input}/{base_name}'):
            os.mkdir(f'{opt.input}/{base_name}')
        
        print(f'\nProcessing {base_name}')
  
        ### added by chenghong
        os.makedirs(f'{opt.input}/{base_name}/camera', exist_ok=True)
        os.makedirs(f'{opt.input}/{base_name}/mesh', exist_ok=True)
            
        img = cv2.imread(img_name)
        
        print('Fitting 3DMM Parameters...')
        proj_params, verts = bilinear_model.fit_image(img)

        print('Warping texture...')
        verts_img = bilinear_model.project(verts, *proj_params, keepz=False)
        texture = bilinear_model.get_texture(img, verts_img)
        bilinear_model.save_obj(f'{opt.input}/{base_name}/mesh/{base_name}.obj', verts, f'./{base_name}.jpg', front=True)
        cv2.imwrite(f'{opt.input}/{base_name}/mesh/{base_name}.jpg', texture)

        texture = cv2.resize(texture[600:2500, 1100:3000], (1024, 1024)).astype(np.uint8)

        mask = (255 - cv2.imread(f'{predef_dir}/front_mask.png')[:, :, 0]).astype(bool)
        new_pixels = color_transfer(texture[mask][:, np.newaxis, :])
        texture[mask] = new_pixels[:, 0, :]
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32)
        texture = np.transpose(texture, (2, 0, 1))
        texture = torch.tensor(texture) / 255
        texture = F.normalize(texture, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        texture = torch.unsqueeze(texture, 0)

        print('Generating displacement maps...')
        dpmap_full = np.zeros((4096, 4096), dtype=np.uint16)
        dpmap_full[...] = 32768
        dpmap_full = Image.fromarray(dpmap_full)
        if opt.name == 'dpmap_rig':
            for i in range(20):
                ipt = torch.cat((texture, pos_maps[:, i * 3:i * 3 + 3]), dim=1)
                dpmap = dpmap_model.inference(ipt, torch.tensor(0))
                dpmap = tensor2im(dpmap.detach()[0], size=(1900, 1900))
                dpmap = Image.fromarray(dpmap)
                dpmap_full.paste(dpmap, (1100, 600, 3000, 2500))
                dpmap_full.save(f'{opt.input}/{base_name}/mesh/{base_name}_dpmap_{str(i)}.png')
        else:
            dpmap = dpmap_model.inference(texture, torch.tensor(0))
            dpmap = tensor2im(dpmap.detach()[0], size=(1900, 1900))
            dpmap = Image.fromarray(dpmap)
            dpmap_full.paste(dpmap, (1100, 600, 3000, 2500))
            dpmap_full.save(f'{opt.input}/{base_name}/mesh/{base_name}_dpmap.png')
            if opt.render:
                print('Rendering results...')
                front_verts = verts[bilinear_model.front_verts_indices]
                tris, vert_texcoords = bilinear_model.tris.copy(), bilinear_model.vert_texcoords.copy()
                for _ in range(3):
                    front_verts, tris, vert_texcoords = subdiv(front_verts, tris, vert_texcoords)
                front_verts = dpmap2verts(front_verts, tris, vert_texcoords, dpmap_full)

                verts_img = bilinear_model.project(front_verts, *proj_params, keepz=True)
                renderer.render(verts_img, tris, (img.shape[1], img.shape[0]), f'{opt.input}/{img_name}',
                                f'{opt.input}/{base_name}/mesh/{base_name}_render.jpg')
        
        ### added by chenghong
        np.save(f'{opt.input}/{base_name}/camera/{base_name}_proj_params.npy', proj_params)


if __name__ == '__main__':
    main()
