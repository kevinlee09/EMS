import os
import cv2
import torch
import imageio
import argparse
import numpy as np

from ext_utils import io3d

from ext_utils.transform import *
from ext_utils.face_utils import *
from ext_utils.matt_utils import *
from ext_utils.optim_utils import *

imageio.plugins.freeimage.download()  # load exr

def trans_from_cam_crop(trans, crop_bbox):
    crop_x0, crop_y0 = crop_bbox[0], crop_bbox[1]
    trans[0] = trans[0] - crop_x0 
    trans[1] = trans[1] + crop_y0

    return trans

def trans_from_cam_pad(trans, orig_img, pad_img):
    H, W = orig_img.shape[:2]
    H_pad, W_pad = pad_img.shape[:2]
    trans[0] = trans[0] + (W_pad - W)/2
    trans[1] = trans[1] - (H_pad - H)/2

    return trans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input dir')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2, use -1 for CPU')
    args = parser.parse_args()
    
    if int(args.gpu_ids) >= 0:
        device = torch.device(f'cuda:{args.gpu_ids}')
        
    revision_data_dir = args.input
    revision_data_list = sorted(os.listdir(revision_data_dir))

    for idx, revision_data_name in enumerate(revision_data_list):
        item = revision_data_name
        print("item: ", item)
        proj_params_path = f"{revision_data_dir}/{item}/camera/{item}_proj_params.npy"
        face_mesh_path = f"{revision_data_dir}/{item}/mesh/{item}.obj"
        img_path = f"{revision_data_dir}/{item}/image/{item}.jpg"
        if not os.path.exists(img_path):
            img_path = f"{revision_data_dir}/{item}/image/{item}.png"
        
        data_processed = revision_data_dir

        meshes = io3d.load_obj_as_mesh(face_mesh_path)

        ref_lmk3d_pts = np.load("assets/mean_head_3d_lmk.npy")
        src_lmk3d_pts = np.load(f"{data_processed}/{item}/landmark/{item}_lmk3d.npy")

        src_lmk3d_pts, ref_lmk3d_pts, lmk_transform = normalize_lmk(src_lmk3d_pts, ref_lmk3d_pts)

        norm_meshes = transform_mesh(meshes, lmk_transform)

        torch.set_grad_enabled(True)
        lmk_opt = opt_lmk_align(src_lmk3d_pts, ref_lmk3d_pts, device=device)
        align_lmk_transform = lmk_opt.get_transform()

        align_lmk3d_pts = transform_lmk(src_lmk3d_pts, align_lmk_transform)
        align_meshes = transform_mesh(norm_meshes, align_lmk_transform)


        """TODO: create processed folder"""
        align_meshes_dir = os.path.join(data_processed, item, "align_mesh")
        if not os.path.exists(align_meshes_dir):
            os.makedirs(align_meshes_dir)

        io3d.save_meshes_as_objs([os.path.join(align_meshes_dir, f"align_lmk_{item}.obj")], align_meshes, save_textures=True)

        camera_dir = os.path.join(data_processed, item, "camera")
        transform_dir = os.path.join(data_processed, item, "transform")

        if not os.path.exists(camera_dir):
            os.makedirs(camera_dir)

        if not os.path.exists(transform_dir):
            os.makedirs(transform_dir)

        transform = {"norm_lmk": lmk_transform, "align_lmk": align_lmk_transform}
        torch.save(transform, os.path.join(transform_dir, "transform.pth"))
        
        real_img = cv2.imread(img_path)
        H, W = real_img.shape[:2]

        face_img, bbox = get_face_bbox_img(real_img)
        face_img_fn = os.path.join(data_processed, item, os.path.basename(img_path).split(".")[0]+"_face.png")
        cv2.imwrite(face_img_fn, face_img)
        
        parsing_mask = face_parsing(face_img, device=device)
        pars_bbox = get_parsing_bbox(parsing_mask)

        eyebrow_crop = face_img[pars_bbox[1]:pars_bbox[3], pars_bbox[0]:pars_bbox[2], :]
        eye_crop_H, eye_crop_W = eyebrow_crop.shape[:2]


        eyebrow_crop_pad = pad_eyebrow_img(eyebrow_crop)
        eye_crop_pad_H, eye_crop_pad_W = eyebrow_crop_pad.shape[:2]


        matte = matting_eyebrow(eyebrow_crop_pad, device=device)   # [600, 1500]
        matte_img = os.path.join(data_processed, item, os.path.basename(img_path).split(".")[0]+"_matte.png")
        cv2.imwrite(matte_img, matte)

        eyebrow_matt_img = get_matte_brow(eyebrow_crop_pad, matte)
        eyeb_img_fn = os.path.join(data_processed, item, os.path.basename(img_path).split(".")[0]+"_eyeb.png")
        cv2.imwrite(eyeb_img_fn, eyebrow_matt_img)



        """ TODO: merge cpp orient to generate 2d orientation map """
        orient_img_exr_fn = os.path.join(data_processed, item, os.path.basename(img_path).split(".")[0]+"_orient.exr")
        if not os.path.exists(orient_img_exr_fn):
            os.system(f"./orient2d/build/orient2d {eyeb_img_fn} {orient_img_exr_fn}")


        orient_exr = imageio.imread(orient_img_exr_fn)
        ## combine mask and orient
        orient_img = get_orient_brow(orient_exr, matte)

        orient_2d_dir = os.path.join(data_processed, item, "orient_2d")
        if not os.path.exists(orient_2d_dir):
            os.makedirs(orient_2d_dir)
            
        orient_img_png_fn = os.path.join(orient_2d_dir, os.path.basename(img_path).split(".")[0]+".png")
        imageio.imwrite(orient_img_png_fn, np.uint8(orient_img))

        threshold = 50    ## binary mask threshold
        orien_img_nonzero = np.where(orient_img > threshold)
        
        mask_img = np.zeros_like(orient_img).astype(np.uint8)
        mask_img[orien_img_nonzero[0], orien_img_nonzero[1], :] = 255.0
        
        ### For debug hard mask
        # cv2.imwrite(f"{item}_mask.png", mask_img.astype(np.uint8))

        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        mask = np.asarray(mask,np.bool)

        orient_img = orient_exr * mask[..., None] * 255.0

        orien_dir = f"{data_processed}/{item}/orient_2d"
        if not os.path.exists(orien_dir):
            os.makedirs(orien_dir)

        imageio.imwrite(f"{orien_dir}/{item}_thresh_{threshold}.png", orient_img)

        from psbody.mesh import Mesh
        proj_params = np.load(proj_params_path, allow_pickle=True)
        print("face_mesh_path: ", face_mesh_path)
        face_mesh = Mesh(filename=face_mesh_path)
        face_verts = face_mesh.v

        trans = proj_params[2]
        trans = trans_from_cam_crop(trans, bbox)
        trans = trans_from_cam_crop(trans, pars_bbox)
        trans = trans_from_cam_pad(trans, eyebrow_crop, eyebrow_crop_pad)
        
        rot_mat_cv = cv2.Rodrigues(proj_params[0])[0]
        trans_new = np.array([[trans[0]], [trans[1] - H], [0]])

        proj = np.concatenate([rot_mat_cv * proj_params[1], trans_new], axis=-1)

        ## opengl coordinates -> opencv
        rot = np.array([[1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]])

        proj_mod = rot @ proj
        proj_mod = torch.from_numpy(proj_mod)
        proj_save = proj_mod[None]   # [1, 3, 4]
        torch.save( proj_save, f"{camera_dir}/projection.pth")

        ### For debug
        # verts = proj_mod[:3, :3] @ face_verts.T  + proj_mod[:3, 3][..., None]
        # print(verts.shape)
        # print(verts)

        # verts = verts.T

        ### For debug
        # draw_points(face_img, verts[:, :2])
        # cv2.imwrite("test_3dmm_proj_test.png", face_img)
        
        ### For debug
        # draw_points(eyebrow_crop, verts[:, :2])
        # cv2.imwrite("test_3dmm_proj_pars.png", eyebrow_crop)

        ## For debug
        # draw_points(eyebrow_crop_pad, verts[:, :2])
        # cv2.imwrite(f"test_proj_{item}.png", eyebrow_crop_pad)
