import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

from lib.model.RootUNet import UNet
from lib.root_util import get_3d_roots

def draw_points(image: np.ndarray, points: np.ndarray):
    """
    Points are expected to have integer coordinates.
    """
    POINT_COLOR = (255, 255,  0)
    radius = max(1, int(min(image.shape[:2]) * 0.004))
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), radius, POINT_COLOR, -1)
    return image

def save_roots_obj(mesh_path, verts):
    file = open(mesh_path, 'w')    
    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    file.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True, help='test data path')
    parser.add_argument('--ckpt_path', type=str, required=True, help='checkpoint path' )
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id for cuda')
    args = parser.parse_args()
    
    SCALE = [0.5]

    if int(args.gpu_id) >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')

    for idx, revision_data_name in enumerate(os.listdir(args.test_data)):
        item = revision_data_name
        print(f"Processing {item} ......")
        
        img_path = os.path.join(args.test_data, item, f"orient_2d/{item}_thresh_50.png")
        mask_fn = os.path.join(args.test_data, item, f"{item}_matte.png")
        face_mesh_path = f"{args.test_data}/{item}/mesh/{item}.obj"    

        model = UNet(n_channels=3, n_out=1)
        model.to(device)
        # model.load_state_dict(torch.load(args.ckpt_path, device)['model'])
        model.load_state_dict(torch.load(args.ckpt_path, device))
        model.eval()
        image_errs = []

        trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


        img = Image.open(img_path).convert('RGB')
        height, width = img.size[0], img.size[1]
        
        mask = cv2.imread(mask_fn)
        mask_nonzero = np.nonzero(mask)
    
        max_mask_x, max_mask_y = np.max(mask_nonzero[0]), np.max(mask_nonzero[1])
        min_mask_x, min_mask_y = np.min(mask_nonzero[0]), np.min(mask_nonzero[1])


        inputs = trans(img)[None, ...].to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.no_grad():
            outputs = model(inputs)
            density = outputs[0, 0].cpu().numpy()
        
   
        if max(width, height) > 1000:
            density_thresh = 1 /  max(height, width) * SCALE[0] 
        else:
            density_thresh = 1 /  max(height, width) 
        
        point_idx = np.asarray(np.where(density > density_thresh)).T

        from sklearn.cluster import dbscan, k_means

        cluster_idx, labels = dbscan(point_idx, eps=1.5, min_samples=5)   
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("n_cluster: ", n_clusters_)     # 509

        cen, y, interia = k_means(point_idx, n_clusters_)
        roots_2d_list = []
        for i in range(len(cen)):
            if cen[i, 0] > max_mask_x or cen[i, 0] < min_mask_x or cen[i, 1] > max_mask_y or cen[i, 1] < min_mask_y :
                continue
            roots_2d_list.append(cen[i, :])


        roots_2d = np.array(roots_2d_list)
        
        #### debug ####
        # real_img = cv2.imread(img_path)
        # draw_points(real_img, roots_2d[:, ::-1])
        # ## For debug
        # cv2.imwrite(f"test_roots_{item}.png", real_img)
        

        """Considering scale and translation for real head mesh."""
        camera_path = os.path.join(args.test_data, item, "camera", f"projection.pth")
        transform_path = os.path.join(args.test_data, item, "transform", "transform.pth")
        transform = torch.load(transform_path)
        camera = torch.load(camera_path)

        from psbody.mesh import Mesh
        face_mesh = Mesh(filename=face_mesh_path)
        face_verts = face_mesh.v
        
        camera = camera[0]
        verts = camera[:3, :3] @ face_verts.T  + camera[:3, 3][..., None]
        verts = verts.T

        # real_img_cp = real_img.copy()
        # draw_points(real_img_cp, verts[:, :2])
        # cv2.imwrite(f"test_root_finder_{item}.png", real_img_cp)

        roots_3d_path = os.path.join(args.test_data, item, "roots_3d")
        if not os.path.exists(roots_3d_path):
            os.makedirs(roots_3d_path)

        roots_3d = get_3d_roots(roots_2d, face_mesh_path, camera_path, is_head=False)

        roots = camera[:3, :3] @ roots_3d.T  + camera[:3, 3][..., None]
        roots = roots.T
        # draw_points(real_img_cp, roots[:, :2])
        # cv2.imwrite("test_root_finder_accurate.png", real_img_cp)

        """transform roots_3d"""
        s1_norm = transform["norm_lmk"][0].numpy()
        t1_norm = transform["norm_lmk"][1].numpy()
        s2_norm = transform["align_lmk"][0].numpy()
        t2_norm = transform["align_lmk"][1].numpy()

        """transform camera """
        total_scale = s1_norm * s2_norm
        total_trans = t1_norm * s2_norm + t2_norm

        camera[:3, :3] = camera[:3, :3] * 1 / total_scale
        camera[:3, 3] -= camera[:3, :3] @ total_trans 
        # print("camera: ", camera)

        camera = camera[None]
        align_camera_path = os.path.join(args.test_data, item, "camera", f"{item}.pth")
        torch.save(camera, align_camera_path)
        
        align_mesh_path = os.path.join(args.test_data, item, f"align_mesh/align_lmk_{item}.obj")
        align_mesh = Mesh(filename=align_mesh_path)
        align_verts = align_mesh.v
        
        camera = camera[0]
        align_verts = camera[:3, :3] @ align_verts.T  + camera[:3, 3][..., None]
        align_verts = align_verts.T

        # Debug
        # real_img_cp = real_img.copy()
        # print("real_img_cp: ", real_img_cp.shape)
        # draw_points(real_img_cp, align_verts[:, :2])
        # cv2.imwrite(f"test_align_{item}.png", real_img_cp)

        roots_3d = roots_3d * transform["norm_lmk"][0].numpy() + transform["norm_lmk"][1].numpy()
        roots_3d = roots_3d * transform["align_lmk"][0].numpy() + transform["align_lmk"][1].numpy()

        save_roots_obj(f"{roots_3d_path}/{item}.obj", roots_3d)
