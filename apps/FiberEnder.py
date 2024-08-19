import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
import point_cloud_utils as pcu

from lib.model import *
from lib.hair_util import *
from lib.options import BaseOptions

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# get options
opt = BaseOptions().parse()

def load_lenNet(net_path, cuda):
    # create net
    net = Len_Cls_Net(opt).to(device=cuda)
    print('Using Network: ', net.name)

    # load checkpoints
    print('loading for net G ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def load_img(img_path):
    orien2d = Image.open(img_path).convert('RGB')
    img_to_tensor = transforms.Compose([
            transforms.Resize([600, 1500]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    orien2d = img_to_tensor(orien2d).float()
    return orien2d

def pred_label_for_eyeb(net, cuda, data):
    image_tensor = data['img'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    point_tensor = data['strands_3d'].to(device=cuda).unsqueeze(0)
    orien_tensor = data['orien_3d'].to(device=cuda).unsqueeze(0)
    length_tensor = data['length'].to(device=cuda).unsqueeze(0)
    eyeb_pred_labels, _, _, _, _, _, _, _ = net.forward(image_tensor, point_tensor, orien_tensor, calib_tensor, length_tensor, is_train=False)
    eyeb_pred_labels = torch.reshape(eyeb_pred_labels, (point_tensor.shape[1], point_tensor.shape[2])) # N x num_pts_per_strand

    return np.array(eyeb_pred_labels.cpu())

def interp_strand(strand, num=20):
    # Cumulative Euclidean distance between successive polygon points.
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(strand, axis=0) ** 2).sum(axis=1))])
    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), num)
    # interpolate x and y coordinates
    pts_interp = np.c_[
    np.interp(d_sampled, d, strand[:, 0]),
    np.interp(d_sampled, d, strand[:, 1]),
    np.interp(d_sampled, d, strand[:, 2]),
    ]

    return pts_interp


def query_id_stop(labels):
    id_stop_list = []
    for idx in range(0, labels.shape[0]):
        label = list(labels[idx])
        label[0] = 1
        try:
            # print(label)
            idx_stop = label.index(0)
        except:
            idx_stop = -1
        id_stop_list.append(idx_stop)
    
    return np.array(id_stop_list)

def interpolate_no_stop(id_stop_list, roots):
    is_stop_list = id_stop_list < 0
    num_no_stop = np.sum(is_stop_list)
    print(num_no_stop)
    if num_no_stop==0:
        return id_stop_list
    
    no_stop_roots = roots[is_stop_list]
    no_stop_query_ids = np.where(is_stop_list)

    can_stop_roots = roots[~is_stop_list]
    can_stop_lengths = id_stop_list[~is_stop_list]

    K = 10
    _, corrs_a_to_b = pcu.k_nearest_neighbors(no_stop_roots, can_stop_roots, K)

    interp_lengths = np.mean(can_stop_lengths[corrs_a_to_b.reshape(-1)].reshape(-1, K), axis=1)
    id_stop_list[no_stop_query_ids] = interp_lengths

    return id_stop_list


def cut_eyebrow_by_label(eyeb_strands, id_stop_list, num_per_strand):
    strands_list = []
    for idx in range(0, eyeb_strands.shape[0]):
        strand = eyeb_strands[idx]
        idx_stop = id_stop_list[idx]
        strand_cut = strand[:idx_stop+1,:]
        strand_cut_interp = interp_strand(strand_cut, num_per_strand)
        strands_list.append(strand_cut_interp)
    
    return np.array(strands_list)

def test(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    root = opt.test_data
    items = sorted(os.listdir(root))
    print(items)

    len_net_path = opt.ckpt_path
    len_net = load_lenNet(len_net_path, cuda)

    with torch.no_grad():
        for item in tqdm(items):
            # Test synthesized data
            eyeb_obj_path = os.path.join(root, item, "recon_fiber_wo_end", item + "_fiber.obj")
            orien_map_path = os.path.join(root, item, "orient_2d", item + '_thresh_50.png') 
            calib_path = os.path.join(root, item, "camera" , item + '.pth')

            eyeb_obj_cut_root = os.path.join(root, item, 'recon_fiber_final')
        
            if not os.path.exists(eyeb_obj_cut_root):
                os.makedirs(eyeb_obj_cut_root)
                
            # label_save_path = os.path.join(label_save_root, item + '_fiber_final.npy')
            eyeb_save_path = os.path.join(eyeb_obj_cut_root, item + '_fiber_final.obj')
            eyeb_npy_save_path = os.path.join(eyeb_obj_cut_root, item + '_fiber_final.npy')
            
            # Load strands and orien
            eyeb_verts, _ = load_obj_with_line_elements(eyeb_obj_path)
            N = 13
            B = int(eyeb_verts.shape[0] / N)
            eyeb_strands = eyeb_verts.reshape((B, N, 3))
            eyeb_roots = eyeb_strands[:, 0, :]
            eyeb_orien = cal_eyeb_orien(eyeb_strands)
            eyeb_strands_tensor = torch.Tensor(eyeb_strands).cuda()
            eyeb_orien_tensor = torch.Tensor(eyeb_orien).cuda()

            # Load calib
            calib = torch.load(calib_path)[0].float().cuda()

            # Load orientation map
            orien_img_tensor = load_img(orien_map_path)

            # Length
            length_tensor = torch.Tensor([N]*B)

            test_data = {'img': orien_img_tensor,
                        'calib': calib,
                        'strands_3d': eyeb_strands_tensor,
                        'orien_3d': eyeb_orien_tensor,
                        'length': length_tensor}
            
            eyeb_pred_labels = pred_label_for_eyeb(len_net, cuda, test_data)
            # np.save(label_save_path, eyeb_pred_labels)
            id_stop_list = interpolate_no_stop(query_id_stop(eyeb_pred_labels), eyeb_roots)
            eyeb_cut_interp = cut_eyebrow_by_label(eyeb_strands, id_stop_list, N)
            np.save(eyeb_npy_save_path, eyeb_cut_interp)
            write_strand2obj(eyeb_save_path, eyeb_cut_interp)


if __name__ == '__main__':
    test(opt)