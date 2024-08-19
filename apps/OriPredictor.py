import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from lib.options import BaseOptions
from lib.hair_util import *
from lib.model import *

from PIL import Image

# get options
opt = BaseOptions().parse()

def load_orien2d(orien2d_path, ):
    orien2d = Image.open(orien2d_path).convert('RGB')
    img_to_tensor = transforms.Compose([
            # transforms.Resize(load_size),
            transforms.Resize([600, 1500]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    orien2d = img_to_tensor(orien2d).float()
    return orien2d

def load_calib(calib_path):
    # loading calibration data
    calib = torch.load(calib_path).float()[0]
    return calib


def load_orienNet(net_path, cuda):
    # create net
    net = HGPIFuNet_orien(opt).to(device=cuda)
    print('Using Network: ', net.name)

    # load checkpoints
    print('loading for net G ...', net_path)
    net.load_state_dict(torch.load(net_path, map_location=cuda))
    net.eval()

    return net

def test(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    
    root = opt.test_data
    items = os.listdir(root)
    print(items)
    
    orien_net_path = opt.ckpt_path
    orien_net = load_orienNet(orien_net_path, cuda)


    with torch.no_grad():
        for item in tqdm(items):
            orien2d_path = os.path.join(root, item, "orient_2d", item + '_thresh_50.png')            
            calib_path = os.path.join(root, item, "camera" , item + '.pth')
            hair_root_path = os.path.join(root, item, "roots_3d", item + '.obj')
    
            orien2d = load_orien2d(orien2d_path)
            calib = load_calib(calib_path)

            test_data = {'img': orien2d, 'calib':calib}

            strands_path = '%s/%s/%s/%s.obj' % (
                        root, item, 'recon_fiber_wo_end', item + "_fiber")
            export_hair_real(orien_net, cuda, test_data, strands_path, hair_root_path, opt)

if __name__ == '__main__':
    test(opt)