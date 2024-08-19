import os
import sys
import cv2
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms

sys.path.append("external/3DDFA_V2")
sys.path.append("external/face-parsing.PyTorch")

from model import BiSeNet                 # face-parsing.PyTorch   
from TDDFA import TDDFA                   # 3DDFA_V2
from FaceBoxes import FaceBoxes           # 3DDFA_V2

def vis_parsing_maps(img, parsing_anno, stride):
    img = np.array(img)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) #+ 255 Use a black mask

    num_of_class = np.max(vis_parsing_anno)

    for part_i in range(1, num_of_class + 1):
        if part_i in [2, 3]: # Only part of eyebrows is needed
            index = np.where(vis_parsing_anno == part_i)
            vis_parsing_anno_color[index[0], index[1], :] = [255, 255, 255] # Make mask to white
            
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    return vis_parsing_anno_color

def face_parsing(src_img, ckpt='79999_iter.pth', device=None):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    save_pth = os.path.join('assets/face_parsing/checkpoint', ckpt)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        orig_H, orig_W = img.shape[:2]
        image = cv2.resize(img, (512, 512))
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        print(np.unique(parsing))

        parsing_mask = vis_parsing_maps(image, parsing, stride=1)
        parsing_mask_orig = cv2.resize(parsing_mask, (orig_H, orig_W))

    return parsing_mask_orig

def get_parsing_bbox(pars_img, ratio=0.025):
    roi = np.nonzero(pars_img)
    roi_arr = np.asarray(roi)
    
    roi_min = np.min(roi_arr, axis=1)[:2]    
    roi_max = np.max(roi_arr, axis=1)[:2]   

    odd_id_min = np.nonzero(roi_min % 2)[0]
    roi_min[odd_id_min] -= 1    
    odd_id_max = np.nonzero(roi_max % 2)[0]
    roi_max[odd_id_max] += 1

    img_h, img_w = pars_img.shape[:2] 
    img_size = max(img_h, img_w)
    
    bbox_trans = int(img_size * ratio)     

    bbox = (roi_min[1]-bbox_trans, roi_min[0]-bbox_trans, roi_max[1]+bbox_trans, roi_max[0]+bbox_trans)
    return bbox

def pad_eyebrow_img(eyebrow_crop_img):
    crop_H, crop_W = eyebrow_crop_img.shape[:2]  
    
    if max(crop_H, crop_W) > 1500 or  min(crop_H, crop_W) > 500:
        pad_size_H = int((800 - crop_H) / 2)     ## padding，[2000, 800]
        pad_size_W = int((2000 - crop_W) / 2)
    else:
        pad_size_H = int((600 - crop_H) / 2)     ## padding，[1500, 600]
        pad_size_W = int((1500 - crop_W) / 2)

    eyebrow_crop_pad = np.pad(eyebrow_crop_img,((pad_size_H,pad_size_H),(pad_size_W,pad_size_W),(0,0)),'constant',constant_values = 255)
    eyebrow_crop_pad = cv2.resize(eyebrow_crop_pad, (1500, 600))
    return eyebrow_crop_pad


def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]

def get_lmk_2d(img):
    config_path = 'external/3DDFA_V2/configs/mb1_120x120.yml'
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    gpu_mode= True
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()
    dense_flag = False
    cur_img = img
    boxes = face_boxes(img)
    n = len(boxes)
    if n <= 0:
        assert 1/0, "no face detected"
    param_lst, roi_box_lst = tddfa(cur_img, boxes)
    lmk_pts = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0].transpose(1, 0)
    lmk_pts = np.asarray(lmk_pts).astype(np.int32)
    
    return lmk_pts


def get_face_bbox_img(img):
    orig_h, orig_w = img.shape[:2]

    face_boxes = FaceBoxes()
    bboxes = face_boxes(img)
    n = len(bboxes)
    if n <= 0:
        assert 1/0, "no face detected"
    
    bbox = pad_bbox(bboxes[0], (orig_w, orig_h), 0.05)
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    assert face_w == face_h        
    
    print('A face is detected. l: %d, t: %d, r: %d, b: %d'
        % (bbox[0], bbox[1], bbox[2], bbox[3]))

    face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    
    return face_img, bbox

def draw_points(image: np.ndarray, points: np.ndarray):
    """
    Points are expected to have integer coordinates.
    """
    POINT_COLOR = (255, 0,  0)
    radius = max(1, int(min(image.shape[:2]) * 0.004))
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), radius, POINT_COLOR, -1)
    return image


if __name__ == "__main__":
    img = cv2.imread("./data_input/000213.png")
    lmk_pts = get_lmk_2d(img)
    draw_points(img, lmk_pts)
    cv2.imwrite("data_out/000213_lmk_img.jpg", img)

    face_img, _ = get_face_bbox_img(img)
    cv2.imwrite("data_out/000213_face.jpg", face_img)
    
    parsing_mask = face_parsing(face_img)
    cv2.imwrite("data_out/000213_parsing_mask_test.png", parsing_mask)
    
    pars_bbox = get_parsing_bbox(parsing_mask)
        
    eyebrow_crop = face_img[pars_bbox[1]:pars_bbox[3], pars_bbox[0]:pars_bbox[2], :]
    cv2.imwrite("data_out/000213_eyebrow_crop_test.png", eyebrow_crop)
    
    eyebrow_crop_pad = pad_eyebrow_img(eyebrow_crop)
    cv2.imwrite("data_out/000213_crop_pad.png", eyebrow_crop_pad)

    

    
    