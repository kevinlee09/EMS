import sys
import toml
import torch
import numpy as np
from torch.nn import functional as F

sys.path.append("external/DAM-Net")
import utils
import networks
from utils import CONFIG

def single_inference(model, image_dict, device=None):
    with torch.no_grad():
        image, trimap = image_dict['image'], image_dict['trimap']
        alpha_shape = image_dict['alpha_shape']
        image = image.to(device)
        trimap = trimap.to(device)
        alpha_pred, _, _ = model(image, trimap)

        if CONFIG.model.trimap_channel == 3:
            trimap_argmax = trimap.argmax(dim=1, keepdim=True)

        alpha_pred[trimap_argmax == 2] = 1
        alpha_pred[trimap_argmax == 0] = 0

        h, w = alpha_shape
        test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        test_pred = test_pred.astype(np.uint8)
        test_pred = test_pred[32:h + 32, 32:w + 32]

        return test_pred


def generator_tensor_dict(image):
    trimap = np.full_like(image[:, :, 0], 128)
    sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape}

    # reshape
    h, w = sample["alpha_shape"]

    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32, 32), (32, 32), (0, 0)), mode="reflect")
        print("padded_img: ", padded_image.shape)
        padded_trimap = np.pad(sample['trimap'], ((32, 32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32, pad_h + 32), (32, pad_w + 32), (0, 0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32, pad_h + 32), (32, pad_w + 32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # convert GBR images to RGB
    image, trimap = sample['image'][:, :, ::-1], sample['trimap']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1
    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    print("sample img: ", sample["image"].shape)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    if CONFIG.model.trimap_channel == 3:
        sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
    elif CONFIG.model.trimap_channel == 1:
        sample['trimap'] = sample['trimap'][None, ...].float()
    else:
        raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample

def matting_eyebrow(img, Model="DAM-Net", device=None):
    config = f"external/{Model}/config/{Model}.toml"
    with open(config) as f:
        utils.load_config(toml.load(f))
    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.to(device)

    # load checkpoint
    checkpoint = torch.load(f"assets/{Model}/checkpoint/best_model.pth")
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

     # inference
    model = model.eval()
    image_dict = generator_tensor_dict(img)
    pred = single_inference(model, image_dict, device=device)
    return pred

def get_matte_brow(orig_img, matte):
    mask = matte[..., None] / 255.0
    matt_img = orig_img * mask \
                    + (1 - mask) * np.full(np.asarray(orig_img).shape, 255.0)

    return matt_img

def get_orient_brow(orient_img_exr, matte):
    orient_img = orient_img_exr * 255.0
    mask = matte[..., None] / 255.0
    orient_brow_img = orient_img * mask  \
                    + (1 - mask) * np.full(np.asarray(orient_img).shape, 0.0)
    return orient_brow_img