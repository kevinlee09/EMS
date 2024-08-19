import torch
import torch.nn as nn
from tqdm import tqdm

class AlignLmk(nn.Module):
    def __init__(self, src_lmk_3d, ref_lmk_3d, device=None):
        super(AlignLmk, self).__init__()
        self.src_lmk = torch.from_numpy(src_lmk_3d).float().to(device).T
        self.ref_lmk = torch.from_numpy(ref_lmk_3d).float().to(device).T

        self.register_buffer('scale', nn.Parameter(torch.Tensor([1])))
        self.register_buffer('trans', nn.Parameter(torch.Tensor([[0.5],[0.5],[0.5]])))

        self.register_parameter('scale_dis', nn.Parameter(torch.ones_like(self.scale)))
        self.register_parameter('trans_dis', nn.Parameter(torch.ones_like(self.trans)))
    
    def get_transform(self):
        scale = self.scale + self.scale_dis
        offset = self.trans + self.trans_dis
        
        scale = scale.detach().cpu()
        offset = offset.squeeze().detach().cpu()
        transform = (scale, offset)
        
        return transform 
        
    def forward(self):
        trans_src_lmk = self.src_lmk * (self.scale + self.scale_dis) + self.trans + self.trans_dis
        loss_pts = torch.abs(trans_src_lmk.T - self.ref_lmk.T) 
        loss_pts[:10] = loss_pts[:10]*2 # enhance eyebrow
        loss_pts[19:29] = loss_pts[19:29]*5 
        loss = torch.sum(loss_pts)

        return loss


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def opt_lmk_align(src_3d_lmk, ref_3d_lmk, device=None):
    
    lr = 5e-2
    num_epoch = 2500
    schedule = [2000,2250]

    ## load_mesh and transform to camera space
    if device is not None:
        lmk_opt = AlignLmk(src_3d_lmk, ref_3d_lmk, device).to(device)
    else: 
        lmk_opt = AlignLmk(src_3d_lmk, ref_3d_lmk)
        
    optimizer = torch.optim.Adam(lmk_opt.parameters(), lr, betas=(0.5, 0.99))
    loop = tqdm(range(0, num_epoch))

    for epoch in loop:
        loss = lmk_opt.forward()
        loop.set_description('Loss: {:04f}'.format(loss.item()))#
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, lr, schedule, 0.5)

    return lmk_opt


    