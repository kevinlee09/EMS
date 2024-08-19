import torch
import torch.nn as nn
import torch.nn.functional as F


class netD_resnet(nn.Module):
    def __init__(self):
        super(netD_resnet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size = 3, stride = 2, padding = 0),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace = True),
            # state size. 1 x 6 x 6
            nn.Conv2d(256, 1, kernel_size = 1, stride = 1, padding = 0),
            nn.Sigmoid())
        
    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

    def backward_D(self):
    ## Synthetic
    # stop backprop to netB by detaching
        _feat_s = self.syn_pool.query(self.feat_syn[self.cfg['DA_LAYER']].detach().cpu())
        pred_syn = self.netD(_feat_s.to(self.device))
        self.loss_D_syn = self.criterionGAN(pred_syn, False)

        ## Real
        _feat_r = self.real_pool.query(self.feat_real[self.cfg['DA_LAYER']].detach().cpu())
        pred_real = self.netD(_feat_r.to(self.device))
        self.loss_D_real = self.criterionGAN(pred_real, True)

        ## Combined
        self.loss_D = (self.loss_D_syn + self.loss_D_real) * 0.5
        self.loss_D.backward()