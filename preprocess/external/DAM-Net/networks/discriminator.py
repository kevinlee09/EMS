import  torch.nn as nn
import  torch

class netD_resnet(nn.Module):
    def __init__(self):
        super(netD_resnet, self).__init__()
        # todo
        self.main = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 6 x 6
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        # 4 512 16 16
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def get_discriminator():
    discriminator = netD_resnet()
    return discriminator


def get_GANLoss():
    criterionGAN = GANLoss()
    return criterionGAN
