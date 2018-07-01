import torch
from torch import nn
from torch.nn import functional as F

from .BaseModels import BaseModule, Conv_block


class ASP(BaseModule):
    # Atrous Spatial Pyramid Pooling with Image Pooling
    # add Vortex pooling https://arxiv.org/pdf/1804.06242v1.pdf
    def __init__(self, in_channel=256, out_channel=256):
        super(ASP, self).__init__()
        asp_rate = [5, 17, 29]
        self.asp = nn.Sequential(
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=1, stride=1, padding=0,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[0], stride=1, padding=(asp_rate[0] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[0],
                                      dilation=asp_rate[0], bias=False, BN=True, activation=None)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[1], stride=1, padding=(asp_rate[1] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[1],
                                      dilation=asp_rate[1], bias=False, BN=True, activation=None)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[2], stride=1, padding=(asp_rate[2] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[2],
                                      dilation=asp_rate[2], bias=False, BN=True, activation=None))
        )

        """ To see why adding gobal average, please refer to 3.1 Global Context in https://www.cs.unc.edu/~wliu/papers/parsenet.pdf """
        self.img_pooling_1 = nn.AdaptiveAvgPool2d(1)
        self.img_pooling_2 = nn.Sequential(
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=False, BN=True, activation=None))

        # self.initialize_weights()
        # self.selu_init_params()

    def forward(self, x):
        avg_pool = self.img_pooling_1(x)
        avg_pool = F.upsample(avg_pool, size=x.shape[2:], mode='bilinear')
        avg_pool = [x, self.img_pooling_2(avg_pool)]
        asp_pool = [layer(x) for layer in self.asp.children()]
        return torch.cat(avg_pool + asp_pool, dim=1)


class RFB(BaseModule):
    #
    def __init__(self):
        pass
