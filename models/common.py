import torch
from torch import nn
from torch.nn import functional as F

from .BaseModels import BaseModule, Conv_block


class SpatialChannelSqueezeExcitation(BaseModule):
    # https://arxiv.org/abs/1709.01507
    # https://arxiv.org/pdf/1803.02579v1.pdf
    def __init__(self, in_channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SpatialChannelSqueezeExcitation, self).__init__()
        linear_nodes = max(in_channel // reduction, 4)  # avoid only 1 node case
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excite = nn.Sequential(
            # check the paper for the number 16 in reduction. It is selected by experiment.
            nn.Linear(in_channel, linear_nodes),
            activation,
            nn.Linear(linear_nodes, in_channel),
            nn.Sigmoid()
        )
        self.spatial_excite = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # channel
        channel = self.avg_pool(x).view(b, c)
        # channel = F.avg_pool2d(x, kernel_size=(h,w)).view(b,c)
        cSE = self.channel_excite(channel).view(b, c, 1, 1)
        x_cSE = torch.mul(x, cSE)

        # spatial
        sSE = self.spatial_excite(x)
        x_sSE = torch.mul(x, sSE)
        return torch.add(x_cSE, x_sSE)


def add_SCSE_block(model_block, in_channel=None):
    if in_channel is None:
        # the first layer is assumed to be conv
        in_channel = model_block[0].out_channels
    model_block.add_module("SCSE", SpatialChannelSqueezeExcitation(in_channel))


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
    # receptive fiedl block  https://arxiv.org/abs/1711.07767 with some changes
    # reference: https://github.com/ansleliu/LightNet/blob/master/modules/rfblock.py
    # https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_mobile.py
    def __init__(self, in_channel, out_channel, activation, add_sece=False):
        super(RFB, self).__init__()
        asp_rate = [5, 17, 29]
        self.act_fn = activation
        # self.act_fn = activation
        self.input_down_channel = nn.Sequential(
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=True, BN=True, activation=activation))

        rfb_linear_conv = [nn.Conv2d(out_channel * 4, out_channel, kernel_size=1, bias=True)]
        if add_sece:
            rfb_linear_conv.append(SpatialChannelSqueezeExcitation(in_channel=out_channel, activation=activation))
        self.rfb_linear_conv = nn.Sequential(*rfb_linear_conv)

        self.rfb = nn.Sequential(
            self.make_pooling_branch(in_channel, out_channel, out_channel, conv_kernel=1,
                                     astro_rate=1, activation=activation, half_conv=False),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=3,
                                     astro_rate=asp_rate[0], activation=activation, half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=5,
                                     astro_rate=asp_rate[1], activation=activation, half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=7,
                                     astro_rate=asp_rate[2], activation=activation, half_conv=True)
        )

    @staticmethod
    def make_pooling_branch(in_channel, mid_channel, out_channel, conv_kernel, astro_rate, activation, half_conv=False):
        # from the paper: we use a 1 x n plus an nx1 conv-layer to take place of the original nxn convlayer
        # similar to EffNet style
        if half_conv:
            m = nn.Sequential(
                *Conv_block(in_channel, mid_channel, kernel_size=1, padding=0,
                            bias=True, BN=True, activation=activation),
                *Conv_block(mid_channel, 3 * mid_channel // 2, kernel_size=(1, conv_kernel),
                            padding=(0, (conv_kernel - 1) // 2), bias=False, BN=True, activation=None),
                *Conv_block(3 * mid_channel // 2, out_channel, kernel_size=(conv_kernel, 1),
                            padding=((conv_kernel - 1) // 2, 0), bias=False, BN=True, activation=None),
                *Conv_block(out_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=True, BN=True, activation=activation, groups=out_channel))
        else:
            m = nn.Sequential(
                *Conv_block(in_channel, out_channel, kernel_size=conv_kernel, padding=(conv_kernel - 1) // 2,
                            bias=True, BN=True, activation=activation),
                *Conv_block(out_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=True, BN=True, activation=activation, groups=out_channel)
            )

        return m

    def forward(self, x):
        # feature pooling
        rfb_pool = [layer(x) for layer in self.rfb.children()]
        rfb_pool = torch.cat(rfb_pool, dim=1)
        rfb_pool = self.rfb_linear_conv(rfb_pool)

        # skip connection
        resi = self.input_down_channel(x)
        return self.act_fn(rfb_pool + resi)
