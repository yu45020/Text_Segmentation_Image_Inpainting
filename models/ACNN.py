"""
Reference:
Atrous Convolutional Neural Network (ACNN) for Semantic Image Segmentation
    with full-scale Feature Maps
By: Xiao-Yun Zhou, Jian-Qing Zheng, Guang-Zhong Yang, (2019)
arXiv:1901.09203 [cs, stat], 10 Feb, 2019
"""
from torch.nn.parameter import Parameter

from models.BaseModels import BaseModule
from torch import nn
# from models.group_batch_norm import GroupBatchNorm2D
from models.common import SpatialChannelSqueezeExcitation
from models.group_batch_norm import GroupNorm2D


class tofp32(nn.Module):
    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def GN_convert_float(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    elif isinstance(module, nn.modules.GroupNorm):
        module.float()
        module.weight = Parameter(module.weight.half())
        module.bias = Parameter(module.bias.half())

    for child in module.children():
        GN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    return nn.Sequential(tofp16(), GN_convert_float(network.half()))


class Atrous_Res_Block(BaseModule):
    # identity mapping
    def __init__(self, channels=64, atrous_rates=(1, 3),
                 act_fn=nn.LeakyReLU(0.1), GN_groups=4):
        super(Atrous_Res_Block, self).__init__()

        self.act_fn = act_fn
        self.body = self.make_body(channels, atrous_rates, GN_groups)

    def make_body(self, channels, atrous_rates, num_groups):
        m = []
        for i in atrous_rates:
            m.append(nn.Sequential(
                GroupNorm2D(num_groups=num_groups, num_channels=channels),
                # nn.GroupNorm(num_groups, num_channels=channels),
                self.act_fn,
                nn.Conv2d(channels, channels, kernel_size=3, padding=(2 * i + 1) // 2, dilation=i, bias=False),
            ))
        return nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return x + out


class ACNN(BaseModule):
    def __init__(self, middle_channels=64, atrous_rates=[1, 3], act_fn=nn.LeakyReLU(0.1), num_atrous_blocks=32):
        super(ACNN, self).__init__()
        # for 256x256 images, 3x3, atrous: (1,3), we need 32 blocks to maximize receptive filed and coverage
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=3, bias=True, padding=1)
        self.out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=3, bias=True, padding=1)
        self.act_fn = act_fn
        self.atrous_blocks = nn.Sequential(
            *[Atrous_Res_Block(middle_channels, atrous_rates, act_fn, GN_groups=middle_channels) for
              _ in range(num_atrous_blocks)])

    def forward(self, x):
        features = self.in_conv(x)
        features = self.atrous_blocks(features)
        out = self.out_conv(features)
        return out


class Atrous_Res_Block_SCSE(Atrous_Res_Block):
    def make_body(self, channels, atrous_rates, num_groups):
        m = []
        for i in atrous_rates:
            m.append(nn.Sequential(
                GroupNorm2D(num_groups=num_groups, num_channels=channels),
                # nn.GroupNorm(num_groups, num_channels=channels),
                self.act_fn,
                nn.Conv2d(channels, channels, kernel_size=3, padding=(2 * i + 1) // 2, dilation=i, bias=False),
            ))
        m.append(SpatialChannelSqueezeExcitation(channels, reduction=16))
        return nn.Sequential(*m)


class ACNN_SCSE(ACNN):
    def __init__(self, middle_channels=64, atrous_rates=[1, 3],
                 act_fn=nn.LeakyReLU(0.1), num_atrous_blocks=32):
        super(ACNN_SCSE, self).__init__(middle_channels=middle_channels, atrous_rates=atrous_rates,
                                        act_fn=act_fn, num_atrous_blocks=num_atrous_blocks)

        self.atrous_blocks = nn.Sequential(*[Atrous_Res_Block_SCSE(middle_channels, atrous_rates, act_fn, GN_groups=4)
                                             for _ in range(num_atrous_blocks)])


class ACNN_V2(ACNN):
    def __init__(self, middle_channels=64, atrous_rates=[1, 3],
                 act_fn=nn.LeakyReLU(0.1), num_atrous_blocks=32):
        super(ACNN_V2, self).__init__(middle_channels=middle_channels, atrous_rates=atrous_rates,
                                      act_fn=act_fn, num_atrous_blocks=num_atrous_blocks)
        atrous_block = [Atrous_Res_Block(middle_channels, atrous_rates, act_fn, GN_groups=4)
                        for _ in range(num_atrous_blocks // 2)]

        atrous_block.extend([Atrous_Res_Block(middle_channels, list(reversed(atrous_rates)), act_fn, GN_groups=4)
                             for _ in range(num_atrous_blocks // 2)])
        self.atrous_blocks = nn.Sequential(*atrous_block)


class ACNN_Reverse(BaseModule):
    def __init__(self, middle_channels=64, atrous_rates=[1, 3], act_fn=nn.LeakyReLU(0.1), num_atrous_blocks=32):
        super(ACNN_Reverse, self).__init__()
        # for 256x256 images, 3x3, atrous: (1,3), we need 32 blocks to maximize receptive filed and coverage
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=3, bias=True, padding=1)
        self.out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=3, bias=True, padding=1)
        self.act_fn = act_fn

        atrous_blocks = []
        for _ in range(num_atrous_blocks):
            atrous_blocks.append(Atrous_Res_Block(middle_channels, atrous_rates, act_fn, GN_groups=middle_channels))
            atrous_rates = list(reversed(atrous_rates))
        self.atrous_blocks = nn.Sequential(*atrous_blocks)

    def forward(self, x):
        features = self.in_conv(x)
        features = self.atrous_blocks(features)
        out = self.out_conv(features)
        return out


class ACNN_V3(BaseModule):
    def __init__(self,
                 middle_channels=64,
                 atrous_rate_list=[[1, 3], [1, 3]],
                 act_fn=nn.LeakyReLU(0.1)):
        super(ACNN_V3, self).__init__()
        # for 256x256 images, 3x3, atrous: (1,3), we need 32 blocks to maximize receptive filed and coverage
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=3, bias=True, padding=1)
        self.out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=3, bias=True, padding=1)
        self.act_fn = act_fn

        atrous_blocks = [Atrous_Res_Block(middle_channels, j, act_fn, GN_groups=middle_channels)
                         for i in atrous_rate_list for j in i]

        self.atrous_blocks = nn.Sequential(*atrous_blocks)

    def forward(self, x):
        features = self.in_conv(x)
        features = self.atrous_blocks(features)
        out = self.out_conv(features)
        return out


class ACNN_Shrink(BaseModule):
    def __init__(self, middle_channels=64, atrous_rates=[1, 3], act_fn=nn.LeakyReLU(0.1), num_atrous_blocks=32):
        super(ACNN_Shrink, self).__init__()
        # for 256x256 images, 3x3, atrous: (1,3), we need 32 blocks to maximize receptive filed and coverage
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=3, bias=True, padding=1,
                                 stride=2)
        self.out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=3, bias=True, padding=1)
        self.act_fn = act_fn
        self.atrous_blocks = nn.Sequential(
            *[Atrous_Res_Block(middle_channels, atrous_rates, act_fn, GN_groups=middle_channels) for
              _ in range(num_atrous_blocks)])

    def forward(self, x):
        features = self.in_conv(x)
        features = self.atrous_blocks(features)
        out = self.out_conv(features)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out


class ACNN_V3_Shrink(BaseModule):
    def __init__(self,
                 middle_channels=64,
                 atrous_rate_list=[[1, 3], [1, 3]],
                 act_fn=nn.LeakyReLU(0.1)):
        super(ACNN_V3_Shrink, self).__init__()
        # for 256x256 images, 3x3, atrous: (1,3), we need 32 blocks to maximize receptive filed and coverage
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=middle_channels, kernel_size=3, bias=True, padding=1,
                                 stride=2)
        self.out_conv = nn.Conv2d(in_channels=middle_channels, out_channels=1, kernel_size=3, bias=True, padding=1)
        self.act_fn = act_fn

        atrous_blocks = [Atrous_Res_Block(middle_channels, j, act_fn, GN_groups=middle_channels)
                         for i in atrous_rate_list for j in i]

        self.atrous_blocks = nn.Sequential(*atrous_blocks)

    def forward(self, x):
        features = self.in_conv(x)
        features = self.atrous_blocks(features)
        out = self.out_conv(features)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out
