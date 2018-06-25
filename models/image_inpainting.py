import torch
from torch import nn

from .BaseModels import BaseModule


class UNet(BaseModule):
    def __init__(self):
        super(UNet, self).__init__()
        self.act_fn = nn.SELU()

        self.down_block = self.make_down_block()
        self.atrous_transition = self.make_atrous_conv()
        self.up_block = self.make_up_block()
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)
        self.selu_init_params()

    def make_down_block(self):
        # down 8x
        m1 = self.pconv_block(3 + 1, 32, kernel_size=5, padding=2, bias=True,
                              BN=False, activation=self.act_fn, direction='down')
        m2 = self.pconv_block(32, 64, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn, direction='down')
        m3 = self.pconv_block(64, 128, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn, direction='down')

        m = list(map(lambda x: nn.Sequential(*x), [m1, m2, m3]))
        return nn.Sequential(*m)

    def make_atrous_conv(self):
        m0 = self.pconv_block(128, 256, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn, direction='down')
        m1 = self.pconv_block(256, 256, kernel_size=3, padding=2, dilation=2, bias=True,
                              BN=False, activation=self.act_fn)

        m2 = self.pconv_block(256, 256, kernel_size=3, padding=4, bias=True, dilation=4,
                              BN=False, activation=self.act_fn)

        m3 = self.pconv_block(256, 32, kernel_size=1, padding=0, bias=True,
                              BN=False, activation=self.act_fn, direction='up') + \
             self.pconv_block(32, 128, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn)
        m = m0 + m1 + m2 + m3
        return nn.Sequential(*m)

    def make_up_block(self):
        m3 = self.pconv_block(256, 32, kernel_size=1, padding=0, bias=True,
                              BN=False, activation=self.act_fn, direction='up') + \
             self.pconv_block(32, 64, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn)

        m4 = self.pconv_block(128, 32, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn, direction='up')
        m5 = self.pconv_block(64, 64, kernel_size=3, padding=1, bias=True,
                              BN=False, activation=self.act_fn, direction='up')
        m = list(map(lambda x: nn.Sequential(*x), [m3, m4, m5]))
        return nn.Sequential(*m)

    def pconv_block(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,
                    BN=False, activation=None, direction=None):

        m = partial_conv_gated_block(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, BN, activation)

        if direction == 'down':
            m.append(nn.AvgPool2d(2))
        elif direction == 'up':
            m.append(nn.Upsample(scale_factor=2, mode='bilinear'))

        return m

    def forward(self, x):
        out_x = [x]
        for layer in self.down_block.children():
            x = layer(x)
            print(x.size())
            out_x.append(x)

        out_x = out_x
        print('down')
        x = self.atrous_transition(x)
        for layer in self.up_block.children():
            print(x.size())
            x = torch.cat([x, out_x.pop(-1)], dim=1)
            x = layer(x)

        x = self.out_conv(x)
        return x
