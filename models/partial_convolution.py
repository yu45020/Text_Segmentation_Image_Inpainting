# Two types:
# 1. "Hard" gated : use 1/0 to update mask.  Image Inpainting for Irregular Holes Using Partial Convolutions
# 2. "Soft" gated : use sigmoid to update both feature & mask  Free-Form Image Inpainting with Gated Convolution

import torch
import torchvision
from torch import nn
from torch.nn.functional import avg_pool2d, upsample

from .BaseModels import BaseModule


class PartialConv(BaseModule):
    # reference:Image Inpainting for Irregular Holes Using Partial Convolutions
    # http://masc.cs.gmu.edu/wiki/partialconv/show?time=2018-05-24+21%3A41%3A10
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting/blob/master/common/net.py
    # mask is binary, 0 is masked point, 1 is not
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias)
        torch.nn.init.constant(self.mask_conv.weight, 1.0)
        torch.nn.init.constant(self.mask_conv.bias, 0.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, args):
        x, mask = args
        output = self.feature_conv(x * mask)
        # memory efficient
        output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)

        output_mask = self.mask_conv(mask)  # mask sums

        update_holes = output_mask != 0
        keep_holes = output_mask == 0
        output[update_holes] = (output[update_holes] - output_bias[update_holes]) \
                               / output_mask[update_holes] + output_bias[[update_holes]]

        output[keep_holes] = 0

        output_mask[update_holes] = 1.0
        output_mask[keep_holes] = 0.0
        return (output, output_mask)


class DoubleAvdPool(nn.AvgPool2d):
    def __init__(self, kernel_size):
        super(DoubleAvdPool, self).__init__(kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, args):
        type(args)
        return tuple(map(lambda x: avg_pool2d(x, kernel_size=self.kernel_size), args))


class DoubleUpSample(nn.Upsample):
    def __init__(self, scale_factor, mode):
        super(DoubleUpSample, self).__init__(scale_factor=scale_factor, mode=mode)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, args):
        return tuple(map(lambda x: upsample(x, scale_factor=self.scale_factor, mode=self.mode), args))


class DoubleActivation(nn.Module):
    def __init__(self, activation):
        super(DoubleActivation, self).__init__()
        self.activation = activation

    def forward(self, args):
        x, mask = args
        return self.activation(x), mask


class DoubleNorm(nn.Module):
    def __init__(self, norm):
        super(DoubleNorm, self).__init__()
        self.norm = norm

    def forward(self, args):
        x, mask = args
        return self.norm(x), mask


def partial_conv_block(in_channels, out_channels, kernel_size, stride=1,
                       padding=0, dilation=1, groups=1, bias=True, BN=True, activation=None):
    m = [PartialConv(in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)]
    if BN:
        m.append(DoubleNorm(nn.BatchNorm2d(out_channels)))
    if activation:
        m.append(DoubleActivation(activation))
    return m


class PartialGatedConv(BaseModule):
    # mask is binary, 0 is masked point, 1 is not
    # https://github.com/JiahuiYu/generative_inpainting/issues/62
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=nn.SELU()):
        super(PartialGatedConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride,
                                      padding, dilation, groups, bias)
        self.act_fn = activation

    def forward(self, x):
        output = self.feature_conv(x)
        feature, gate = output.chunk(2, dim=1)
        return self.act_fn(feature) * torch.sigmoid(gate)


def partial_gated_conv_block(in_channels, out_channels, kernel_size, stride=1,
                             padding=0, dilation=1, groups=1, bias=True, BN=False, activation=nn.SELU()):
    m = [PartialGatedConv(in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, bias, activation)]
    if BN:
        m.append(nn.BatchNorm2d(out_channels))

    return m


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

        m = partial_gated_conv_block(in_channels, out_channels, kernel_size, stride,
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


class Vgg19Extractor(BaseModule):
    def __init__(self, pretrained=True):
        super(Vgg19Extractor, self).__init__()
        vgg19 = torchvision.models.vgg16(pretrained=pretrained)
        feature1 = nn.Sequential(*vgg19.features[:5])
        feature2 = nn.Sequential(*vgg19.features[5:10])
        feature3 = nn.Sequential(*vgg19.features[10:17])
        self.features = nn.Sequential(*[feature1, feature2, feature3])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, img):
        result = []
        for layer in self.features.children():
            img = layer(img)
            result.append(img)
        return result
