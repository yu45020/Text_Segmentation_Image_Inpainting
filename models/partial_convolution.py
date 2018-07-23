# Two types:
# 1. "Hard" gated : use 1/0 to update mask.  Image Inpainting for Irregular Holes Using Partial Convolutions
# 2. "Soft" gated : use sigmoid to update both feature & mask  Free-Form Image Inpainting with Gated Convolution

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import avg_pool2d

from .BaseModels import BaseModule


class PartialConv(BaseModule):
    # reference:
    # Image Inpainting for Irregular Holes Using Partial Convolutions
    # http://masc.cs.gmu.edu/wiki/partialconv/show?time=2018-05-24+21%3A41%3A10
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting/blob/master/common/net.py
    # mask is binary, 0 is holes; 1 is not
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, ):
        super(PartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        nn.init.kaiming_normal_(self.feature_conv.weight)
        # self.mask_conv = partial(F.conv2d, weight=torch.ones_like(self.feature_conv.weight),
        #                          bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        # torch.nn.init.constant_(self.mask_conv.bias, 0.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, args):
        x, mask = args
        output = self.feature_conv(x * mask)
        output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)  # mask sums
            # ones = self.mask_conv(torch.ones_like(mask))

        update_holes = output_mask > 0
        mask_sum = torch.where(update_holes, output_mask, torch.ones_like(output))

        # See 2nd reference
        # scale = torch.div(ones, mask_sum)
        # aa = torch.where(update_holes, scale, torch.zeros_like(scale))
        # print(f"max value of scale is {torch.max(aa)}")
        output_pre = (output - output_bias) / mask_sum + output_bias

        output = torch.where(update_holes, output_pre, torch.zeros_like(output))
        new_mask = update_holes.float()
        # output = output_pre * new_mask

        return output, new_mask


class SoftPartialConv(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, ):
        super(SoftPartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                   bias=False)

    def forward(self, args):
        x, mask = args
        output = self.feature_conv(x)

        mask_output = self.mask_conv(1 - mask)  # holes are 1; else 0
        mask_attention = F.tanh(mask_output)  # non-holes positions are 0
        output = output + mask_attention * output

        valid_idx = mask_attention == 0
        new_mask = torch.where(valid_idx, torch.ones_like(output), F.sigmoid(mask_output))
        return output, new_mask


def partial_convolution_block(in_channels, out_channels, kernel_size, stride=1, padding=0,
                              dilation=1, groups=1, bias=True, BN=True, activation=True):
    m = [PartialConv(in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)]
    if BN:
        m += [PartialActivatedBN(out_channels, activation)]
    if not BN and activation:
        m += [PartialActivation(activation)]

    return nn.Sequential(*m)


class PartialActivatedBN(BaseModule):
    def __init__(self, channel, act_fn):
        super(PartialActivatedBN, self).__init__()
        if act_fn:
            self.bn_act = nn.Sequential(nn.BatchNorm2d(channel), act_fn)
        else:
            self.bn_act = nn.Sequential(nn.BatchNorm2d(channel))

    def forward(self, args):
        x, mask = args
        return self.bn_act(x), mask


class PartialActivation(BaseModule):
    def __init__(self, activation):
        super(PartialActivation, self).__init__()
        self.act_fn = activation

    def forward(self, args):
        x, mask = args
        return self.act_fn(x), mask


class DoubleAvdPool(nn.AvgPool2d):
    def __init__(self, kernel_size):
        super(DoubleAvdPool, self).__init__(kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, args):
        type(args)
        return tuple(map(lambda x: avg_pool2d(x, kernel_size=self.kernel_size), args))


class DoubleUpSample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(DoubleUpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, args):
        x, mask = args
        return self.upsample(x), self.upsample(mask)


class PartialGatedConv(BaseModule):
    # mask is binary, 0 is masked point, 1 is not
    # https://github.com/JiahuiYu/generative_inpainting/issues/62
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, BN=False, activation=nn.SELU()):
        super(PartialGatedConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias)
        if BN:
            self.bn_act = nn.Sequential(nn.BatchNorm2d(out_channels), activation)
        else:
            self.bn_acf = activation

    def forward(self, x):
        output = self.feature_conv(x)
        mask = self.mask_conv(x)
        return self.bn_act(output * F.sigmoid(mask))


class PartialGatedActivatedBN(BaseModule):
    def __init__(self, channel, activation):
        super(PartialGatedActivatedBN, self).__init__()
        self.bn_act = nn.Sequential(nn.BatchNorm2d(channel),
                                    activation)

    def forward(self, x):
        return self.bn_act(x)


def partial_gated_conv_block(in_channels, out_channels, kernel_size, stride=1,
                             padding=0, dilation=1, groups=1, bias=True, BN=True, activation=None):
    m = [PartialGatedConv(in_channels, out_channels, kernel_size, stride,
                          padding, dilation, groups, bias, BN, activation)]

    return nn.Sequential(*m)
