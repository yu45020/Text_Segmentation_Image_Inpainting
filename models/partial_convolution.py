# Two types:
# 1. "Hard" gated : use 1/0 to update mask.  Image Inpainting for Irregular Holes Using Partial Convolutions
# 2. "Soft" gated : use sigmoid to update both feature & mask  Free-Form Image Inpainting with Gated Convolution

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import avg_pool2d

from .BaseModels import BaseModule

try:
    from .inplace_abn import InPlaceABN  # only works in GPU

    inplace_batch_norm = True
except ImportError:
    inplace_batch_norm = False


class PartialConv(BaseModule):
    # reference:
    # Image Inpainting for Irregular Holes Using Partial Convolutions
    # http://masc.cs.gmu.edu/wiki/partialconv/show?time=2018-05-24+21%3A41%3A10
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting/blob/master/common/net.py
    # mask is binary, 0 is holes; 1 is not
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
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
        if self.feature_conv.bias is not None:
            output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)  # mask sums
            # ones = self.mask_conv(torch.ones_like(mask))

            # used to check whether holes are in the same positions across channels
            # if self.mask_conv.kernel_size[0] == 1:
            #     a = output_mask[:, :1, :, :]
            #     b = mask[:, :1, :, :]
            #
            #     assert torch.equal(a / torch.max(output_mask), b)
            #     assert torch.equal(a.expand_as(output_mask), output_mask)
            #     assert torch.equal(b.expand_as(mask), mask)

        update_holes = output_mask > 0
        mask_sum = torch.where(update_holes, output_mask, torch.ones_like(output))

        # See 2nd reference, but takes more time to run
        # scale = torch.div(ones, mask_sum)

        output_pre = (output - output_bias) / mask_sum + output_bias

        output = torch.where(update_holes, output_pre, torch.zeros_like(output))
        new_mask = update_holes.float()
        # output = output_pre * new_mask

        return output, new_mask


class PartialConv1x1(BaseModule):
    """
    Optimization for encoder :
    if the input mask have holes in the same positions across channels,
    then 1x1 partial convolution is equivalent to a standard 1x1 convolution because holes are not updated.

    By assert checking, encoder and feature pooling are eligible,
    but decoder needs to concatenate encoder's mask, so it fails.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv1x1, self).__init__()
        assert kernel_size == 1 and stride == 1 and padding == 0
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        nn.init.kaiming_normal_(self.feature_conv.weight)

    def forward(self, args):
        x, mask = args
        out_x = self.feature_conv(x)
        out_m = mask[:, :1, :, :].expand_as(out_x)
        return out_x, out_m


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
                              dilation=1, groups=1, bias=False, BN=True, activation=True, use_1_conv=False):
    if use_1_conv:
        m = [PartialConv1x1(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias)]
    else:
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

        if inplace_batch_norm:
            if act_fn:
                self.bn_act = InPlaceABN(channel, activation="leaky_relu", slope=0.3)
            else:
                self.bn_act = InPlaceABN(channel, activation='none')

        else:
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
