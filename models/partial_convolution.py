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
                 padding=0, dilation=1, groups=1, bias=True,
                 same_holes=False):
        # same holes: holes are in the same position in all layers. used in the encoder part

        super(PartialConv, self).__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        nn.init.kaiming_normal_(self.feature_conv.weight)

        self.same_holes = same_holes
        mask_in_channel = 1 if same_holes else in_channels
        mask_out_channel = 1 if same_holes else out_channels
        mask_groups = 1 if same_holes else groups
        self.mask_conv = nn.Conv2d(mask_in_channel, mask_out_channel, kernel_size, stride,
                                   padding, dilation, mask_groups, bias=False)

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
            if self.same_holes:
                output_mask = self.mask_conv(mask[:, :1])  # mask sums
                no_update_holes = output_mask == 0
                output_mask *= self.feature_conv.in_channels
            else:
                output_mask = self.mask_conv(mask)  # mask sums
                no_update_holes = output_mask == 0

        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        # See 2nd reference, but takes more time to run
        # scale = torch.div(ones, mask_sum)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        if self.same_holes:
            new_mask = new_mask.expand_as(output)
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


class PartialConvNoHoles(PartialConv):
    """
    Optimization for encoder :
    Used for the decoder part. After successive partial convolution, the decoder should have no holes in the masks.
    The u-net structure links the encoder mask with decoder mask, so 1x1 convolution will fill all holes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConvNoHoles, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                 padding, dilation, groups, bias)
        assert self.feature_conv.groups == 1

    def forward(self, args):
        x, mask = args
        output = self.feature_conv(x * mask)
        if self.feature_conv.bias is not None:
            output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        mask_sum = output_mask

        output = (output - output_bias) / mask_sum + output_bias
        new_mask = torch.ones_like(output)

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
                              dilation=1, groups=1, bias=False, BN=True, activation=True,
                              use_1_conv=False, no_holes_1_conv=False, same_holes=False):
    if use_1_conv:
        m = [PartialConv1x1(in_channels, out_channels, kernel_size, stride,
                            padding, dilation, groups, bias)]
    elif no_holes_1_conv:
        m = [PartialConvNoHoles(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)]
    else:
        m = [PartialConv(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, same_holes)]
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
