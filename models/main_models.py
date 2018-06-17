import math

import torch
from torch.utils.checkpoint import checkpoint

from .BaseModels import BaseModule, Conv_block
from torch import nn
from .MobileNetV2 import MobileNetV2
from collections import OrderedDict
from torch.nn import functional as F


class MobileNetEncoder(MobileNetV2):
    def __init__(self, pre_train_checkpoint=None, drop_last2=True):
        super(MobileNetEncoder, self).__init__(drop_last2=drop_last2)
        self.inverted_residual_setting = [
            # t, c, n, s, dial
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],  # # Rethinking Atrous Convolution for Semantic Image Segmentation
            [6, 160, 3, 1, 2],  # origin: [6, 160, 3, 2,]   out stride 32x  --> they are replaced by astro conv
            [6, 320, 1, 1, 2],  # origin:  [6, 320, 1, 1]   out stride 32x
        ]
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting, drop_last2)
        self.freeze_params(pre_train_checkpoint)

    def freeze_params(self, pre_train_checkpoint=None, free_last_blocks=2):
        if pre_train_checkpoint:
            self.load_state_dict(torch.load(pre_train_checkpoint))
            # the last 4 blocks are changed from stride of 2 to dilation of 2
        for i in range(len(self.features) - free_last_blocks):
            for params in self.features[i].parameters():
                params.requires_grad = False


class ASP(BaseModule):
    # Atrous Spatial Pyramid Pooling with Image Pooling
    def __init__(self, in_channel, out_channel=256):
        super(ASP, self).__init__()

        self.asp = nn.Sequential(
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=1, stride=1, padding=0,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=4, dilation=4,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=8, dilation=8,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12,
                                      bias=False, BN=True, activation=None))
        )

        """ To see why adding gobal average, please refer to 3.1 Global Context in https://www.cs.unc.edu/~wliu/papers/parsenet.pdf """
        self.img_pooling_1 = nn.AdaptiveAvgPool2d(1)
        self.img_pooling_2 = nn.Sequential(
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=False, BN=True, activation=None))

        self.initialize_weights()

    def forward(self, x):
        avg_pool = self.img_pooling_1(x)
        avg_pool = F.upsample(avg_pool, size=x.shape[2:], mode='bilinear')
        avg_pool = [self.img_pooling_2(avg_pool)]

        asp_pool = [layer(x) for layer in self.asp.children()]
        return torch.cat(avg_pool + asp_pool, dim=1)

    def forward_checkpoint(self, x):
        avg_pool = checkpoint(self.img_pooling_1, x)
        avg_pool = F.upsample(avg_pool, size=x.shape[2:], mode='bilinear')
        avg_pool = [checkpoint(self.img_pooling_2, avg_pool)]

        asp_pool = [checkpoint(layer, x) for layer in self.asp.children()]
        return torch.cat(avg_pool + asp_pool, dim=1)


class ImageInpainting(BaseModule):
    def __init__(self):
        super(ImageInpainting, self).__init__()
        pass


class ImageInpaintingEncoder(BaseModule):
    # use partial convolution to update masks
    # and use pre-train backbone to extra features
    # No back prop update is used
    def __init__(self, encoder):
        super(ImageInpaintingEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.freeze_params(free_last_blocks=0)  # freeze all layers
        self.mask_convs = self.make_mask_convs(encoder)

    def make_mask_convs(self, encoder):
        channels = self.get_in_out_channels(encoder)
        kernels = [7, 5, 5, 3, 3, 3, 3, 3]  # follow Mobile Net V2
        strides = [2, 1, 2, 2, 2, 1, 1, 1]  # same
        m = [nn.Conv2d(*i, k, s, padding=0, bias=False) for i, k, s in zip(channels, kernels, strides)]
        out = nn.Sequential(*m)
        for layer in out.children():
            for param in layer.parameters():
                torch.nn.init.constant_(param, 1)
                param.requires_grad = False
        return out

    @staticmethod
    def get_in_out_channels(encoder):
        blocks = [[encoder.features[0][0].in_channels, encoder.features[0][0].out_channels]]
        for i in range(1, len(encoder.features)):
            for layers in encoder.features[i][0].children():
                in_channel = layers[0].in_channels
                for layer in layers.children():
                    if isinstance(layer, nn.Conv2d):
                        out_channel = layer.out_channels
                blocks.append([in_channel, out_channel])
        return blocks

    def forward(self, x, input_mask):
        assert x.size() == input_mask.size()
        out_mask = []
        for layer, mask_layer in zip(self.encoder.features.children(), self.mask_convs):
            x = layer(x)
            pad_size = (1 - mask_layer.stride[0] + mask_layer.kernel_size[0]) // 2
            input_mask = nn.ZeroPad2d(pad_size)(input_mask)
            input_mask = mask_layer(input_mask)
            out_mask.append(input_mask)
            print(x.size(), input_mask.size())
        return x, out_mask


class TextSegament(BaseModule):
    def __init__(self, encoder_checkpoint=None):
        super(TextSegament, self).__init__()

        self.encoder = MobileNetEncoder(encoder_checkpoint,
                                        drop_last2=True)  # may need to retrain the last 4 layers
        self.layer_4x_conv = nn.Sequential(*Conv_block(24, 128, kernel_size=3, padding=1,
                                                       bias=False, BN=True, activation=nn.ReLU6()))
        self.feature_pooling = ASP(self.encoder.last_channel, out_channel=256)
        self.transition_2_decoder = nn.Sequential(*Conv_block(256 * 5, 128, kernel_size=1,
                                                              bias=False, BN=True, activation=nn.ReLU6()))

        self.smooth_4x_conv = nn.Sequential(*Conv_block(128 * 2, 64, kernel_size=1,
                                                        bias=False, BN=True, activation=nn.ReLU6()),
                                            *Conv_block(64, 64, kernel_size=3, padding=1, groups=64,
                                                        bias=False, BN=True, activation=nn.ReLU6()),
                                            *Conv_block(64, 32, kernel_size=3, padding=1,
                                                        bias=False, BN=True, activation=nn.ReLU6()))

        self.out_conv = nn.Sequential(*Conv_block(32, 4, kernel_size=3, padding=1,
                                                  bias=False, BN=False, activation=None))
        self.softmax2d = nn.Softmax2d()  # to get mask

    def forward(self, x):
        for index, layer in enumerate(self.encoder.features.children()):
            x = layer(x)
            if index == 3:
                layer_out4x = self.layer_4x_conv(x)
            else:
                continue
        x = self.feature_pooling(x)
        x = self.transition_2_decoder(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = torch.cat([x, layer_out4x], dim=1)
        x = self.smooth_4x_conv(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = self.out_conv(x)
        return x

    def forward_checkpoint(self, x):
        for index, layer in enumerate(self.encoder.features.children()):
            x = checkpoint(layer, x)
            if index == 3:
                layer_out4x = checkpoint(self.layer_4x_conv, x)

        x = checkpoint(self.feature_pooling, x)
        x = self.transition_2_decoder(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = torch.cat([x, layer_out4x], dim=1)
        x = checkpoint(self.smooth_4x_conv, x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = checkpoint(self.out_conv, x)
        return x
