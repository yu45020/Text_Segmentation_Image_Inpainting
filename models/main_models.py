import math

import torch
from torch.utils.checkpoint import checkpoint

from .BaseModels import BaseModule, Conv_block, Partial_Conv_block
from torch import nn
from .MobileNetV2 import MobileNetV2
from collections import OrderedDict
from torch.nn import functional as F


class MobileNetEncoder(MobileNetV2):
    def __init__(self, pre_train_checkpoint=None, drop_last2=True, add_partial=False):
        super(MobileNetEncoder, self).__init__(drop_last2=drop_last2, add_partial=add_partial)
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
    def __init__(self, encoder_checkpoint=None):
        super(ImageInpainting, self).__init__()
        self.encoder = MobileNetEncoder(drop_last2=True, add_partial=True)

        self.encoder_2_decoder = self.make_transitor(self.encoder.last_channel, 256, 3)

        self.up_sampler1 = self.make_upsampler(256, 32)
        self.decoder1 = self.make_decoder(32 + 32, 256)

        self.up_sampler2 = self.make_upsampler(256, 32)
        self.decoder2 = self.make_decoder(32 + 24, 128)

        self.up_sampler3 = self.make_upsampler(128, 32)
        self.decoder3 = self.make_decoder(32 + 16, 128)

        self.up_sampler4 = self.make_upsampler(128, 32)
        self.out_conv = self.make_decoder(32 + 3, 3)
        if encoder_checkpoint:
            self.encoder.freeze_params(encoder_checkpoint, free_last_blocks=0)

    def make_transitor(self, in_channel, out_channel, num_layers):
        m = Partial_Conv_block(in_channels=in_channel, out_channels=48, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1, bias=False, BN=True, activation=nn.ReLU6())
        out_c = 48
        for _ in range(num_layers):
            m.extend(Partial_Conv_block(in_channels=out_c, out_channels=out_channel, kernel_size=1, stride=1,
                                        padding=0, dilation=1, groups=1, bias=False, BN=True, activation=nn.ReLU6()))
            out_c = out_channel

        return nn.Sequential(*m)

    def make_upsampler(self, in_channel, out_channel):
        m1 = nn.Sequential(*Conv_block(in_channel, out_channel, 1, 1, padding=0,
                                       bias=False, BN=True, activation=None),
                           nn.Upsample(scale_factor=2, mode='nearest'))

        return m1

    def make_decoder(self, in_channel, out_channel):
        m = nn.Sequential(
            *Partial_Conv_block(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                padding=1, bias=True, BN=True, activation=nn.ReLU6()),
            *Partial_Conv_block(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1,
                                padding=1, bias=True, BN=True, activation=nn.ReLU6())
        )
        return m

    def forward(self, args):
        x, mask = args
        out_mask = [mask]
        out_x = [x]
        for index, layer in enumerate(self.encoder.features):
            x, mask = layer((x, mask))
            if index in [1, 2, 3]:  # right before down scale in the encoder
                out_x.append(x)
                out_mask.append(mask)
        x.size()
        mask.size()
        x, mask = self.encoder_2_decoder((x, mask))
        x = self.up_sampler1(x)
        mask = self.up_sampler1(mask)
        x = torch.cat([x, out_x[-1]], dim=1)
        mask = torch.cat([mask, out_mask[-1]], dim=1)

        x, mask = self.decoder1((x, mask))
        x = self.up_sampler2(x)
        mask = self.up_sampler2(mask)
        x = torch.cat([x, out_x[-2]], dim=1)
        mask = torch.cat([mask, out_mask[-2]], dim=1)
        x, mask = self.decoder2((x, mask))

        x = self.up_sampler3(x)
        mask = self.up_sampler3(mask)
        x = torch.cat([x, out_x[-3]], dim=1)
        mask = torch.cat([mask, out_mask[-3]], dim=1)
        x, mask = self.decoder3((x, mask))

        x = self.up_sampler4(x)
        mask = self.up_sampler4(mask)
        x = torch.cat([x, out_x[-4]], dim=1)
        mask = torch.cat([mask, out_mask[-4]], dim=1)
        x, mask = self.out_conv((x, mask))
        return x


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
