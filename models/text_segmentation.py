# Encoder : MobileNet V2
# Feature Pooling: Receptive Field Block (RFB)
# Decoder : 2x up sample ---> concatenate with 1/4 input feature maps ---> 4x up sample
#           Transpose convolution will have strong checkerboard effect;
#           pixel shuffling may make lots of false positive
#           bilinear up sampling might be enough

import torch
from torch import nn
from torch.nn import functional as F

from .BaseModels import BaseModule, Conv_block
from .MobileNetV2 import DilatedMobileNetV2, InvertedResidual
from .Xception import Xception
from .common import RFB, ASP


class TextSegament(BaseModule):
    def __init__(self, encoder_checkpoint=None, free_last_blocks=-1, width_mult=2):
        """

        :param encoder_checkpoint: Encoder check point, either string or torch.load(checkpoint)
        :param free_last_blocks: 0: freeze all encoder, -1: re-train all, 1~n: freeze the first {} blocks
        :param width_mult: width multiplier for Mobile Net V2
        """
        super(TextSegament, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3)  # in place batch norm only support LeakyRelu, ELU
        # use the pre-train weights to initialize the model
        self.encoder = DilatedMobileNetV2(width_mult=width_mult, activation=self.act_fn,
                                          bias=False, add_sece=True, add_partial=False)

        # down scale 1/2 features
        self.feature_avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # self.encoder.last_channel --|
        # all feature maps with 1/8 out stride
        feature_channels = sum([i[0].out_channels for i in self.encoder.features[3:]])
        self.feature_pooling = RFB(feature_channels, 256, activation=self.act_fn, add_sece=True)

        # concatenate 3 features of 1/2 and 1/4  == 144
        concat_c = sum([i[0].out_channels for i in self.encoder.features[:3]])
        self.feature_4x_conv = InvertedResidual(concat_c, 128, stride=1, expand_ratio=1, dilation=1,
                                                activation=self.act_fn, add_sece=True)

        # concatenate input features in 1/4
        self.smooth_feature_4x_conv = nn.Sequential(
            InvertedResidual(256 + 128, 128, stride=1, expand_ratio=1, dilation=2,
                             activation=self.act_fn, add_sece=True),
            InvertedResidual(128, 128, stride=1, expand_ratio=1, dilation=1,
                             activation=self.act_fn, add_sece=True)
        )

        self.out_conv = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=True, stride=1),
                                      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
                                      )

        self.initialize_weights()
        self.encoder.load_pre_train_checkpoint(encoder_checkpoint, free_last_blocks)

    def forward(self, x):
        layer_out = []  # contains 1/2, 1/2, 1/4 feature maps
        for layer in self.encoder.features[:3]:
            x = layer(x)
            layer_out.append(x)

        layer_out[0] = self.feature_avg_pool(layer_out[0])
        layer_out[1] = self.feature_avg_pool(layer_out[1])
        layer_out = torch.cat(layer_out, dim=1)

        pooled_features = []  # 1/8 feature maps with various dilation rate
        for layer in self.encoder.features[3:]:
            x = layer(x)
            pooled_features.append(x)

        x = self.feature_pooling(torch.cat(pooled_features, dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # concatenate inpute features
        layer_out = self.feature_4x_conv(layer_out)
        x = torch.cat([layer_out, x], dim=1)

        x = self.smooth_feature_4x_conv(x)
        x = self.out_conv(x)
        return x


class XecptionTextSegment(BaseModule):
    def __init__(self):
        super(XecptionTextSegment, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3)
        self.encoder = Xception(color_channel=3, act_fn=self.act_fn)
        self.feature_pooling = ASP(self.encoder.last_feature_channels, 256, self.act_fn)

        self.feature_4x_conv = nn.Sequential(
            *Conv_block(self.encoder.x4_feature_channels, 48, kernel_size=1,
                        bias=False, BN=True, activation=self.act_fn))

        self.out_conv = nn.Sequential(
            *Conv_block(48 + 256, 256, kernel_size=3, stride=1, padding=1,
                        bias=False, BN=True, activation=self.act_fn),
            *Conv_block(256, 1, kernel_size=3, stride=1, padding=1,
                        bias=False, BN=True, activation=self.act_fn),
        )

    def forward(self, x):
        x, x4_features = self.encoder(x)
        x = self.feature_pooling(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x4_features = self.feature_4x_conv(x4_features)
        x = torch.cat([x, x4_features], dim=1)
        x = self.out_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
