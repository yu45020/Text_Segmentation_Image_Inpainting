# Encoder : MobileNet V2 +  Atrous Spatial Pyramid Pooling with Image Pooling
# Decoder : 2 4x up sample covolution block
# reference : Deeplab v3+ , MobileNetV2

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential

from .BaseModels import BaseModule, Conv_block
from .MobileNetV2 import DilatedMobileNetV2


#
#
# class MobileNetEncoder(MobileNetV2):
#     def __init__(self, add_partial=False, activation=nn.ReLU6(),
#                  bias=False, width_mult=1):
#         super(MobileNetEncoder, self).__init__(add_partial=add_partial, width_mult=width_mult)
#         self.add_partial = add_partial
#         self.bias = bias
#         self.width_mult = width_mult
#         self.act_fn = activation
#         self.out_stride = 8
#         # # Rethinking Atrous Convolution for Semantic Image Segmentation
#         self.inverted_residual_setting = [
#             # t, c, n, s, dila  # input size
#             [1, 16, 1, 1, 1],  # 1/2
#             [6, 24, 2, 2, 1],  # 1/4
#             [6, 32, 3, 2, 1],  # 1/8
#             [6, 64, 4, 1, 2],  # <-- add astrous conv and keep 1/8
#             [6, 96, 3, 1, 4],
#             [6, 160, 3, 1, 8],
#             [6, 320, 1, 1, 16],
#         ]
#         self.features = self.make_inverted_resblocks(self.inverted_residual_setting)
#
#     def load_pre_train_checkpoint(self, pre_train_checkpoint, free_last_blocks):
#         if pre_train_checkpoint:
#             if isinstance(pre_train_checkpoint, str):
#                 self.load_state_dict(torch.load(pre_train_checkpoint, map_location='cpu'))
#             else:
#                 self.load_state_dict(pre_train_checkpoint)
#             print("Encoder check point is loaded")
#         else:
#             print("No check point for the encoder is loaded. ")
#         if free_last_blocks >= 0:
#             self.freeze_params(free_last_blocks)
#
#         else:
#             print("All layers in the encoders are re-trained. ")
#
#     def freeze_params(self, free_last_blocks=2):
#         # the last 4 blocks are changed from stride of 2 to dilation of 2
#         for i in range(len(self.features) - free_last_blocks):
#             for params in self.features[i].parameters():
#                 params.requires_grad = False
#         print("{}/{} layers in the encoder are freezed.".format(len(self.features) - free_last_blocks,
#                                                                 len(self.features)))


class SpatialChannelSqueezeExcitation(BaseModule):
    # https://arxiv.org/pdf/1803.02579v1.pdf
    def __init__(self, in_channel):
        super(SpatialChannelSqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excite = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 2, in_channel),
            nn.Sigmoid()
        )
        self.spatial_excite = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        channel = self.avg_pool(x).view(b, c)
        cSE = self.channel_excite(channel).view(b, c, 1, 1)
        x_cSE = torch.mul(x, cSE)

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
    def __init__(self, in_channel, out_channel, activation):
        super(RFB, self).__init__()
        asp_rate = [5, 17, 29]
        # self.act_fn = activation
        self.input_down_channel = nn.Sequential(
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=False, BN=True, activation=False))

        self.rfb_linear_conv = nn.Conv2d(out_channel * 4, out_channel, kernel_size=1, bias=False)
        self.rfb = nn.Sequential(
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=1,
                                     astro_rate=1, half_conv=False),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=3,
                                     astro_rate=asp_rate[0], half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=5,
                                     astro_rate=asp_rate[1], half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=7,
                                     astro_rate=asp_rate[2], half_conv=True)
        )

    @staticmethod
    def make_pooling_branch(in_channel, mid_channel, out_channel, conv_kernel, astro_rate, half_conv=False):
        # from the paper: we use a 1 x n plus an nx1 conv-layer to take place of the original nxn convlayer
        # similar to EffNet style
        if half_conv:
            m = nn.Sequential(
                *Conv_block(in_channel, mid_channel, kernel_size=1, padding=0,
                            bias=False, BN=True, activation=None),
                *Conv_block(mid_channel, 3 * mid_channel // 2, kernel_size=(1, conv_kernel),
                            padding=(0, (conv_kernel - 1) // 2),
                            bias=False, BN=True, activation=None),
                *Conv_block(3 * mid_channel // 2, out_channel, kernel_size=(conv_kernel, 1),
                            padding=((conv_kernel - 1) // 2, 0),
                            bias=False, BN=True, activation=None),
                *Conv_block(out_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=False, BN=True, activation=None, groups=out_channel))
        else:
            m = nn.Sequential(
                *Conv_block(in_channel, mid_channel, kernel_size=conv_kernel, padding=(conv_kernel - 1) // 2,
                            bias=False, BN=True, activation=None),
                *Conv_block(mid_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=False, BN=True, activation=None, groups=mid_channel))

        return m

    def forward(self, x):
        # feature poolings
        rfb_pool = [layer(x) for layer in self.rfb.children()]
        rfb_pool = torch.cat(rfb_pool, dim=1)
        rfb_pool = self.rfb_linear_conv(rfb_pool)

        # skip connection
        resi = self.input_down_channel(x)
        return rfb_pool + resi


class TextSegament(BaseModule):
    def __init__(self, encoder_checkpoint=None, free_last_blocks=-1, width_mult=1):
        super(TextSegament, self).__init__()
        self.act_fn = nn.SELU(inplace=True)
        self.bias = True
        # use the pre-train weights to initialize the model
        self.encoder = DilatedMobileNetV2(activation=self.act_fn, bias=False, width_mult=width_mult)

        # down scale 1/2 features
        self.feature_avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.encoder.last_channel --|
        # all feature maps
        feature_channels = sum([i[0].out_channels for i in self.encoder.features[3:]])
        self.feature_pooling = RFB(feature_channels, 256, activation=self.act_fn)

        self.up_4x_conv = nn.Sequential(*Conv_block(256, 256, kernel_size=3, padding=1, groups=1,
                                                    bias=self.bias, BN=True, activation=self.act_fn),
                                        SpatialChannelSqueezeExcitation(256))
        # concatenate 3 features of 1/2 and 1/4
        concat_c = sum([i[0].out_channels for i in self.encoder.features[:3]])
        self.feature_4x_conv = nn.Sequential(*Conv_block(concat_c, 256, kernel_size=3, padding=1, groups=1,
                                                         bias=self.bias, BN=True, activation=self.act_fn),
                                             SpatialChannelSqueezeExcitation(256))
        # concatenate input features in 1/4
        self.smooth_feature_4x_conv = nn.Sequential(*Conv_block(512, 64, kernel_size=1, padding=0, groups=1,
                                                                bias=self.bias, BN=True, activation=self.act_fn),
                                                    nn.Upsample(scale_factor=4, mode='bilinear'),
                                                    *Conv_block(64, 128, kernel_size=3, padding=1, groups=64,
                                                                bias=self.bias, BN=False, activation=None),
                                                    SpatialChannelSqueezeExcitation(128))

        self.out_conv = nn.Conv2d(128, 2, kernel_size=1, padding=0, bias=self.bias)
        # init weights
        if isinstance(self.act_fn, torch.nn.SELU):
            self.selu_init_params()
        else:
            self.initialize_weights()
        self.encoder.load_pre_train_checkpoint(encoder_checkpoint, free_last_blocks)
        # add channel squeeze and spatial excitation blocks except the last block
        # for i in self.encoder.features:
        #     add_SCSE_block(i)

    def forward(self, x):
        layer_out = []
        for layer in self.encoder.features[:3]:
            x = checkpoint_sequential(layer, len(list(layer)), x)
            # x = layer(x)
            layer_out.append(x)

        layer_out[0] = self.feature_avg_pool(layer_out[0])
        layer_out[1] = self.feature_avg_pool(layer_out[1])
        layer_out = torch.cat(layer_out, dim=1)

        pooled_features = []
        for layer in self.encoder.features[3:]:
            x = checkpoint_sequential(layer, len(list(layer)), x)
            # x = layer(x)
            pooled_features.append(x)

        x = self.feature_pooling(torch.cat(pooled_features, dim=1))
        x = F.upsample(x, scale_factor=2, mode='bilinear')

        x = self.up_4x_conv(x)
        # concatenate inpute features

        layer_out = self.feature_4x_conv(layer_out)
        x = torch.cat([layer_out, x], dim=1)
        x = self.smooth_feature_4x_conv(x)

        x = self.out_conv(x)
        return x

    def forward_checkpoint(self, x):
        with torch.no_grad():
            return self.forward(x)
