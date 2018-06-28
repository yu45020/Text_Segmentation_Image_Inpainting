# Encoder : MobileNet V2 +  Atrous Spatial Pyramid Pooling with Image Pooling
# Decoder : 2 4x up sample covolution block
# reference : Deeplab v3+ , MobileNetV2

import torch
from torch import nn
from torch.nn import functional as F

from .BaseModels import BaseModule, Conv_block
from .MobileNetV2 import MobileNetV2


class MobileNetEncoder(MobileNetV2):
    def __init__(self, pre_train_checkpoint=None, free_last_blocks=None, add_partial=False, activation=nn.ReLU6(),
                 bias=False, width_mult=1):
        super(MobileNetEncoder, self).__init__(add_partial=add_partial, width_mult=width_mult)
        self.add_partial = add_partial
        self.bias = bias
        self.act_fn = activation
        self.inverted_residual_setting = [
            # t, c, n, s, dila
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],  # # Rethinking Atrous Convolution for Semantic Image Segmentation
            [6, 160, 3, 1, 2],  # origin: [6, 160, 3, 2,]   out stride 32x  --> they are replaced by astro conv
            [6, 320, 1, 1, 2],  # origin:  [6, 320, 1, 1]   out stride 32x
        ]
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting)
        if pre_train_checkpoint:
            if isinstance(pre_train_checkpoint, str):
                self.load_state_dict(torch.load(pre_train_checkpoint, map_location='cpu'))
            else:
                self.load_state_dict(pre_train_checkpoint)
            print("Encoder check point is loaded")
        else:
            print("No check point for the encoder is loaded. ")
        if free_last_blocks >= 0:
            self.freeze_params(free_last_blocks)

        else:
            print("All layers in the encoders are re-trained. ")

    def freeze_params(self, free_last_blocks=2):
        # the last 4 blocks are changed from stride of 2 to dilation of 2
        for i in range(len(self.features) - free_last_blocks):
            for params in self.features[i].parameters():
                params.requires_grad = False
        print("{}/{} layers in the encoder are freezed.".format(len(self.features) - free_last_blocks,
                                                                len(self.features)))


class ASP(BaseModule):
    # Atrous Spatial Pyramid Pooling with Image Pooling
    def __init__(self, in_channel, out_channel=256):
        super(ASP, self).__init__()

        self.asp = nn.Sequential(
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=1, stride=1, padding=0,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12,
                                      bias=False, BN=True, activation=None)),
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=18, dilation=18,
                                      bias=False, BN=True, activation=None))
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
        avg_pool = [self.img_pooling_2(avg_pool)]

        asp_pool = [layer(x) for layer in self.asp.children()]
        return torch.cat(avg_pool + asp_pool, dim=1)


class TextSegament(BaseModule):
    def __init__(self, encoder_checkpoint=None, free_last_blocks=False, width_mult=1):
        super(TextSegament, self).__init__()
        self.act_fn = nn.SELU()
        self.bias = True

        self.layer_4x_conv = nn.Sequential(*Conv_block(int(24 * width_mult), 128, kernel_size=3, padding=1,
                                                       bias=self.bias, BN=True, activation=self.act_fn))
        # self.encoder.last_channel --|
        self.feature_pooling = ASP(int(320 * width_mult), out_channel=256)

        # decoder
        self.transition_2_decoder = nn.Sequential(*Conv_block(256 * 5, 128, kernel_size=1,
                                                              bias=self.bias, BN=True, activation=self.act_fn))

        self.smooth_4x_conv = nn.Sequential(*Conv_block(128 * 2, 64, kernel_size=1,
                                                        bias=self.bias, BN=True, activation=self.act_fn),
                                            *Conv_block(64, 64, kernel_size=3, padding=1, groups=64,
                                                        bias=self.bias, BN=True, activation=self.act_fn),
                                            *Conv_block(64, 32, kernel_size=3, padding=1,
                                                        bias=self.bias, BN=True, activation=self.act_fn))

        self.out_conv = nn.Sequential(*Conv_block(32, 2, kernel_size=3, padding=1,
                                                  bias=self.bias, BN=False, activation=None))
        self.softmax2d = nn.Softmax2d()  # to get mask
        if isinstance(self.act_fn, torch.nn.SELU):
            self.selu_init_params()
        else:
            self.initialize_weights()
        # use the pre-train weights to initialize the model
        self.encoder = MobileNetEncoder(encoder_checkpoint, free_last_blocks, width_mult=width_mult,
                                        activation=nn.ReLU6(), bias=False)  # may need to retrain the last 4 layers

    def forward(self, x):
        for index, layer in enumerate(self.encoder.features.children()):
            x = layer(x)
            if index == 2:
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
        with torch.no_grad():
            return self.forward(x)
