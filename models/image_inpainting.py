import torch
from torch import nn

from models.BaseModels import BaseModule
from models.MobileNetV2 import PartialInvertedResidual
from models.partial_convolution import partial_convolution_block, DoubleUpSample


class ImageFill(BaseModule):
    def __init__(self):
        super(ImageFill, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3)
        # up-sampling must be nearest s.t. masks are either 0 or 1
        self.double_upscale = DoubleUpSample(scale_factor=2, mode='nearest')
        encoder = [
            # i, o, k, s, p, d, t, n
            [64, 128, 3, 2, 1, 1, 4, 2],
            [128, 256, 3, 2, 1, 1, 4, 2],
            [256, 256, 3, 2, 1, 1, 4, 2],

        ]
        self.encoder = nn.Sequential(
            partial_convolution_block(3, 64, 7, 2, 3, 1, bias=True, BN=False, activation=self.act_fn),
            *self.make_layers(encoder, use_1_conv=True, same_holes=True))

        # With large dilation rate, convolutions can easily overlap holes
        dilated_layers = [
            # in_c, out_c, k, s, p, d, t, n
            [256, 256, 3, 1, 2, 2, 4, 2],
            [256, 256, 3, 1, 4, 4, 4, 2],
            [256, 256, 3, 1, 8, 8, 4, 2]
        ]
        self.dilated_layers = nn.Sequential(*self.make_layers(dilated_layers, no_holes_1_conv=True, same_holes=True))

        decoder = [
            # in_c, out_c, k, s, p, d, t, n
            [256 + 256, 256, 3, 1, 1, 1, 2, 1],
            [256 + 128, 128, 3, 1, 1, 1, 2, 1],
            [128 + 64, 32, 3, 1, 1, 1, 2, 1],

        ]
        self.decoder = nn.Sequential(
            *self.make_layers(decoder, no_holes_1_conv=True, same_holes=True),
            partial_convolution_block(32 + 3, 3, 3, 1, 1, 1, bias=True, BN=False, activation=False))

    def make_layers(self, settings, use_1_conv=False, no_holes_1_conv=False, same_holes=False):
        # similar to mobile net v2's inverted residual block
        m = []
        for in_c, out_c, k, s, p, d, t, n in settings:
            layer = []
            for i in range(n):
                if i == 0:
                    layer.append(
                        PartialInvertedResidual(in_c, out_c, k, s, p, d, t, bias=False, BN=True, activation=self.act_fn,
                                                use_1_conv=use_1_conv, no_holes_1_conv=no_holes_1_conv,
                                                same_holes=same_holes))
                else:
                    layer.append(
                        PartialInvertedResidual(in_c, out_c, k, 1, p, d, t, bias=False, BN=True, activation=self.act_fn,
                                                use_1_conv=use_1_conv, no_holes_1_conv=no_holes_1_conv,
                                                same_holes=same_holes))
                in_c = out_c

            m.append(nn.Sequential(*layer))
        return m

    def forward(self, args):
        # mask: 1: ground truth, 0: holes

        x, mask = args
        feature_x, feature_mask = [x], [mask]
        for index, layer in enumerate(self.encoder):
            x, mask = layer((x, mask))
            feature_x.append(x)
            feature_mask.append(mask)

        feature_x = feature_x[:-1]
        feature_mask = feature_mask[:-1]
        x, mask = self.dilated_layers((x, mask))

        for layer in self.decoder:
            x_up, mask_up = self.double_upscale((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))
        return x


# #
# import random
#
# random.seed(0)
# model = ImageFill()
# model.total_parameters()
# x = torch.randn(1, 3, 512, 512)
# mask = torch.randn(1, 3, 512, 512) > 0
# mask = mask.float()
# # #
# import time
#
# st = time.time()
# with torch.no_grad():
#     a = model((x, mask))
#     # a = model(x)
# print(time.time() - st)

# #


class ImageFillOrigin(BaseModule):
    def __init__(self):
        super(ImageFillOrigin, self).__init__()
        # self.act_fn = nn.LeakyReLU(0.3)
        # up-sampling must be nearest s.t. masks are either 0 or 1
        self.double_upscale = DoubleUpSample(scale_factor=2, mode='nearest')
        encoder = [
            # i, o, k, s, p, d, t, n
            [64, 128, 5, 2, 2, 1, 1, 1],
            [128, 256, 5, 2, 2, 1, 1, 1],
            [256, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],

        ]
        # self.encoder = nn.Sequential(
        #     partial_convolution_block(3, 64, 7, 2, 3, 1, bias=True, BN=False, activation=self.act_fn),
        #     *self.make_layers(encoder, use_1_conv=True))

        self.encoder = nn.Sequential(
            partial_convolution_block(3, 64, 7, 2, 3, 1, bias=True, BN=False, activation=nn.ReLU(), same_holes=True),
            *self.make_layer_v2(encoder, act_fn=nn.ReLU(), same_holes=True))

        decoder = [
            # in_c, out_c, k, s, p, d, t, n
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 256, 256, 3, 1, 1, 1, 1, 1],
            [256 + 128, 128, 3, 1, 1, 1, 1, 1],
            [128 + 64, 64, 3, 1, 1, 1, 1, 1],

        ]
        # self.decoder = nn.Sequential(
        #     *self.make_layers(decoder, drop_first_1_conv=True),
        #     partial_convolution_block(32 + 3, 3, 3, 1, 1, 1, bias=True, BN=False, activation=False))

        # decoder has no holes in mask, so its convolution with encoder will fill encoder's holes
        self.decoder = nn.Sequential(
            *self.make_layer_v2(decoder, act_fn=nn.LeakyReLU(0.2), same_holes=False),
            partial_convolution_block(64 + 3, 3, 3, 1, 1, 1, bias=True, BN=False, activation=False, same_holes=False))

    def make_layer_v2(self, settings, act_fn, no_holes_1_conv=False, same_holes=False):
        m = []
        for in_c, out_c, k, s, p, d, t, n in settings:
            layer = partial_convolution_block(in_c, out_c, k, s, p, d,
                                              groups=1, BN=True, activation=act_fn, bias=False,
                                              no_holes_1_conv=no_holes_1_conv, same_holes=same_holes)
            m.append(nn.Sequential(layer))
        return m

    def forward(self, args):
        # mask: 1: ground truth, 0: holes

        x, mask = args
        feature_x, feature_mask = [x], [mask]
        for index, layer in enumerate(self.encoder):
            x, mask = layer((x, mask))
            feature_x.append(x)
            feature_mask.append(mask)
            # unique, count = np.unique(mask.numpy(), return_counts=True)
            # print(f"Unique number {unique} with counts: {count}")

        # raise NotImplemented

        # del feature_pool_m
        # del feature_pool_x
        feature_x = feature_x[:-1]
        feature_mask = feature_mask[:-1]
        for layer in self.decoder:
            x_up, mask_up = self.double_upscale((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))

            # unique, count = np.unique(mask.numpy(), return_counts=True)
            # print(f"Unique number {unique} with counts: {count}")

        return x


class DoublePartialResidual(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, expansion=1, BN=True, activation=True, bias=False,
                 use_1_conv=False, no_holes_1_conv=False, same_holes=False,
                 dilation_rate=(1, 1), *args, **kwargs):
        super(DoublePartialResidual, self).__init__()

        self.conv1 = partial_convolution_block(in_channels, out_channels, kernel_size, stride, padding=dilation_rate[0],
                                               dilation=dilation_rate[0],
                                               BN=BN, activation=activation, bias=bias,
                                               use_1_conv=use_1_conv, no_holes_1_conv=no_holes_1_conv,
                                               same_holes=same_holes)
        self.conv2 = partial_convolution_block(out_channels, out_channels, kernel_size, 1, padding=dilation_rate[1],
                                               dilation=dilation_rate[1],
                                               BN=BN, activation=activation, bias=bias,
                                               use_1_conv=use_1_conv, no_holes_1_conv=no_holes_1_conv,
                                               same_holes=same_holes)

    def forward(self, args):
        x, mask = args
        out_x, out_mask = self.conv1((x, mask))
        x, mask = self.conv2((out_x, out_mask))
        return x + out_x, mask


class ImageFillOriginV2(BaseModule):
    def __init__(self):
        super(ImageFillOriginV2, self).__init__()
        # self.act_fn = nn.LeakyReLU(0.3)
        # up-sampling must be nearest s.t. masks are either 0 or 1
        self.double_upscale = DoubleUpSample(scale_factor=2, mode='nearest')
        encoder = [
            # i, o, k, s, p, d, t, n
            [64, 128, 3, 2, 1, 1, 1, 1],
            [128, 256, 3, 2, 1, 1, 1, 1],
            [256, 256, 3, 2, 1, 1, 1, 1],
            [256, 256, 3, 2, 1, 1, 1, 1],
            [256, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],
            [512, 512, 3, 2, 1, 1, 1, 1],

        ]
        # self.encoder = nn.Sequential(
        #     partial_convolution_block(3, 64, 7, 2, 3, 1, bias=True, BN=False, activation=self.act_fn),
        #     *self.make_layers(encoder, use_1_conv=True))

        self.encoder = nn.Sequential(
            partial_convolution_block(3, 64, 5, 2, 2, 1, bias=False, BN=True,
                                      activation=nn.LeakyReLU(0.2), same_holes=True),
            *self.make_layer_v2(encoder, act_fn=nn.LeakyReLU(0.2), same_holes=True, dilation_rate=(1, 2)))

        decoder = [
            # in_c, out_c, k, s, p, d, t, n
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 512, 512, 3, 1, 1, 1, 1, 1],
            [512 + 256, 256, 3, 1, 1, 1, 1, 1],
            [256 + 256, 256, 3, 1, 1, 1, 1, 1],
            [256 + 256, 256, 3, 1, 1, 1, 1, 1],
            [256 + 128, 128, 3, 1, 1, 1, 1, 1],
            [128 + 64, 64, 3, 1, 1, 1, 1, 1],

        ]

        self.decoder = nn.Sequential(
            *self.make_layer_v2(decoder, act_fn=nn.LeakyReLU(0.2), same_holes=False, dilation_rate=(2, 1)),
            partial_convolution_block(64 + 3, 3, 3, 1, 1, 1, bias=True, BN=False,
                                      activation=nn.ReLU(), same_holes=False))

    @staticmethod
    def make_layer_v2(settings, act_fn, same_holes=False, dilation_rate=(1, 1)):
        m = []
        for in_c, out_c, k, s, p, d, t, n in settings:
            layer = DoublePartialResidual(in_c, out_c, k, s, p, d,
                                          BN=True, activation=act_fn, bias=False,
                                          same_holes=same_holes, dilation_rate=dilation_rate)
            m.append(nn.Sequential(layer))
        return m

    def forward(self, args):
        # mask: 1: ground truth, 0: holes

        x, mask = args
        feature_x, feature_mask = [x], [mask]
        for index, layer in enumerate(self.encoder):
            x, mask = layer((x, mask))
            feature_x.append(x)
            feature_mask.append(mask)

        feature_x = feature_x[:-1]
        feature_mask = feature_mask[:-1]
        for layer in self.decoder:
            x_up, mask_up = self.double_upscale((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))

        return x


class ImageFillOriginV3(BaseModule):
    def __init__(self):
        super(ImageFillOriginV3, self).__init__()
        # self.act_fn = nn.LeakyReLU(0.3)
        # up-sampling must be nearest s.t. masks are either 0 or 1
        self.double_upscale = DoubleUpSample(scale_factor=2, mode='nearest')
        encoder = [
            # i, o, k, s, p, d, t, n
            [64, 128, 5, 2, 2, 1, 1, 1],
            [128, 256, 5, 2, 2, 1, 1, 1],
            [256, 256, 3, 2, 1, 1, 1, 1],

        ]

        self.encoder = nn.Sequential(
            partial_convolution_block(3, 64, 7, 2, 3, 1, bias=True, BN=False, activation=nn.ReLU(), same_holes=True),
            *self.make_layer_v2(encoder, act_fn=nn.ReLU(), same_holes=True))

        dilated = [
            [256, 256, 3, 1, 2, 2, 1, 1],
            [256, 256, 3, 1, 4, 4, 1, 1],
            [256, 256, 3, 1, 8, 8, 1, 1],

        ]
        self.dilated = nn.Sequential(*self.make_layer_v2(dilated, act_fn=nn.LeakyReLU(0.2), same_holes=True))

        decoder = [
            # in_c, out_c, k, s, p, d, t, n
            [256 + 256, 256, 3, 1, 1, 1, 1, 1],
            [256 + 128, 128, 3, 1, 1, 1, 1, 1],
            [128 + 64, 64, 3, 1, 1, 1, 1, 1],

        ]

        self.decoder = nn.Sequential(
            *self.make_layer_v2(decoder, act_fn=nn.LeakyReLU(0.2), same_holes=False),
            partial_convolution_block(64 + 3, 3, 3, 1, 1, 1, bias=True, BN=False, activation=False, same_holes=False))

    @staticmethod
    def make_layer_v2(settings, act_fn, same_holes=False):
        m = []
        for in_c, out_c, k, s, p, d, t, n in settings:
            layer = DoublePartialResidual(in_c, out_c, k, s, p, d,
                                          BN=True, activation=act_fn, bias=False,
                                          same_holes=same_holes)
            m.append(nn.Sequential(layer))
        return m

    def forward(self, args):
        # mask: 1: ground truth, 0: holes

        x, mask = args
        feature_x, feature_mask = [x], [mask]
        for index, layer in enumerate(self.encoder):
            x, mask = layer((x, mask))
            feature_x.append(x)
            feature_mask.append(mask)

        feature_x = feature_x[:-1]
        feature_mask = feature_mask[:-1]
        x, mask = self.dilated((x, mask))
        for layer in self.decoder:
            x_up, mask_up = self.double_upscale((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))

        return x
