import torch
from torch import nn

from .BaseModels import BaseModule
from .MobileNetV2 import PartialInvertedResidual
from .partial_convolution import partial_convolution_block, DoubleUpSample


class ImageFill(BaseModule):
    def __init__(self):
        super(ImageFill, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3)  # be consistent with the setting in place batch norm
        # up-sampling must be nearest s.t. masks are either 0 or 1
        self.double_upscale = DoubleUpSample(scale_factor=2, mode='nearest')
        encoder = [
            # i, o, k, s, p, d, t, n
            [64, 128, 3, 2, 1, 1, 4, 2],
            [128, 256, 3, 2, 1, 1, 4, 3],  # <---usually the updated masks have no holes after this block
            [256, 256, 3, 2, 1, 1, 4, 2],

        ]
        self.encoder = nn.Sequential(partial_convolution_block(3, 64, 7, 2, 3, 1, BN=False, activation=self.act_fn),
                                     *self.make_layers(encoder))

        # With large dilation rate, convolutions can easily overlap holes
        feature_pooling = [
            # in_c, out_c, k, s, p, d, t, n
            [256, 256, 3, 1, 2, 2, 2, 1],
            [256, 256, 3, 1, 5, 5, 2, 1],
            [256, 256, 3, 1, 9, 9, 2, 1]
        ]
        self.feature_pooling = nn.Sequential(*self.make_layers(feature_pooling))

        # 3x3 convolution is used to reduce gridding artifact. See "Dilated Residual Networks"
        self.feature_cat = nn.Sequential(*self.make_layers([[256 * 4, 256, 3, 1, 1, 1, 1, 1]]))
        decoder = [
            # in_c, out_c, k, s, p, d, t, n
            [256 + 256, 256, 3, 1, 1, 1, 2, 2],
            [256 + 128, 128, 3, 1, 1, 1, 2, 2],
            [128 + 64, 32, 3, 1, 1, 1, 2, 2],

        ]
        self.decoder = nn.Sequential(*self.make_layers(decoder),
                                     partial_convolution_block(32 + 3, 3, 3, 1, 1, 1, BN=False, activation=False))

    def make_layers(self, settings):
        # similar to mobile net v2's inverted residual block
        m = []
        for in_c, out_c, k, s, p, d, t, n in settings:
            layer = []
            for i in range(n):
                if i == 0:
                    layer.append(PartialInvertedResidual(in_c, out_c, k, s, p, d, t, BN=True, activation=self.act_fn))
                else:
                    layer.append(PartialInvertedResidual(in_c, out_c, k, 1, p, d, t, BN=True, activation=self.act_fn))
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

            # unique, count = np.unique(mask.numpy(), return_counts=True)
            # print(f"Unique number {unique} with counts: {count}")

        feature_pool_x = [feature_x.pop(-1)]
        feature_pool_m = [feature_mask.pop(-1)]
        for layer in self.feature_pooling:
            x, mask = layer((x, mask))
            feature_pool_x.append(x)
            feature_pool_m.append(mask)

            # unique, count = np.unique(mask.numpy(), return_counts=True)
            # print(f"Feature Pooling Unique number {unique} with counts: {count}")

        x = torch.cat(feature_pool_x, dim=1)
        mask = torch.cat(feature_pool_m, dim=1)
        x, mask = self.feature_cat((x, mask))

        # unique, count = np.unique(mask.numpy(), return_counts=True)
        # print(f"Unique number {unique} with counts: {count}")
        # raise NotImplemented

        # del feature_pool_m
        # del feature_pool_x
        for layer in self.decoder:
            x_up, mask_up = self.double_upscale((x, mask))
            x_h = torch.cat([x_up, feature_x.pop(-1)], dim=1)
            mask_h = torch.cat([mask_up, feature_mask.pop(-1)], dim=1)
            x, mask = layer((x_h, mask_h))

        return x
