import torch
from torch import nn
from torch.nn import functional as F

from models.MobileNetV2 import PartialInvertedResidual
from .BaseModels import BaseModule
from .partial_convolution import PartialGatedConv
from .spectral_norm import spectral_norm


class ImageFill(BaseModule):
    def __init__(self, width_mult=2):
        super(ImageFill, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3, inplace=True)

        self.input_conv = PartialGatedConv(3, 32 * 2, kernel_size=3, stride=2,
                                           padding=1, bias=True, activation=self.act_fn)
        self.encoder = self.make_encoder(32, width_mult)

        cat_feat_num = sum([i[0].out_channels for i in self.encoder[3:]])
        self.decoder = self.make_decoder(in_channel=cat_feat_num, width_mul=width_mult)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def make_encoder(self, in_channel, width_mul):
        settings = [
            # t, c, n, s, dila  # input output
            [1, 16, 1, 1, 1],  # 1/2 ---> 1/2
            [6, 24, 2, 2, 1],  # 1/2 ---> 1/4
            [6, 32, 3, 2, 1],  # 1/4 ---> 1/8
            [6, 64, 4, 1, 2],  # <-- add astrous conv and keep 1/8
            [6, 96, 3, 1, 4],
            [6, 160, 3, 1, 8],
        ]
        return self.make_res_blocks(settings, in_channel, width_mul)

    def make_decoder(self, in_channel, width_mul):
        settings = [
            # t, c, n, s, dila
            [6, 160, 2, 1, 1],
            [6, 64, 2, 1, 1],
            [1, 16, 1, 1, 1]
        ]
        return self.make_res_blocks(settings, in_channel, width_mul, upsample=True)

    def make_res_blocks(self, settings, in_channel, width_mul, upsample=False):
        features = []
        for t, c, n, s, d in settings:
            out_channel = self._make_divisible(c * width_mul, divisor=8)
            # out_channel = int(c * self.width_mult)
            block = []
            for i in range(n):
                if i == 0:
                    block.append(PartialInvertedResidual(in_channel, out_channel, s, t, d,
                                                         activation=self.act_fn, bias=True))
                else:
                    block.append(PartialInvertedResidual(in_channel, out_channel, 1, t, d,
                                                         activation=self.act_fn, bias=True))
                in_channel = out_channel
            if upsample:
                block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            features.append(nn.Sequential(*block))

        return nn.Sequential(*features)

    @staticmethod
    def _make_divisible(v, divisor=8, min_value=None):
        # https://github.com/tensorflow/models/blob/7367d494135368a7790df6172206a58a2a2f3d40/research/slim/nets/mobilenet/mobilenet.py#L62
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor

        return new_v

    def forward(self, x):
        x = self.input_conv(x)
        for layer in self.encoder[:3]:
            x = layer(x)
        features = []
        for layer in self.encoder[3:]:
            x = layer(x)
            features.append(x)

        x = self.decoder(torch.cat(features, dim=1))
        x = self.out_conv(x)
        return x


class SPGAN(BaseModule):
    def __init__(self):
        super(SPGAN, self).__init__()
        self.act_fn = nn.LeakyReLU(0.3, inplace=True)
        channel_list = [3, 128, 128, 256, 256, 256]
        m = []
        for i in range(len(channel_list) - 1):
            m.append(self.separable_conv(channel_list[i], channel_list[i + 1]))
        self.features = nn.Sequential(*m)
        self.out_linear = spectral_norm(nn.Linear(256 * 16, 1))

    def sp_cov(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0),
               dilation=1, groups=1, bias=True, activation=None):
        m = nn.Sequential(spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                  padding, dilation, groups, bias)),
                          activation)
        return m

    def separable_conv(self, in_channel, out_channel):
        # similar to EffNet
        mid_channel = in_channel // 2 * 3
        m = nn.Sequential(
            self.sp_cov(in_channel, mid_channel, kernel_size=1, bias=True, activation=self.act_fn),
            self.sp_cov(mid_channel, mid_channel, kernel_size=(1, 5), bias=True, groups=mid_channel,
                        activation=self.act_fn, padding=(0, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            self.sp_cov(mid_channel, mid_channel, kernel_size=(5, 1), bias=False, groups=mid_channel,
                        activation=self.act_fn, padding=(2, 0)),
            self.sp_cov(mid_channel, out_channel, kernel_size=(2, 1), stride=(2, 1),
                        bias=False, activation=self.act_fn),
        )
        return m

    def forward(self, x):
        x = self.features(x)
        batch = x.size(0)
        x = F.adaptive_avg_pool2d(x).view(batch, -1)  #
        x = self.out_linear(x)
        return x

#
#
# Generator = ImageFill(width_mult=2)
# Generator.total_parameters()

# Discriminator = SPGAN()
# optim_gen = optim.SGD(Generator.parameters(), lr=1e-3)
# optim_disc = optim.SGD(Discriminator.parameters(), lr=1e-3)
# loss_l1 = nn.L1Loss()
#
# out = Generator(z)
# loss_disc = F.relu(1.0 - Discriminator(target)).mean() + F.relu(1.0 + Discriminator(out)).mean()
# loss_disc.backward()
# optim_disc.step()
# optim_disc.zero_grad()
#
# loss_gen = -Discriminator().mean()
# loss = loss_gen + loss_l1(out, target)
# loss.backward()
# optim_gen.step()
# optim_disc.zero_grad()
# optim_gen.zero_grad()
