# reference:
# https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# https://arxiv.org/pdf/1801.04381.pdf
import re

import torch

from torch import nn
import math
from .BaseModels import BaseModule, Conv_block, Partial_Conv_block
from torch.utils.checkpoint import checkpoint


class MobileNetV2(BaseModule):
    def __init__(self, drop_last2=True, init_weights=False, width_mult=1,
                 n_class=1000, input_size=224, add_partial=False):

        super(MobileNetV2, self).__init__()
        self.add_partial = add_partial
        self.conv_block = Conv_block if not add_partial else Partial_Conv_block
        self.res_block = InvertedResidual if not add_partial else PartialInvertedResidual

        self.width_mult = width_mult
        self.inverted_residual_setting = [
            # t, c, n, s, dial
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 2, 1],
            [6, 96, 3, 1, 1],
            [6, 160, 3, 2, 1],
            [6, 320, 1, 1, 1],
        ]
        self.last_channel = 0  # last one is avg pool
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting,
                                                     drop_last2, input_size)
        if not drop_last2:
            self.classifier = self.make_classifier(n_class)
        if init_weights:
            self.initialize_weights()


    def make_inverted_resblocks(self, settings, drop_last2, input_size=224):
        in_channel = int(32 * self.width_mult)
        last_channel = int(1280 * self.width_mult) if self.width_mult > 1 else 1280
        # first_layer
        features = [nn.Sequential(*self.conv_block(3, in_channel, kernel_size=3, stride=2,
                                                   padding=1, bias=False,
                                                   BN=True, activation=nn.ReLU6()))]

        for t, c, n, s, d in settings:
            out_channel = int(c * self.width_mult)
            block = []
            for i in range(n):
                if i == 0:
                    block.append(self.res_block(in_channel, out_channel, s, t, d))
                else:
                    block.append(self.res_block(in_channel, out_channel, 1, t, 1))
                in_channel = out_channel
            features.append(nn.Sequential(*block))
        # last layer
        self.last_channel = out_channel

        if not drop_last2:
            features.extend(self.conv_block(in_channel, last_channel, 1, 1, 0, bias=False,
                                            BN=True, activation=nn.ReLU6()))
            features.append(nn.AvgPool2d(input_size // 32))
            self.last_channel = last_channel
        return nn.Sequential(*features)

    def make_classifier(self, n_class):
        m = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class)
        )
        return m

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, n)
                m.bias.data.zero_()

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        if self.add_partial:  # remove all mask conv
            own_name = list(filter(lambda x: 'mask_conv' not in x, list(own_state)))[:len(own_state)]
            state_dict = {k: v for k, v in zip(own_name, state_dict.values())}
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param.data)
                except Exception as e:
                    print("-----------------------------------------")
                    print("Parameter {} fails to load.".format(name))
                    print(e)
            else:
                print("Parameter {} is not in the model. ".format(name))

    def forward(self, x):
        return self.features(x)

    def forward_checkpoint(self, x):
        with self.set_activation_inplace():
            return checkpoint(self.forward, x)

    def forward_classifier(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        return self.classifier(x)


class InvertedResidual(BaseModule):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, dilation, conv_block_fn=Conv_block):
        super(InvertedResidual, self).__init__()
        self.conv_bloc = conv_block_fn
        self.stride = stride
        assert stride in [1, 2]

        self.res_connect = self.stride == 1 and in_channel == out_channel
        self.conv = self.make_body(in_channel, out_channel, stride, expand_ratio, dilation)

    def make_body(self, in_channel, out_channel, stride, expand_ratio, dilation):
        # standard convolution
        mid_channel = in_channel * expand_ratio
        m = self.conv_bloc(in_channel, mid_channel,
                           1, 1, 0, bias=False,
                           BN=True, activation=nn.ReLU6())
        # depth-wise separable convolution
        m += self.conv_bloc(mid_channel, mid_channel, 3, stride, padding=1 + (dilation - 1),
                            dilation=dilation, groups=mid_channel, bias=False,
                            BN=True, activation=nn.ReLU6())
        m += self.conv_bloc(mid_channel, out_channel, 1, 1, 0, bias=False, BN=True)
        return nn.Sequential(*m)

    def forward(self, x):
        if self.res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def forward_checkpoint(self, x):
        if self.res_connect:
            return x + checkpoint(self.conv, x)
        else:
            return checkpoint(self.conv, x)


#
#
class PartialInvertedResidual(InvertedResidual):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, dilation, conv_block_fn=Partial_Conv_block):
        super(PartialInvertedResidual, self).__init__(in_channel=in_channel,
                                                      out_channel=out_channel,
                                                      stride=stride,
                                                      expand_ratio=expand_ratio,
                                                      dilation=dilation,
                                                      conv_block_fn=conv_block_fn)

    def forward(self, args):
        if self.res_connect:
            x, mask = args
            out, out_mask = self.conv((x, mask))
            out = out + x

            out_mask = out_mask + mask
            out_mask = torch.clamp(out_mask, min=0, max=1)
            return out, out_mask
        else:
            return self.conv(args)

    def forward_checkpoint(self, args):
        if self.res_connect:
            x, mask = args
            out, out_mask = checkpoint(self.conv, (x, mask))
            out = out + x

            out_mask = out_mask + mask
            out_mask = torch.clamp(out_mask, min=0, max=1)
            return out, out_mask
        else:
            return checkpoint(self.conv, args)
