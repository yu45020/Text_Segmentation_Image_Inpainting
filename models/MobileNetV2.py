# reference:
# https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
# https://arxiv.org/pdf/1801.04381.pdf

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .BaseModels import BaseModule, Conv_block
from .partial_convolution import partial_gated_conv_block


class MobileNetV2(BaseModule):
    def __init__(self, width_mult=1, add_partial=False, activation=nn.ReLU6(), bias=False):

        super(MobileNetV2, self).__init__()
        self.add_partial = add_partial
        self.conv_block = Conv_block if not add_partial else partial_gated_conv_block
        self.res_block = InvertedResidual if not add_partial else PartialInvertedResidual
        self.act_fn = activation
        self.bias = bias
        self.width_mult = width_mult
        self.out_stride = 32  # 1/32 of input size
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
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting)

    def make_inverted_resblocks(self, settings):
        in_channel = int(32 * self.width_mult)

        # first_layer
        features = [nn.Sequential(*self.conv_block(3, in_channel, kernel_size=3, stride=2,
                                                   padding=(3 - 1) // 2, bias=self.bias,
                                                   BN=True, activation=self.act_fn))]

        for t, c, n, s, d in settings:
            out_channel = int(c * self.width_mult)
            block = []
            for i in range(n):
                if i == 0:
                    block.append(self.res_block(in_channel, out_channel, s, t, d,
                                                activation=self.act_fn, bias=self.bias))
                else:
                    block.append(self.res_block(in_channel, out_channel, 1, t, 1,
                                                activation=self.act_fn, bias=self.bias))
                in_channel = out_channel
            features.append(nn.Sequential(*block))
        # last layer
        self.last_channel = out_channel
        return nn.Sequential(*features)

    def load_pre_train_checkpoint(self, pre_train_checkpoint, free_last_blocks):
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

    #    for partial conv ---- will not use
    # def load_state_dict(self, state_dict, strict=True):
    #     own_state = self.state_dict()
    #     if self.add_partial:  # remove all mask conv
    #         own_name = list(filter(lambda x: 'mask_conv' not in x, list(own_state)))[:len(own_state)]
    #         state_dict = {k: v for k, v in zip(own_name, state_dict.values())}
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             try:
    #                 own_state[name].copy_(param.data)
    #             except Exception as e:
    #                 print("-----------------------------------------")
    #                 print("Parameter {} fails to load.".format(name))
    #                 print(e)
    #         else:
    #             print("Parameter {} is not in the model. ".format(name))

    def forward(self, x):
        return self.features(x)

    def forward_checkpoint(self, x):
        with self.set_activation_inplace():
            return checkpoint(self.forward, x)


class InvertedResidual(BaseModule):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, dilation, conv_block_fn=Conv_block,
                 activation=nn.ReLU6(), bias=False):
        super(InvertedResidual, self).__init__()
        self.conv_bloc = conv_block_fn
        self.stride = stride
        self.act_fn = activation
        self.bias = bias
        self.in_channels = in_channel
        self.out_channels = out_channel
        # assert stride in [1, 2]

        self.res_connect = self.stride == 1 and in_channel == out_channel
        self.conv = self.make_body(in_channel, out_channel, stride, expand_ratio, dilation)

    def make_body(self, in_channel, out_channel, stride, expand_ratio, dilation):
        # standard convolution
        mid_channel = in_channel * expand_ratio
        m = self.conv_bloc(in_channel, mid_channel,
                           1, 1, 0, bias=self.bias,
                           BN=True, activation=self.act_fn)
        # depth-wise separable convolution
        m += self.conv_bloc(mid_channel, mid_channel, 3, stride, padding=1 + (dilation - 1),
                            dilation=dilation, groups=mid_channel, bias=self.bias,
                            BN=True, activation=self.act_fn)
        # linear to preserve info : see the section: linear bottleneck. Removing the activation improves the result
        m += self.conv_bloc(mid_channel, out_channel, 1, 1, 0, bias=self.bias, BN=True, activation=None)
        return nn.Sequential(*m)

    def forward(self, x):
        if self.res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def forward_checkpoint(self, x):
        with torch.no_grad():
            return self.forward(x)


class PartialInvertedResidual(InvertedResidual):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, dilation, conv_block_fn=partial_gated_conv_block,
                 activation=nn.ReLU6(), bias=False):
        super(PartialInvertedResidual, self).__init__(in_channel=in_channel,
                                                      out_channel=out_channel,
                                                      stride=stride,
                                                      expand_ratio=expand_ratio,
                                                      dilation=dilation,
                                                      conv_block_fn=conv_block_fn)
        self.act_fn = activation
        self.bias = bias

    def forward(self, args):
        if self.res_connect:
            x, mask = args
            out, out_mask = self.conv((x, mask))
            out = out + x

            out_mask = out_mask + mask
            # out_mask = torch.clamp(out_mask, min=0, max=1)
            return out, out_mask
        else:
            return self.conv(args)

    def forward_checkpoint(self, args):
        with torch.no_grad():
            return self.forward(args)


class DilatedMobileNetV2(MobileNetV2):
    def __init__(self, width_mult=1, activation=nn.ReLU6(), bias=False, add_partial=False, ):
        super(DilatedMobileNetV2, self).__init__(width_mult=width_mult, activation=activation,
                                                 bias=bias, add_partial=add_partial, )
        self.add_partial = add_partial
        self.bias = bias
        self.width_mult = width_mult
        self.act_fn = activation
        self.out_stride = 8
        # # Rethinking Atrous Convolution for Semantic Image Segmentation
        self.inverted_residual_setting = [
            # t, c, n, s, dila  # input size
            [1, 16, 1, 1, 1],  # 1/2
            [6, 24, 2, 2, 1],  # 1/4
            [6, 32, 3, 2, 1],  # 1/8
            [6, 64, 4, 1, 2],  # <-- add astrous conv and keep 1/8
            [6, 96, 3, 1, 4],
            [6, 160, 3, 1, 8],
            [6, 320, 1, 1, 16],
        ]
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting)


class MobileNetV2Classifier(BaseModule):
    def __init__(self, num_class, input_size=512, width_mult=1.4):
        super(MobileNetV2Classifier, self).__init__()
        self.act_fn = nn.SELU(inplace=True)
        self.feature = DilatedMobileNetV2(width_mult=width_mult, activation=self.act_fn,
                                          bias=False, add_partial=False)
        self.pre_classifier = nn.Sequential(
            *Conv_block(self.feature.last_channel, 1024, kernel_size=1,
                        bias=False, BN=True, activation=self.act_fn),
            nn.AvgPool2d(input_size // self.feature.out_stride))  # b,c, 1,1

        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1024, num_class)
        )
        if isinstance(self.act_fn, nn.SELU):
            self.selu_init_params()
        else:
            self.initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.pre_classifier(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
