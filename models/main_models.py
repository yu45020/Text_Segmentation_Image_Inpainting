import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .BaseModels import BaseModule, Conv_block, Partial_Conv_block, DoubleUpSample, PartialConvBlock
from .MobileNetV2 import MobileNetV2


class MobileNetEncoder(MobileNetV2):
    def __init__(self, pre_train_checkpoint=None, drop_last2=True, add_partial=False):
        super(MobileNetEncoder, self).__init__(drop_last2=drop_last2, add_partial=add_partial)
        self.add_partial = add_partial
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
        self.features = self.make_inverted_resblocks(self.inverted_residual_setting, drop_last2)
        self.freeze_params(pre_train_checkpoint)

    def freeze_params(self, pre_train_checkpoint=None, free_last_blocks=2):
        if pre_train_checkpoint:
            if isinstance(pre_train_checkpoint, str):
                self.load_state_dict(torch.load(pre_train_checkpoint, map_location='cpu'))
            else:
                self.load_state_dict(pre_train_checkpoint)
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

        self.out_conv = nn.Sequential(*Conv_block(32, 2, kernel_size=3, padding=1,
                                                  bias=False, BN=False, activation=None))
        self.softmax2d = nn.Softmax2d()  # to get mask

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
        for index, layer in enumerate(self.encoder.features.children()):
            x = checkpoint(layer, x)
            if index == 2:
                layer_out4x = checkpoint(self.layer_4x_conv, x)

        x = checkpoint(self.feature_pooling, x)
        x = self.transition_2_decoder(x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = torch.cat([x, layer_out4x], dim=1)
        x = checkpoint(self.smooth_4x_conv, x)
        x = F.upsample(x, scale_factor=4, mode='bilinear')
        x = checkpoint(self.out_conv, x)
        return x


class ImageInpainting(BaseModule):
    # Free-Form Image Inpainting with Gated Convolution
    def __init__(self, pre_trained_encoder=None, fix_encoder_weights=True):
        super(ImageInpainting, self).__init__()
        self.encoder = MobileNetEncoder(add_partial=True)
        self.deco_act_fn = nn.SELU()
        # selu_init_params = 320
        self.decoder = self.make_decoder(320, 64, self.deco_act_fn)
        self.out_pconv = PartialConvBlock(64, 3, kernel_size=3, padding=1,
                                          bias=False, BN=True, activation=self.deco_act_fn)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)

        self.selu_init_params()
        if pre_trained_encoder:
            self.load_encoder_conv_weights(pre_trained_encoder, fix_encoder_weights)

    def load_encoder_conv_weights(self, pre_trained, fix_encoder_weights):
        own_encoder_state = list(self.encoder.state_dict())
        conv_names = list(filter(lambda x: 'mask_conv' not in x, own_encoder_state))
        if isinstance(pre_trained, str):
            encoder_weights = torch.load(pre_trained)
        else:
            encoder_weights = pre_trained
        assert len(conv_names) == len(encoder_weights.state_dict())
        new = {k: v for k, v in zip(conv_names, encoder_weights.state_dict().values())}
        self.encoder.load_state_dict(new)
        print("Load pre-trained encoder weights.")
        counter = 0
        if fix_encoder_weights:
            for name, model in self.encoder.named_modules():
                if isinstance(model, nn.Conv2d):
                    check = [x for x in conv_names if name in x]
                    if check:
                        for param in model.parameters():
                            param.requires_grad = False
                        counter += 1
            print("Fix {} of Conv weights".format(counter))

    def make_decoder(self, inchannel, out_channel, act_fn):
        channel_list = [64, 256, 256, 128, out_channel]
        m1 = Partial_Conv_block(inchannel, 64, 1, bias=False,
                                BN=True, activation=act_fn)

        m = [self.double_partial_block(channel_list[i], channel_list[i + 1], act_fn)
             for i in range(len(channel_list) - 1)]
        m1.extend(m)
        return nn.Sequential(*m1)

    def double_partial_block(self, in_channel, out_channel, act_fn):
        m = Partial_Conv_block(in_channel, out_channel, kernel_size=3,
                               padding=1, bias=False, BN=True, activation=nn.ReLU6())
        m += Partial_Conv_block(out_channel, out_channel, kernel_size=3, padding=1,
                                bias=False, BN=True, activation=act_fn)
        m += [DoubleUpSample(scale_factor=2, mode='bilinear')]

        return nn.Sequential(*m)

    def forward(self, args):
        x, mask = args
        x = x * mask

        x, mask = self.encoder((x, mask))
        x, mask = self.decoder((x, mask))
        x, mask = self.out_pconv((x, mask))
        return x
