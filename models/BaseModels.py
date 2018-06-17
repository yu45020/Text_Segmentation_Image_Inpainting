import math
import torch
from contextlib import contextmanager
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential, checkpoint
import warnings

warnings.simplefilter('ignore')


def weights_init(init_type='gaussian'):
    # copy from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class BaseModule(nn.Module):
    def __init__(self):
        self.act_fn = None
        super(BaseModule, self).__init__()

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
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param.data)
                except Exception as e:
                    print("Parameter {} fails to load.".format(name))
                    print("-----------------------------------------")
                    print(e)
            else:
                print("Parameter {} is not in the model. ".format(name))

    @contextmanager
    def set_activation_inplace(self):
        if hasattr(self, 'act_fn') and hasattr(self.act_fn, 'inplace'):
            # save memory
            self.act_fn.inplace = True
            yield
            self.act_fn.inplace = False
        else:
            yield

    def total_parameters(self):
        return sum([i.numel() for i in self.parameters()])

    def forward(self, *x):
        raise NotImplementedError


def Conv_block(in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True,
               BN=False, activation=None):
    m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                   padding, dilation, groups, bias)]
    if BN:
        m.append(nn.BatchNorm2d(out_channels))
    if activation:
        m.append(activation)
    return m


class PartialConvBlock(BaseModule):
    # reference :
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 BN=False, activation=None):
        super(PartialConvBlock, self).__init__()
        self.pconv = nn.Sequential(*Conv_block(in_channels, out_channels, kernel_size, stride,
                                               padding, dilation, groups, bias,
                                               BN, activation))

        self.mask_conv = nn.Sequential(*Conv_block(in_channels, out_channels, kernel_size, stride,
                                                   padding=0, dilation=dilation, groups=groups, bias=False,
                                                   BN=BN, activation=nn.ReLU()))

        for param in self.mask_conv.parameters():
            torch.nn.init.constant_(param, 1)
            param.requires_grad = False

    def forward(self, x, input_mask):
        output = self.pconv(x)
        output_mask = self.mask_conv(input_mask)
        out_holes = output_mask != 0.0
        output[out_holes] = output[out_holes] / torch.sum(output_mask)

        in_holes = output_mask == 0.0
        output[in_holes].fill_(0.0)

        output_mask[out_holes].fill_(1.0)
        output_mask[in_holes].fill_(0.0)

        return output, output_mask


def Partial_Conv_block(in_channels, out_channels, kernel_size, stride=1,
                       padding=0, dilation=1, groups=1, bias=True,
                       BN=False, activation=None):
    m = PartialConvBlock(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, BN, activation)

    return [m]
