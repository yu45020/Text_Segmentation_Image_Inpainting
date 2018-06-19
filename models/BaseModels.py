import math
import warnings
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.functional import avg_pool2d, upsample

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

    def selu_init_params(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                i.weight.data.normal_(0.0, 1.0 / math.sqrt(i.weight.numel()))
                if i.bias is not None:
                    i.bias.data.fill_(0)


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
    # mask is binary, 0 is masked point, 1 is not

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, BN=True, activation=None):
        super(PartialConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        self.mask_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                 padding, dilation, groups, bias))

        out = []
        if BN:
            out.append(nn.BatchNorm2d(out_channels))
        if activation:
            out.append(activation)
        if len(out):
            self.out = nn.Sequential(*out)

    def forward(self, args):
        x, mask = args
        feature = self.conv(x)
        mask = self.mask_conv(mask)
        gate = torch.sigmoid(mask)
        x = feature * gate
        if hasattr(self, 'out'):
            x = self.out(x)
        return x, mask

def Partial_Conv_block(in_channels, out_channels, kernel_size, stride=1,
                       padding=0, dilation=1, groups=1, bias=True,
                       BN=False, activation=None):
    m = PartialConvBlock(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, BN, activation)

    return [m]


class DoubleAvdPool(nn.AvgPool2d):
    def __init__(self, kernel_size):
        super(DoubleAvdPool, self).__init__(kernel_size=kernel_size)
        self.kernel_size = kernel_size

    def forward(self, *args):
        return tuple(map(lambda x: avg_pool2d(x, kernel_size=self.kernel_size), *args))


class DoubleUpSample(nn.Upsample):
    def __init__(self, scale_factor, mode):
        super(DoubleUpSample, self).__init__(scale_factor=scale_factor, mode=mode)
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, *args):
        return tuple(map(lambda x: upsample(x, scale_factor=self.scale_factor, mode=self.mode), *args))
