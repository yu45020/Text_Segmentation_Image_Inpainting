import torch
from torch import nn

from models.common import CNNLSTMClassifier
from .BaseModels import BaseModule, Conv_block, DSConvBlock

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class ResidualBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, BN=True, activation=None, expand_channel_first=True):
        super(ResidualBlock, self).__init__()
        if expand_channel_first:
            middle_channel = out_channels
        else:
            middle_channel = in_channels
        self.conv = nn.Sequential(
            DSConvBlock(in_channels, middle_channel, kernel_size, 1, padding,
                        dilation, bias, BN, activation),
            DSConvBlock(middle_channel, out_channels, kernel_size, 1, padding,
                        dilation, bias, BN, activation),
            DSConvBlock(out_channels, out_channels, kernel_size, stride, padding,
                        dilation, bias, BN, activation=None)
        )

        if (stride > 1) or (in_channels != out_channels):
            self.residual_conv = nn.Sequential(
                *Conv_block(in_channels, out_channels, kernel_size=1, stride=stride,
                            bias=False, BN=True, activation=None)
            )
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        return x + residual


class Xception(BaseModule):
    def __init__(self, color_channel=3, act_fn=nn.LeakyReLU(0.3)):
        super(Xception, self).__init__()
        self.act_fn = act_fn
        self.entry_flow_1 = self.make_entry_flow_1(color_channel, 128)  # 1/4
        self.entry_flow_2 = self.make_entry_flow_2(128, 728)  # 1/16
        self.middle_flow = self.make_middle_flow(728, 728, repeat_blocks=16)
        self.exit_flow = self.make_exit_flow(728, 2048)
        self.x4_feature_channels = 128
        self.last_feature_channels = 2048

    def make_entry_flow_1(self, in_channel, out_channel):
        m = nn.Sequential(
            *Conv_block(in_channel, 32, 3, stride=2, padding=1,
                        bias=False, BN=True, activation=self.act_fn),
            *Conv_block(32, 64, 3, stride=1, padding=1,
                        bias=False, BN=True, activation=self.act_fn),
            ResidualBlock(64, out_channel, 3, stride=2, padding=1,
                          dilation=1, bias=False, BN=True, activation=self.act_fn),
        )
        return m

    def make_entry_flow_2(self, in_channel, out_channel):
        m = nn.Sequential(
            ResidualBlock(in_channel, 256, 3, stride=2, padding=1,
                          dilation=1, bias=False, BN=True, activation=self.act_fn),
            ResidualBlock(256, out_channel, 3, stride=2, padding=1,
                          dilation=1, bias=False, BN=True, activation=self.act_fn)
        )
        return m

    def make_middle_flow(self, in_channel=728, out_channel=728, repeat_blocks=16):
        m = []
        for i in range(repeat_blocks):
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=1,
                                   dilation=1, bias=False, BN=True, activation=self.act_fn))
        return nn.Sequential(*m)

    def make_exit_flow(self, in_channel=728, out_channel=2048):
        m = nn.Sequential(
            ResidualBlock(in_channel, 1024, 3, stride=1, padding=2, dilation=2, bias=False,
                          BN=True, activation=self.act_fn, expand_channel_first=False),
            DSConvBlock(1024, 1536, 3, 1, 2, dilation=2, bias=False,
                        BN=True, activation=self.act_fn),
            DSConvBlock(1536, 1536, 3, 1, 2, dilation=2, bias=False,
                        BN=True, activation=self.act_fn),
            DSConvBlock(1536, out_channel, 3, 1, 2, dilation=2, bias=False,
                        BN=True, activation=self.act_fn),

        )
        return m

    def forward(self, x):
        x = self.entry_flow_1(x)
        x4_features = x
        x = self.entry_flow_2(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, x4_features


class XceptionClassifier(BaseModule):
    def __init__(self, num_class):
        super(XceptionClassifier, self).__init__()
        self.num_class = num_class
        self.act_fn = nn.LeakyReLU(0.3)
        self.encoder = Xception(3, self.act_fn)
        self.feature_conv = nn.Sequential(
            *Conv_block(self.encoder.last_feature_channels, num_class, 3, stride=1, padding=1, bias=False,
                        BN=True, activation=self.act_fn)
        )
        self.cnn_lstm_classifier = CNNLSTMClassifier(num_class=num_class, lstm_hidden=256, batch_first=True)

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.feature_conv(x)
        category_scores, transform_box = self.cnn_lstm_classifier(x)
        return category_scores, transform_box

    def predict(self, category_scores):
        scores, index = category_scores.max(1)
        return scores
