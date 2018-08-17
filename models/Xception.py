import torch
from memory_profiler import profile
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
                        dilation, bias, BN, activation, activation),
            DSConvBlock(middle_channel, out_channels, kernel_size, 1, padding,
                        dilation, bias, BN, activation, activation),
            DSConvBlock(out_channels, out_channels, kernel_size, stride, padding,
                        dilation, bias, BN, activation, None)
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
        self.entry_flow_2 = self.make_entry_flow_2(128, 512)  # 1/16
        self.middle_flow = self.make_middle_flow(512, 512, repeat_blocks=8, rate=(2, 4))
        self.exit_flow = self.make_exit_flow(512, 512, rate=(2, 1))
        self.x4_feature_channels = 128
        self.last_feature_channels = 512

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
            ResidualBlock(256, out_channel, 3, stride=1, padding=2,  # need to change  if want out-stride of 8
                          dilation=2, bias=False, BN=True, activation=self.act_fn)
        )
        return m

    def make_middle_flow(self, in_channel=728, out_channel=728, repeat_blocks=16, rate=(2, 4)):

        m = []
        # for i in range(repeat_blocks):
        #     m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=rate,
        #                            dilation=rate, bias=False, BN=True, activation=self.act_fn))

        #  Effective Use of Dilated Convolutions for Segmenting Small Object Instances in Remote Sensing Imagery
        # by Ryuhei Hamaguchi & Aito Fujita & Keisuke Nemoto & Tomoyuki Imaizumi & Shuhei Hikosaka
        for i in range(repeat_blocks // 2):
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=rate[0],
                                   dilation=rate[0], bias=False, BN=True, activation=self.act_fn))
        for i in range(repeat_blocks // 2):
            m.append(ResidualBlock(in_channel, out_channel, 3, stride=1, padding=rate[1],
                                   dilation=rate[1], bias=False, BN=True, activation=self.act_fn))
        return nn.Sequential(*m)

    def make_exit_flow(self, in_channel=728, out_channel=2048, rate=(2, 1)):
        m = nn.Sequential(
            ResidualBlock(in_channel, 512, 3, stride=1, padding=rate[0], dilation=rate[0], bias=False,
                          BN=True, activation=self.act_fn, expand_channel_first=False),
            ResidualBlock(512, out_channel, 3, stride=1, padding=rate[0], dilation=rate[0], bias=False,
                          BN=True, activation=self.act_fn, expand_channel_first=False),
            ResidualBlock(512, out_channel, 3, stride=1, padding=rate[1], dilation=rate[1], bias=False,
                          BN=True, activation=self.act_fn, expand_channel_first=False),
            ResidualBlock(512, out_channel, 3, stride=1, padding=rate[1], dilation=rate[1], bias=False,
                          BN=True, activation=self.act_fn, expand_channel_first=False),
        )
        return m

    @profile
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
            *Conv_block(self.encoder.last_feature_channels, num_class, 1, stride=1, padding=0, bias=False,
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


from torch.nn.functional import affine_grid, grid_sample


class XceptionClassifierV2(BaseModule):
    def __init__(self, num_class):
        super(XceptionClassifierV2, self).__init__()
        self.num_class = num_class
        self.act_fn = nn.LeakyReLU(0.3, inplace=True)  # nn.SELU(inplace=True)
        self.encoder = Xception(3, self.act_fn)
        self.feature_conv = nn.Sequential(
            *Conv_block(self.encoder.last_feature_channels, num_class, 1, stride=1, padding=0, bias=False,
                        BN=True, activation=self.act_fn)
        )
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        lstm_hidden = 256
        self.lstm = nn.LSTM(num_class, lstm_hidden, num_layers=1, batch_first=True)
        self.lstm_linear_z = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 4), self.act_fn)
        self.lstm_linear_score = nn.Linear(lstm_hidden, num_class)
        self.st_theta_linear = nn.Sequential(nn.Linear(lstm_hidden // 4, 2 * 3))
        self.anchor_box = FloatTensor([(0, 0), (0.3, 0.3), (0.3, -0.3), (-0.3, -0.3), (-0.3, 0.3)
                                       ])

    def cnn_lstm_classifier(self, input_img):
        # Multi-label Image Recognition by Recurrently Discovering Attentional Regions by Wang, chen,  Li, Xu, and Lin
        # LSTM input: step size is one, feature size is num_class (channels)

        img = input_img
        batch = input_img.size(0)

        category_scores = []
        transform_box = []
        # h = c = torch.zeros(1, batch, self.lstm.hidden_size).cuda()
        features = self.global_avg(img).view(batch, 1, -1)
        y, (h, c) = self.lstm(features)
        #         s = self.lstm_linear_score(y.view(batch, -1))
        #         category_scores.append(s)
        for i in range(4 + 1):  # 4 anchor points and repeated 4 times
            z = self.lstm_linear_z(h.transpose(0, 1).view(batch, -1))  # y.view(batch, -1)
            st_theta = self.st_theta_linear(z).view(batch, 2, 3)
            st_theta[:, :, -1] = st_theta[:, :, -1].clone() + self.anchor_box[i]

            st_theta[:, 1, 0] = 0 * st_theta[:, 1, 0].clone()
            st_theta[:, 0, 1] = 0 * st_theta[:, 0, 1].clone()

            transform_box.append(st_theta)

            img = self.spatial_transformer(input_img, st_theta)
            features = self.global_avg(img).view(batch, 1, -1)

            # y.size = batch, seq_len (1) , num_direc*hidden_size
            # h, c size = num_layer*bi-direc, batch, hidden_size
            y, (h, c) = self.lstm(features, (h, c))

            s = self.lstm_linear_score(
                y.view(batch, -1))  # the paper use the hidden state to get scores  h.transpose(0, 1).view(batch, -1)
            category_scores.append(s)

        category_scores = torch.stack(category_scores, dim=1)  # size: batch, category regions, category
        transform_box = torch.stack(transform_box, dim=1)  # the first one is free. size: batch, regions, 2,3
        return category_scores, transform_box

    @staticmethod
    def spatial_transformer(input_image, theta):
        # reference: Spatial Transformer Networks https://arxiv.org/abs/1506.02025
        # https://blog.csdn.net/qq_39422642/article/details/78870629
        grids = affine_grid(theta, input_image.size())

        output_img = grid_sample(input_image, grids)
        return output_img

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.feature_conv(x)
        category_scores, transform_box = self.cnn_lstm_classifier(x)
        return category_scores, transform_box
