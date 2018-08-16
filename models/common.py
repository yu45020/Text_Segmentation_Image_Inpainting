import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import affine_grid, grid_sample

from .BaseModels import BaseModule, Conv_block

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class SpatialChannelSqueezeExcitation(BaseModule):
    # https://arxiv.org/abs/1709.01507
    # https://arxiv.org/pdf/1803.02579v1.pdf
    def __init__(self, in_channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SpatialChannelSqueezeExcitation, self).__init__()
        linear_nodes = max(in_channel // reduction, 4)  # avoid only 1 node case
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excite = nn.Sequential(
            # check the paper for the number 16 in reduction. It is selected by experiment.
            nn.Linear(in_channel, linear_nodes),
            activation,
            nn.Linear(linear_nodes, in_channel),
            nn.Sigmoid()
        )
        self.spatial_excite = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # channel
        channel = self.avg_pool(x).view(b, c)
        # channel = F.avg_pool2d(x, kernel_size=(h,w)).view(b,c)
        cSE = self.channel_excite(channel).view(b, c, 1, 1)
        x_cSE = torch.mul(x, cSE)

        # spatial
        sSE = self.spatial_excite(x)
        x_sSE = torch.mul(x, sSE)
        return torch.add(x_cSE, x_sSE)


def add_SCSE_block(model_block, in_channel=None):
    if in_channel is None:
        # the first layer is assumed to be conv
        in_channel = model_block[0].out_channels
    model_block.add_module("SCSE", SpatialChannelSqueezeExcitation(in_channel))


class ASP(BaseModule):
    # Atrous Spatial Pyramid Pooling with Image Pooling
    # add Vortex pooling https://arxiv.org/pdf/1804.06242v1.pdf
    def __init__(self, in_channel=256, out_channel=256, act_fn=None):
        super(ASP, self).__init__()
        asp_rate = [3, 9, 27]
        self.asp = nn.Sequential(
            nn.Sequential(*Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=1,
                                      bias=False, BN=True, activation=act_fn)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[0], stride=1, padding=(asp_rate[0] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[0],
                                      dilation=asp_rate[0], bias=False, BN=True, activation=act_fn)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[1], stride=1, padding=(asp_rate[1] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[1],
                                      dilation=asp_rate[1], bias=False, BN=True, activation=act_fn)),
            nn.Sequential(nn.AvgPool2d(kernel_size=asp_rate[2], stride=1, padding=(asp_rate[2] - 1) // 2),
                          *Conv_block(in_channel, out_channel, kernel_size=3, stride=1, padding=asp_rate[2],
                                      dilation=asp_rate[2], bias=False, BN=True, activation=act_fn))
        )

        """ To see why adding gobal average, please refer to 3.1 Global Context in https://www.cs.unc.edu/~wliu/papers/parsenet.pdf """

        self.img_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=False, BN=True, activation=act_fn))

        self.out_conv = nn.Sequential(*Conv_block(out_channel * 5, out_channel, kernel_size=1, bias=False,
                                                  BN=True, activation=act_fn))
        # self.initialize_weights()
        # self.selu_init_params()

    def forward(self, x):
        avg_pool = self.img_pooling(x)
        avg_pool = [F.interpolate(avg_pool, size=x.shape[2:], mode='bilinear', align_corners=False)]

        asp_pool = [layer(x) for layer in self.asp.children()]
        out_features = torch.cat(avg_pool + asp_pool, dim=1)
        out = self.out_conv(out_features)
        return out


class RFB(BaseModule):
    # receptive fiedl block  https://arxiv.org/abs/1711.07767 with some changes
    # reference: https://github.com/ansleliu/LightNet/blob/master/modules/rfblock.py
    # https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_mobile.py
    def __init__(self, in_channel, out_channel, activation, add_sece=False):
        super(RFB, self).__init__()
        asp_rate = [5, 17, 29]
        self.act_fn = activation
        # self.act_fn = activation
        self.input_down_channel = nn.Sequential(
            *Conv_block(in_channel, out_channel, kernel_size=1, bias=True, BN=True, activation=activation))

        rfb_linear_conv = [nn.Conv2d(out_channel * 4, out_channel, kernel_size=1, bias=True)]
        if add_sece:
            rfb_linear_conv.append(SpatialChannelSqueezeExcitation(in_channel=out_channel, activation=activation))
        self.rfb_linear_conv = nn.Sequential(*rfb_linear_conv)

        self.rfb = nn.Sequential(
            self.make_pooling_branch(in_channel, out_channel, out_channel, conv_kernel=1,
                                     astro_rate=1, activation=activation, half_conv=False),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=3,
                                     astro_rate=asp_rate[0], activation=activation, half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=5,
                                     astro_rate=asp_rate[1], activation=activation, half_conv=True),
            self.make_pooling_branch(in_channel, out_channel // 2, out_channel, conv_kernel=7,
                                     astro_rate=asp_rate[2], activation=activation, half_conv=True)
        )

    @staticmethod
    def make_pooling_branch(in_channel, mid_channel, out_channel, conv_kernel, astro_rate, activation, half_conv=False):
        # from the paper: we use a 1 x n plus an nx1 conv-layer to take place of the original nxn convlayer
        # similar to EffNet style
        if half_conv:
            m = nn.Sequential(
                *Conv_block(in_channel, mid_channel, kernel_size=1, padding=0,
                            bias=False, BN=True, activation=activation),
                *Conv_block(mid_channel, 3 * mid_channel // 2, kernel_size=(1, conv_kernel),
                            padding=(0, (conv_kernel - 1) // 2), bias=False, BN=True, activation=None),
                *Conv_block(3 * mid_channel // 2, out_channel, kernel_size=(conv_kernel, 1),
                            padding=((conv_kernel - 1) // 2, 0), bias=False, BN=True, activation=None),
                *Conv_block(out_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=False, BN=True, activation=activation, groups=out_channel))
        else:
            m = nn.Sequential(
                *Conv_block(in_channel, out_channel, kernel_size=conv_kernel, padding=(conv_kernel - 1) // 2,
                            bias=False, BN=True, activation=activation),
                *Conv_block(out_channel, out_channel, kernel_size=3, dilation=astro_rate, padding=astro_rate,
                            bias=False, BN=True, activation=activation, groups=out_channel)
            )

        return m

    def forward(self, x):
        # feature pooling
        rfb_pool = [layer(x) for layer in self.rfb.children()]
        rfb_pool = torch.cat(rfb_pool, dim=1)
        rfb_pool = self.rfb_linear_conv(rfb_pool)

        # skip connection
        resi = self.input_down_channel(x)
        return self.act_fn(rfb_pool + resi)


class CNNLSTMClassifier(BaseModule):
    def __init__(self, num_class, lstm_hidden=256, batch_first=True, act_fn=nn.LeakyReLU(0.3)):
        super(CNNLSTMClassifier, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(num_class, lstm_hidden, num_layers=1, batch_first=batch_first)
        self.lstm_linear_z = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden // 4), act_fn)
        self.lstm_linear_score = nn.Linear(lstm_hidden, num_class)
        self.st_theta_linear = nn.Sequential(nn.Linear(lstm_hidden // 4, 2 * 3))
        self.anchor_box = FloatTensor([(0, 0), (0.4, 0.4), (0.4, -0.4), (-0.4, -0.4), (-0.4, 0.4)
                                       ])

    @staticmethod
    def spatial_transformer(input_image, theta):
        # reference: Spatial Transformer Networks https://arxiv.org/abs/1506.02025
        # https://blog.csdn.net/qq_39422642/article/details/78870629
        grids = affine_grid(theta, input_image.size())

        output_img = grid_sample(input_image, grids)
        return output_img

    def forward(self, input_img):
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
            st_theta[:, 0, 1] = 0 * st_theta[:, 0, 1].clond()

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
