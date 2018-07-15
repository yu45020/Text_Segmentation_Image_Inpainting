import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _assert_no_grad

from models.BaseModels import BaseModule

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class MultiClassFocalLoss(nn.Module):
    # focal loss
    # https://arxiv.org/pdf/1708.02002.pdf
    # copy from  https://github.com/clcarwin/focal_loss_pytorch

    # alpha=0.75 gives the best for this project
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])

        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryFocalLoss(nn.BCEWithLogitsLoss):
    # gamma 0 gives the best AP scores
    def __init__(self, gamma=0, background_weights=1, words_weights=2):
        super(BinaryFocalLoss, self).__init__(size_average=True, reduce=False)
        self.gamma = gamma
        self.background_weights = background_weights
        self.words_weights = words_weights

    def forward(self, input, target):
        input = self.flatten_images(input)
        target = self.flatten_images(target)
        weights = torch.where(target > 0, torch.ones_like(target) * self.words_weights,  # words are 1
                              torch.ones_like(target) * self.background_weights)
        pt = F.logsigmoid(-input * (target * 2 - 1))
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weights, size_average=True, reduce=False)

        loss = (pt * self.gamma).exp() * loss
        return loss.mean()

    @staticmethod
    def flatten_images(x):
        assert x.dim() == 4 and x.size(1) == 1
        x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
        x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
        x = x.contiguous().view(-1, x.size(2))
        return x


class SoftBootstrapCrossEntropy(nn.BCELoss):
    """
    TRAINING DEEP NEURAL NETWORKS ON NOISY LABELS WITH BOOTSTRAPPING (https://arxiv.org/pdf/1412.6596.pdf)
    # Tensorflow: https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/object_detection/core/losses.py#L275-L336
    ++++   Use with caution ++++
    with this loss, the model may learn to detect words that are not labeled.
    but  not all words are necessary whited out
    """

    def __init__(self, beta=0.95, background_weight=1, words_weight=2,
                 size_average=True, reduce=True):
        super(SoftBootstrapCrossEntropy, self).__init__(size_average=size_average, reduce=reduce)
        self.beta = beta
        self.background_weight = background_weight
        self.words_weight = words_weight
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        # if a pixel's probability > 0.5, then assume it is true since labels might be noisy
        input = self.flatten_images(input)
        target = self.flatten_images(target)
        weights = torch.where(target > 0, torch.ones_like(target) * self.words_weight,  # words are 1
                              torch.ones_like(target) * self.background_weight)

        bootstrap_target = self.beta * target + (1 - self.beta) * (F.sigmoid(input) > 0.5).float()
        return F.binary_cross_entropy_with_logits(input, bootstrap_target, weight=weights,
                                                  size_average=self.size_average, reduce=self.reduce)

    @staticmethod
    def flatten_images(x):
        assert x.dim() == 4 and x.size(1) == 1
        x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
        x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
        x = x.contiguous().view(-1, x.size(2))
        return x


# +++++++++++++++++++++++++++++++++++++
#           Loss for classification with LSTM
# -------------------------------------

# reference: Multi-label Image Recognition by Recurrently Discovering Attentional Regions
# by Wang, chen,  Li, Xu, and Lin

class BCERegionLoss(nn.Module):
    def __init__(self):
        super(BCERegionLoss, self).__init__()
        self.anchor_box = FloatTensor([(0.4, 0.4), (0.4, -0.4), (-0.4, -0.4), (-0.4, 0.4)]).unsqueeze(-1)
        self.scale_alpha = FloatTensor([1])
        self.positive_beta = FloatTensor([0.2])
        self.bce = nn.BCEWithLogitsLoss()

    def scale_loss(self, scale):
        #         assert scale.size(1) == scale.size(2)
        sx = scale[:, 0, 0]
        ls = torch.pow(F.relu(torch.abs(sx) - self.scale_alpha), 2)
        sy = scale[:, 1, 1]
        ly = torch.pow(F.relu(torch.abs(sy) - self.scale_alpha), 2)
        positive_loss = F.relu(self.positive_beta - sx) + F.relu(self.positive_beta - sy)

        loss = 0.1 * positive_loss + ls + ly
        return loss.sum().view(1)

    def anchor_loss(self, attention_region):
        # input: num_class, 2 (anchor x, y) , 1   -self.anchor_box
        distance = 0.5 * torch.pow(attention_region - self.anchor_box, 2).sum(1)
        return 0.01 * distance.sum().view(1)

    def forward(self, input, target):
        category, transform_box = input
        #         scores, index = category.max(1)
        #         bce_loss = self.bce(scores, target)
        # all regions' predictions are checked
        bce_loss = FloatTensor([0])
        for i in range(category.size(1)):
            bce_loss = bce_loss + self.bce(category[:, i, :], target)
        bce_loss = bce_loss / category.size(1)

        regions = transform_box[:, 1:, :, 2:]
        region_loss = torch.cat([self.anchor_loss(i) for i in regions]).mean()

        scales = transform_box[:, :, :, :2]
        scale_loss = torch.cat([self.scale_loss(i) for i in scales]).mean()

        # spatial transform theta matrix (batch, 2, 3)
        # sum over the second axis so that transformed regions will not be 0 padded
        boundary = torch.abs(transform_box).sum(-1)
        boundary = torch.pow(F.relu(boundary - 1), 2)
        boundary_loss = 0.5 * boundary.view(boundary.size(0), -1).sum(-1).mean()
        return bce_loss, bce_loss + 0.01 * region_loss + 0.05 * scale_loss + 0.5 * boundary_loss


# +++++++++++++++++++++++++++++++++++++
#           Loss for inpainting
# -------------------------------------
# will be changed
class InpaintingLoss(nn.Module):
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
    # Image Inpainting for Irregular Holes Using Partial Convolutions
    def __init__(self, feature_encoder, feature_range=3):
        super(InpaintingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.feature_encoder = FeatureExtractor(feature_encoder, feature_range)

    def forward(self, raw_input, mask, output, origin):
        comp_img = raw_input * mask + output * (1 - mask)
        # extra features
        feature_comp = self.feature_encoder(comp_img)
        feature_output = self.feature_encoder(output)
        feature_origin = self.feature_encoder(origin)

        loss_validate = self.l1(mask * output, mask * origin)
        loss_hole = self.l1((1 - mask) * output, (1 - mask) * origin)
        loss_total_var = total_variation_loss(comp_img)  # total variation (smoothing penalty)

        # perceptual loss
        loss_perceptual_1 = sum(map(lambda x, y: self.l1(x, y), feature_comp, feature_origin))
        loss_perceptual_2 = sum(map(lambda x, y: self.l1(x, y), feature_output, feature_origin))
        loss_perceptual = loss_perceptual_1 + loss_perceptual_2

        # style loss
        loss_style_1 = sum(map(lambda x, y: self.l1(gram_matrix(x), gram_matrix(y)),
                               feature_output, feature_origin))
        loss_style_2 = sum(map(lambda x, y: self.l1(gram_matrix(x), gram_matrix(y)),
                               feature_comp, feature_origin))
        loss_style = loss_style_1 + loss_style_2

        # weights are recommended in the paper P7
        loss = 1.0 * loss_validate + 6.0 * loss_hole + \
               0.1 * loss_total_var + 0.05 * loss_perceptual + 120 * loss_style
        return loss


class FeatureExtractor(nn.Module):
    def __init__(self, encoder, feature_range=3):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(*[encoder.features[i] for i in range(feature_range)])
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = []
        for layer in self.layers:
            x = layer(x)
            out.append(x)
        return out


class Vgg19Extractor(BaseModule):
    def __init__(self, pretrained=True):
        super(Vgg19Extractor, self).__init__()
        vgg19 = torchvision.models.vgg16(pretrained=pretrained)
        feature1 = nn.Sequential(*vgg19.features[:5])
        feature2 = nn.Sequential(*vgg19.features[5:10])
        feature3 = nn.Sequential(*vgg19.features[10:17])
        self.features = nn.Sequential(*[feature1, feature2, feature3])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, img):
        result = []
        for layer in self.features.children():
            img = layer(img)
            result.append(img)
        return result


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss
