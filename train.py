from dataloader import ImageSet, ImageLoader
from models.TextSegamentation import TextSegament
from torch import nn, optim
import torch
from models.TextSegamentation import TextSegament
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

img_set = ImageSet('./images/',
                   max_img=10,
                   augment_per_img=2,
                   img_size=(320, 480))
img_loader = ImageLoader(img_set, batch_size=1, shuffle=True)

# class_weight = torch.FloatTensor([1.0, 6.0])
loss = nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
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


loss = FocalLoss()
# loss = nn.CrossEntropyLoss()
model = TextSegament(encoder_checkpoint='checkpoints/mobilenetv2_718.pth')
trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(trainable_params)
x, y = next(iter(img_loader))
out = model.forward(x)
loss(out, y)
out.size()
