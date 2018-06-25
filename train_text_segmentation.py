import torch
from torch import optim

from dataloader import ImageSet, MaskLoader
from loss import FocalLoss
from models.text_segmentation import TextSegament

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

img_set = ImageSet('./images/',
                   max_img=10, target_folder='clean',
                   img_size=(320, 480))

img_loader = MaskLoader(img_set, batch_size=2, shuffle=True)

# class_weight = torch.FloatTensor([1.0, 6.0])
# criteria = nn.CrossEntropyLoss(class_weight)


criteria = FocalLoss(alpha=0.75)
# loss = nn.CrossEntropyLoss()
# model = TextSegament(encoder_checkpoint='checkpoints/mobilenetv2_718.pth', free_last_blocks=None)
model = TextSegament(encoder_checkpoint=None, free_last_blocks=None)
model.total_parameters()

trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(trainable_params)
if use_cuda:
    model = model.cuda()
    criteria = criteria.cuda()

x, y = next(iter(img_loader))
out = model.forward(x)
loss = criteria(out, y)
out.size()

torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
model.gra
