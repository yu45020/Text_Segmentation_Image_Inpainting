import torch
from torch import optim

from dataloader import ImageSet, ImageLoader
from loss import FocalLoss
from models.text_segmentation import TextSegament

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
# criteria = nn.CrossEntropyLoss(class_weight)


criteria = FocalLoss()
# loss = nn.CrossEntropyLoss()
model = TextSegament(encoder_checkpoint='checkpoints/mobilenetv2_718.pth')
trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(trainable_params)
x, y = next(iter(img_loader))
out = model.forward(x)
loss = criteria(out, y)
out.size()
