import glob
import os
import random
import re
from itertools import chain

import torch
from PIL import Image, ImageChops
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ToTensor, RandomResizedCrop, Compose, Normalize, transforms
from torchvision.transforms.functional import resized_crop, to_tensor

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
"""
Filter difference in brightness, to some degree. 
If this value is set very high, say 0.8, some words are filtered out;
if too small, say <0.1, the mask may have noisy white points, and the model will fail to converge. 
VERY IMPORTANT: Generate masks before dumping data into the model. Noisy data or almost black masks hurt performances.
"""
brightness_difference = 0.4  # in [0,1]


class TextSegmentationData(Dataset):
    def __init__(self, img_folder, max_img=False, img_size=(512, 512)):
        # get raw images
        self.images = glob.glob(os.path.join(img_folder, "raw/*"))
        assert len(self.images) > 0
        self.max_img = max_img
        # if len(self.images) > max_img and max_img:
        #     self.images = random.choices(self.images, k=max_img)
        self.images = random.choices(self.images, k=max_img)
        print("Find {} images. ".format(len(self.images)))

        self.img_size = img_size
        # image augment
        self.transformer = Compose([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        # avoid multiprocessing on the same image
        img_raw = Image.open(img_file).convert('RGB')
        img_clean = Image.open(re.sub("raw", 'clean', img_file)).convert("RGB")
        img_raw, img_mask = self.process_images(img_raw, img_clean)
        return img_raw, img_mask

    def process_images(self, raw, clean):
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.1, 2.0), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size)
        clean_img = resized_crop(clean, i, j, h, w, self.img_size)

        # get mask before further image augment
        mask_tensor_long = self.get_mask(raw_img, clean_img)
        raw_img = self.transformer(raw_img)
        return raw_img, mask_tensor_long

    @staticmethod
    def get_mask(raw_pil, clean_pil):
        # use PIL ! It will take care the difference in brightness/contract
        mask = ImageChops.difference(raw_pil, clean_pil)
        mask = to_tensor(mask)
        mask = mask[:1, :, :] > brightness_difference  # single channel
        return mask.long()


class EvaluateSet(Dataset):
    def __init__(self, img_folder=None):
        self.eval_imgs = [glob.glob(img_folder + "**/*.{}".format(i), recursive=True) for i in ['jpg', 'jpeg', 'png']]
        self.eval_imgs = list(chain.from_iterable(self.eval_imgs))
        self.transformer = Compose([ToTensor(),
                                    transforms.Lambda(lambda x: x.unsqueeze(0))
                                    ])
        self.normalizer = Compose([transforms.Lambda(lambda x: x.squeeze(0)),
                                   Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                   transforms.Lambda(lambda x: x.unsqueeze(0))
                                   ])
        print("Find {} test images. ".format(len(self.eval_imgs)))

    def __len__(self):
        return len(self.eval_imgs)

    def __getitem__(self, item):
        img = Image.open(self.eval_imgs[item]).convert("RGB")
        return self.resize_pad_tensor(img), self.eval_imgs[item]

    def resize_pad_tensor(self, pil_img):
        origin = self.transformer(pil_img)
        fix_len = 512
        long = max(pil_img.size)
        ratio = fix_len / long
        new_size = tuple(map(lambda x: int(x * ratio), pil_img.size))
        img = pil_img.resize(new_size, Image.LANCZOS)
        # img = pil_img
        img = self.transformer(img)

        _, _, h, w = img.size()
        if fix_len > w:

            boarder_pad = (0, fix_len - w, 0, 0)
        else:

            boarder_pad = (0, 0, 0, fix_len - h)

        img = pad(img, boarder_pad, value=0)
        mask_resizer = self.resize_mask(boarder_pad, pil_img.size)
        return self.normalizer(img), origin, mask_resizer

    @staticmethod
    def resize_mask(padded_values, origin_size):
        unpad = tuple(map(lambda x: -x, padded_values))
        upsampler = nn.Upsample(size=tuple(reversed(origin_size)), mode='bilinear', align_corners=False)
        m = Compose([
            torch.nn.ZeroPad2d(unpad),
            transforms.Lambda(lambda x: upsampler(x.float())),
            transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1) > 0)
        ])
        return m
