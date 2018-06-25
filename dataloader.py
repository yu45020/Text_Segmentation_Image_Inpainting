import glob
import os
import random
import re
from itertools import chain
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import torch
from PIL import Image
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter, ToTensor, RandomResizedCrop, Compose, Normalize, transforms
from torchvision.transforms.functional import resized_crop, to_tensor

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
"""
Filter difference in brightness, to some degree. 
If this value is set very high, say 0.8, some words are hard to detect;
if too small, say <0.1, the mask may have noisy white points, and the model will fail to converge. 
Recommend to generate masks to check the optimal value
"""
brightness_filter = 0.3  # in [0,1]


class ImageSet(Dataset):
    def __init__(self, img_folder, max_img=False):
        # get raw images
        self.images = glob.glob(os.path.join(img_folder, "raw/*"), recursive=True)
        assert len(self.images) > 0
        self.max_img = max_img
        print("Find {} images. ".format(len(self.images)))

    def __len__(self):
        return self.max_img

    def __getitem__(self, item):
        img_file = random.choice(self.images)
        # avoid multiprocessing on the same image
        img_raw_ = Image.open(img_file).convert('RGB')
        img_raw = Image.new('RGB', img_raw_.size)
        img_core = img_raw_.getdata()
        img_raw.putdata(img_core)

        img_clean_ = Image.open(re.sub("raw", 'clean', img_file)).convert("RGB")
        img_clean = Image.new("RGB", img_clean_.size)
        img_core_ = img_clean_.getdata()
        img_clean.putdata(img_core_)
        return img_raw, img_clean


class MaskLoader(DataLoader):
    # mask: 1 is masked word; 0 is background
    def __init__(self, dataset,
                 batch_size=2,
                 shuffle=True,
                 img_size=(512, 512),
                 num_workers=0
                 ):
        super(MaskLoader, self).__init__(dataset,
                                         batch_size,
                                         shuffle,
                                         num_workers=num_workers,
                                         collate_fn=self.batch_collector)

        assert all(x % 16 == 0 for x in img_size), 'Image size must be multiple of 16'
        self.img_size = img_size
        self.dataset = dataset
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

        # MobileNet V2 pre-train checkpoint needs to normalize image
        # https://github.com/tonylins/pytorch-mobilenet-v2/issues/9
        self.transformer = Compose([ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

    def batch_collector(self, batch):
        raw_clean = self.image_augment(batch)

        raw_imgs = [i[0] for i in raw_clean]
        raw_imgs = torch.stack(raw_imgs, dim=0).contiguous()

        mask_imgs = [i[1] for i in raw_clean]
        # masks are in single channel
        mask_imgs = torch.stack(mask_imgs, dim=0).contiguous()

        # clean_img_masks = torch.abs(clean_imgs - raw_imgs)  # usually use white to cover words
        # clean_img_masks = clean_img_masks[:, 0, :, :]
        # clean_img_masks = torch.where(clean_img_masks > brightness_filter,  # filter difference in brightness
        #                               torch.ones_like(clean_img_masks),
        #                               torch.zeros_like(clean_img_masks))
        # clean_img_masks = clean_img_masks.long()
        # if use_cuda:
        #     raw_imgs = raw_imgs.cuda()
        #     mask_imgs = mask_imgs.cuda(async=True)

        return raw_imgs, mask_imgs

    def image_augment(self, pil_imgs):
        with ThreadPool(cpu_count()) as p:
            out = p.map(self._image_augment, pil_imgs)
        return out

    def _image_augment(self, pil_img):
        raw, clean = pil_img
        # crop then resize
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.1, 2.0), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size)
        clean_img = resized_crop(clean, i, j, h, w, self.img_size)

        # get mask before further image augment
        mask_tensor = self.get_mask(raw_img, clean_img)

        # add color noise
        raw_img = self.color_jitter(raw_img)
        # to tensor
        raw_img = self.transformer(raw_img)
        return raw_img, mask_tensor

    @staticmethod
    def get_mask(raw_pil, clean_pil):
        raw_tensor = to_tensor(raw_pil)
        clean_tensor = to_tensor(clean_pil)
        mask = torch.abs(raw_tensor - clean_tensor)  # usually use white to cover words
        mask = mask[0, :, :]  # single channel
        mask = torch.where(mask > brightness_filter,  # filter difference in brightness
                           torch.ones_like(mask),
                           torch.zeros_like(mask))
        return mask.long()


class InpaintingData(MaskLoader):
    def __init__(self, dataset,
                 batch_size=2,
                 shuffle=True):
        super(MaskLoader, self).__init__(dataset,
                                         batch_size,
                                         shuffle,
                                         )

        self.dataset = dataset

    @staticmethod
    def add_random_mask(mask, length=5000):
        # random walk
        action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        b, c, h, w = mask.size()
        x = random.randint(h // 3, 4 * h // 5)
        y = random.randint(w // 3, 4 * w // 5)
        # x, y = random.randint(0, min(h, w)), random.randint(0, min(h, w))
        for i in range(length):
            step = random.choice(action_list)
            x = min(h - 1, x + step[0])
            y = min(w - 1, y + step[1])
            mask[:, :, x, y] = 1
        return mask

    def batch_collector(self, batch):
        raw_clean = self.process_images(batch)
        raw_imgs = [self.img_transform(i[0]) for i in raw_clean]
        raw_imgs = torch.stack(raw_imgs, dim=0).contiguous()

        clean_imgs = [self.img_transform(i[1]) for i in raw_clean]
        clean_imgs = torch.stack(clean_imgs, dim=0).contiguous()

        # usually use white to cover words, so words' pixels become 1 after subtraction, the rest are 0
        clean_img_masks = torch.abs(clean_imgs - raw_imgs)
        clean_img_masks = self.add_random_mask(clean_img_masks)  # add 1s in random position

        # masks are marked as 0, the rest are 1
        clean_img_masks = torch.where(clean_img_masks > brightness_filter,  # filter difference in brightness
                                      torch.zeros_like(clean_img_masks),
                                      torch.ones_like(clean_img_masks))

        clean_img_masks = clean_img_masks[:, :1, :, :].contiguous()
        stacked_imgs = [torch.cat([i, j]) for i, j in zip(raw_imgs, clean_img_masks)]
        stacked_imgs = torch.stack(stacked_imgs, dim=0)
        if use_cuda:
            stacked_imgs = stacked_imgs.cuda()
            clean_imgs = clean_imgs.cuda(async=True)
            # [img, mask],  mask,
        return stacked_imgs, stacked_imgs[:, 3:4, :, :], clean_imgs


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

#
# transforms.Lambda(lambda x: x.squeeze(1).float()),
# ToPILImage(),
# Resize(tuple(reversed(origin_size))),  # torchvision uses (h, w)
# ToTensor(),
