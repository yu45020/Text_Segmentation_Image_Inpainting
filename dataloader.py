import glob
import random
import re
from itertools import chain
from multiprocessing.dummy import Pool as ThreadPool

import torch

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import resized_crop, to_tensor


class ImageSet(Dataset):
    def __init__(self, img_folder,
                 max_img=100,
                 augment_per_img=2,
                 img_size=(320, 480)):
        assert all(x % 16 == 0 for x in img_size), 'Image size must be multiple of 16'
        self.img_size = img_size
        self.augment_per_img = augment_per_img
        self.images = self.process_image(img_folder, max_img)
        print("Find {} images. ".format(self.__len__()))

    def process_image(self, img_folder, max_img):
        raw_files = glob.glob(img_folder + "raw/*", recursive=True)
        if len(raw_files) > max_img:
            raw_files = random.choices(raw_files, k=max_img)
        assert len(raw_files) > 0
        pool = ThreadPool(4)
        patch_grids = pool.map(self._process_image, raw_files)
        pool.close()
        pool.join()

        return list(chain.from_iterable(patch_grids))

    def _process_image(self, img_file):
        # img_raw = Image.open(img_file).convert("RGB")
        img_raw = Image.open(img_file).convert("YCbCr")
        img_clean = Image.open(re.sub("raw", 'clean', img_file))
        imgs = [self.image_augment((img_raw, img_clean)) for _ in range(self.augment_per_img)]
        return imgs

    def image_augment(self, pil_imgs):
        raw, clean = pil_imgs
        i, j, h, w = transforms.RandomResizedCrop.get_params(raw, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.))
        raw = resized_crop(raw, i, j, h, w, size=self.img_size)
        clean = resized_crop(clean, i, j, h, w, self.img_size)
        return raw, clean

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # (raw, clean)
        return self.images[item]


class ImageLoader(DataLoader):
    def __init__(self, dataset,
                 batch_size=1,
                 shuffle=True):
        self.dataset = dataset
        super(ImageLoader, self).__init__(dataset,
                                          batch_size,
                                          shuffle,
                                          collate_fn=self.batch_collector)
        # self.img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # https://github.com/tonylins/pytorch-mobilenet-v2/issues/9
        # mobilenet v2 is pre-trained on ImageNet
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224,0.225]),

        # ])

    @staticmethod
    def img_transform(pil_img):
        channels = pil_img.split()
        img = to_tensor(channels[0])  # either gray or Y from YCbCr
        img = img.expand(3, img.size(1), img.size(2))
        return img

    def batch_collector(self, batch):
        raw_clean = batch
        raw_imgs = [self.img_transform(i[0]) for i in raw_clean]
        raw_imgs = torch.stack(raw_imgs, dim=0).contiguous()

        clean_imgs = [self.img_transform(i[1]) for i in raw_clean]
        clean_imgs = torch.stack(clean_imgs, dim=0).contiguous()
        clean_img_masks = torch.abs(clean_imgs - raw_imgs)  # usually use white to cover words
        clean_img_masks = clean_img_masks[:, 0, :, :]
        clean_img_masks = torch.where(clean_img_masks > 0.,
                                      torch.ones_like(clean_img_masks),
                                      torch.zeros_like(clean_img_masks))
        clean_img_masks = clean_img_masks.long()
        if use_cuda:
            raw_imgs = raw_imgs.cuda()
            clean_img_masks = clean_img_masks.cuda(async=True)

        return raw_imgs, clean_img_masks


class InpaintingData(ImageLoader):
    def __init__(self, dataset,
                 batch_size=1,
                 shuffle=True):
        self.dataset = dataset
        super(ImageLoader, self).__init__(dataset,
                                          batch_size,
                                          shuffle,
                                          collate_fn=self.batch_collector)

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
        raw_clean = batch
        raw_imgs = [self.img_transform(i[0]) for i in raw_clean]
        raw_imgs = torch.stack(raw_imgs, dim=0).contiguous()

        clean_imgs = [self.img_transform(i[1]) for i in raw_clean]
        clean_imgs = torch.stack(clean_imgs, dim=0).contiguous()

        # usually use white to cover words, so words' pixels become 1 after subtraction, the rest are 0
        clean_img_masks = torch.abs(clean_imgs - raw_imgs)
        clean_img_masks = self.add_random_mask(clean_img_masks)  # add 1s in random position

        # masks are marked as 0, the rest are 1
        clean_img_masks = torch.where(clean_img_masks > 0.2,
                                      torch.zeros_like(clean_img_masks),
                                      torch.ones_like(clean_img_masks))

        idx = clean_img_masks.long()
        clean_img_masks[0][0]
        idx[0][0][0]
        if use_cuda:
            raw_imgs = raw_imgs.cuda()
            clean_img_masks = clean_img_masks.cuda(async=True)
            clean_imgs = clean_imgs.cuda(async=True)
        return raw_imgs, clean_img_masks, clean_imgs
