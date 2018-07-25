import glob
import os
import random
import re
from itertools import chain

import numpy as np
import torch
from PIL import Image, ImageChops
from PIL import ImageDraw
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, ToTensor, RandomResizedCrop, Compose, Normalize, transforms, Grayscale, \
    RandomGrayscale
from torchvision.transforms.functional import resized_crop, to_tensor

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
"""
Filter difference in brightness, to some degree. 
If you have perfect pairs (only the text parts are removed), then set it to 0. 
If this value is set very high, say 0.8, some words are filtered out;
if too small, say <0.1, the mask may have noisy white points, and the model will fail to converge. 
VERY IMPORTANT: Generate masks before dumping data into the model. Noisy data or almost black masks hurt performances.
"""
brightness_difference = 0.4  # in [0,1]


class TextSegmentationData(Dataset):
    def __init__(self, image_folder, mean, std, max_images=False, image_size=(512, 512)):
        # get raw images

        self.images = glob.glob(os.path.join(image_folder, "raw/*"))
        assert len(self.images) > 0
        if max_images:
            self.images = random.choices(self.images, k=max_images)
        print("Find {} images. ".format(len(self.images)))
        self.grayscale = Grayscale(num_output_channels=1)
        self.img_size = image_size
        # image augment
        self.transformer = Compose([ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                                    ToTensor(),
                                    Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        # avoid multiprocessing on the same image
        img_raw = Image.open(img_file).convert('RGB')
        img_clean = Image.open(re.sub("raw", 'clean', img_file)).convert("RGB")
        img_raw, img_mask = self.process_images(img_raw, img_clean)
        # recommend to use nn.MaxPool2d(kernel_size=7, stride=1, padding=3) on the mask
        # so regions around the words can also be whited ou
        return img_raw, img_mask

    def process_images(self, raw, clean):
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.5, 2.0), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size, interpolation=Image.BICUBIC)
        clean_img = resized_crop(clean, i, j, h, w, self.img_size, interpolation=Image.BICUBIC)

        # get mask before further image augment
        mask_tensor = self.get_mask(raw_img, clean_img)

        raw_img = self.transformer(raw_img)
        return raw_img, mask_tensor

    def get_mask(self, raw_pil, clean_pil):
        # use PIL ! It will take care the difference in brightness/contract
        mask = ImageChops.difference(raw_pil, clean_pil)
        mask = self.grayscale(mask)  # single channel
        mask = to_tensor(mask)
        mask = mask > brightness_difference
        return mask.float()  # .long()


class ImageInpaintingData(Dataset):
    def __init__(self, image_folder, max_images=False, image_size=(512, 512), add_random_masks=False):
        super(ImageInpaintingData, self).__init__()

        if isinstance(image_folder, str):
            self.images = glob.glob(os.path.join(image_folder, "raw/*"))
        else:
            self.images = list(chain.from_iterable([glob.glob(os.path.join(i, "raw/*")) for i in image_folder]))
        assert len(self.images) > 0

        if max_images:
            self.images = random.choices(self.images, k=max_images)
        print(f"Find {len(self.images)} images.")

        self.img_size = image_size

        self.transformer = Compose([RandomGrayscale(p=0.4),
                                    ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                                    ToTensor(),
                                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.add_random_masks = add_random_masks
        self.random_mask = RandomMask(image_size[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        img_raw = Image.open(img_file).convert('RGB')
        img_clean = Image.open(re.sub("raw", 'clean', img_file)).convert("RGB")
        img_raw, masks, img_clean = self.process_images(img_raw, img_clean)
        return img_raw, masks, img_clean

    def process_images(self, raw, clean):
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.5, 2.0), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size, interpolation=Image.BICUBIC)
        if self.add_random_masks:
            raw_img = self.random_mask.draw(raw_img)

        clean_img = resized_crop(clean, i, j, h, w, self.img_size, interpolation=Image.BICUBIC)

        # get mask before further image augment
        mask = self.get_mask(raw_img, clean_img)
        mask_t = to_tensor(mask)
        mask_t = (mask_t > brightness_difference).float()

        mask_t = torch.max(mask_t, dim=0, keepdim=True)
        mask_t = torch.nn.functional.max_pool2d(mask_t, kernel_size=9, stride=1, padding=4)

        # corrupt the clean images rather than using the raw ones 
        binary_mask = (1 - mask_t)  # valid positions are 1; holes are 0
        binary_mask = binary_mask.expand(3, -1, -1)
        clean_img = self.transformer(clean_img)
        corrupted_img = clean_img * binary_mask
        return corrupted_img, binary_mask, clean_img

    @staticmethod
    def get_mask(raw_pil, clean_pil):
        mask = ImageChops.difference(raw_pil, clean_pil)
        # mask_array = np.array(mask)
        # mask_array = np.where(mask_array > 90, mask_array, np.zeros_like(mask_array))
        # mask = Image.fromarray(mask_array)
        return mask


class TestDataset(TextSegmentationData):
    def __init__(self, image_folder, max_images=False, image_size=(512, 512), random_crop=True):
        super(TestDataset, self).__init__(image_folder, False, False,
                                          max_images, image_size)
        self.transformer = Compose([ToTensor(),
                                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.random_mask = RandomMask(image_size[0])
        self.random_crop = random_crop
        self.images = self.gen_img(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images

    def gen_img(self, item):
        img_file = self.images[item]
        # avoid multiprocessing on the same image
        img_raw = Image.open(img_file).convert('RGB')
        img_raw = self.random_mask.draw(img_raw)

        img_clean = Image.open(re.sub("raw", 'clean', img_file)).convert("RGB")
        img_raw, masks, img_clean = self.process_images(img_raw, img_clean)
        return img_raw, masks, img_clean

    def process_images(self, raw, clean):
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.5, 2.0), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size, interpolation=Image.BICUBIC)
        clean_img = resized_crop(clean, i, j, h, w, self.img_size, interpolation=Image.BICUBIC)

        # get mask before further image augment
        mask = self.get_mask(raw_img, clean_img)
        mask_t = to_tensor(mask)
        mask_t = (mask_t > 0).float()
        mask_t = torch.nn.functional.max_pool2d(mask_t, kernel_size=5, stride=1, padding=2)
        # mask_t = mask_t.byte()

        raw_img = ImageChops.difference(mask, clean_img)
        return self.transformer(raw_img), 1 - mask_t, self.transformer(clean_img)

    def get_mask(self, raw_pil, clean_pil):
        mask = ImageChops.difference(raw_pil, clean_pil)
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 90, mask_array, np.zeros_like(mask_array))
        mask = Image.fromarray(mask_array)
        return mask


class DanbooruDataset(Dataset):
    def __init__(self, image_folder, name_tag_dict, mean, std,
                 image_size=512, max_images=False, num_class=1000):
        super(DanbooruDataset, self).__init__()
        assert image_size % 16 == 0

        self.images = glob.glob(os.path.join(image_folder, '*'))
        assert len(self.images) > 0
        if max_images:
            self.images = random.choices(self.images, k=max_images)
        print("Find {} images. ".format(len(self.images)))

        self.name_tag_dict = name_tag_dict
        self.img_transform = self.transformer(mean, std)
        # one hot encoding
        self.onehot = torch.eye(num_class)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        image = self.img_transform(Image.open(image_file).convert("RGB"))

        basename = os.path.basename(image_file).split('.')[0]
        tags = self.name_tag_dict[basename]
        target = self.onehot.index_select(0, torch.LongTensor(tags)).sum(0)  # (1, num_class)
        return image, LongTensor(target)

    @staticmethod
    def transformer(mean, std):
        m = Compose([RandomGrayscale(p=0.2),
                     # RandomHorizontalFlip(p=0.2), don't use them since label locations are not available
                     # RandomVerticalFlip(p=0.2),
                     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                     ToTensor(),
                     Normalize(mean, std)])
        return m


class EvaluateSet(Dataset):
    def __init__(self, mean, std, img_folder=None, resize=512):
        self.eval_imgs = [glob.glob(img_folder + "**/*.{}".format(i), recursive=True) for i in ['jpg', 'jpeg', 'png']]
        self.eval_imgs = list(chain.from_iterable(self.eval_imgs))
        assert resize % 8 == 0
        self.resize = resize
        self.transformer = Compose([ToTensor(),
                                    transforms.Lambda(lambda x: x.unsqueeze(0))
                                    ])
        self.normalizer = Compose([transforms.Lambda(lambda x: x.squeeze(0)),
                                   Normalize(mean=mean,
                                             std=std),
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
        fix_len = self.resize
        long = min(pil_img.size)
        ratio = fix_len / long
        new_size = tuple(map(lambda x: int(x * ratio) // 8 * 8, pil_img.size))
        img = pil_img.resize(new_size, Image.BICUBIC)
        # img = pil_img
        img = self.transformer(img)

        _, _, h, w = img.size()
        if w > fix_len:

            boarder_pad = (0, w - fix_len, 0, 0)
        else:

            boarder_pad = (0, 0, 0, h - fix_len)

        img = pad(img, boarder_pad, value=0)
        mask_resizer = self.resize_mask(boarder_pad, pil_img.size)
        return self.normalizer(img), origin, mask_resizer

    @staticmethod
    def resize_mask(padded_values, origin_size):
        # resize generated mask back to the input image size
        unpad = tuple(map(lambda x: -x, padded_values))
        upsampler = nn.Upsample(size=tuple(reversed(origin_size)), mode='bilinear', align_corners=False)
        m = Compose([
            torch.nn.ZeroPad2d(unpad),
            transforms.Lambda(lambda x: upsampler(x.float())),
            transforms.Lambda(lambda x: x.expand(-1, 3, -1, -1) > 0)
        ])
        return m


class RandomMask:
    def __init__(self, size, offset=10):
        self.size = size - offset
        self.offset = offset

    def draw(self, pil_img):
        draw = ImageDraw.Draw(pil_img)
        # draw liens
        for i in range(np.random.randint(1, 4)):
            cords = np.random.randint(self.offset, self.size, 4)
            width = np.random.randint(5, 15)
            draw.line(cords.tolist(), width=width, fill=255)
        # draw circles
        for i in range(np.random.randint(1, 4)):
            cords = np.random.randint(self.offset, self.size, 2)
            cords.sort()
            ex = np.random.randint(10, 50, 2)
            draw.ellipse(np.concatenate([cords, cords + ex]).tolist(), fill=255)
        return pil_img

