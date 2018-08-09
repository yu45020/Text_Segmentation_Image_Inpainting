import glob
import os
import random
import re
from itertools import chain

import cv2
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


def draw_contour(img, mask):
    a, b, c = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in b:
        approx = cv2.approxPolyDP(cnt, 0, True)
        cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
    return img


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
        # masks are  pre-generated with cv2.dilate of 5 on images
        img_mask = Image.open(re.sub("raw", 'mask', img_file)).convert("L")
        img_raw, img_mask = self.process_images(img_raw, img_mask)
        return img_raw, img_mask

    def process_images(self, raw, clean):
        i, j, h, w = RandomResizedCrop.get_params(raw, scale=(0.5, 2), ratio=(3. / 4., 4. / 3.))
        raw_img = resized_crop(raw, i, j, h, w, size=self.img_size, interpolation=Image.BICUBIC)
        mask_img = resized_crop(clean, i, j, h, w, self.img_size, interpolation=Image.BICUBIC)

        # get mask before further image augment
        mask_tensor = self.get_mask(raw_img, mask_img)
        raw_img = self.transformer(raw_img)
        return raw_img, mask_tensor

    def get_mask(self, raw_pil, mask_pil):
        # use PIL ! It will take care the difference in brightness/contract
        # raw = raw_pil.convert("L")
        # clean = clean_pil.convert("L")
        # mask = ImageChops.difference(raw, clean)
        mask = np.array(mask_pil)
        mask = np.where(mask > brightness_difference * 255, np.uint8(255), np.uint8(0))
        # kernel size should not be too large
        # find tune it s.t. all words are just blurred
        mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=1)
        # mask = draw_contour(mask, mask)
        mask = np.expand_dims(mask, -1)
        mask = to_tensor(mask)
        # mask = mask > brightness_difference
        return mask  # .float()  # .long()


class ImageInpaintingData(Dataset):
    def __init__(self, image_folder, max_images=False, image_size=(512, 512), add_random_masks=False):
        super(ImageInpaintingData, self).__init__()

        if isinstance(image_folder, str):
            self.images = glob.glob(os.path.join(image_folder, "clean/*"))
        else:
            self.images = list(chain.from_iterable([glob.glob(os.path.join(i, "clean/*")) for i in image_folder]))
        assert len(self.images) > 0

        if max_images:
            self.images = random.choices(self.images, k=max_images)
        print(f"Find {len(self.images)} images.")

        self.img_size = image_size

        self.transformer = Compose([RandomGrayscale(p=0.4),
                                    # ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                                    ToTensor(),
                                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.add_random_masks = add_random_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_file = self.images[item]
        img_clean = Image.open(img_file).convert('RGB')
        img_mask = Image.open(re.sub("clean", 'mask', img_file)).convert("L")
        img_raw, masks, img_clean = self.process_images(img_clean, img_mask)
        return img_raw, masks, img_clean

    def process_images(self, clean, mask):
        i, j, h, w = RandomResizedCrop.get_params(clean, scale=(0.5, 2.0), ratio=(3. / 4., 4. / 3.))
        clean_img = resized_crop(clean, i, j, h, w, size=self.img_size, interpolation=Image.BICUBIC)
        mask = resized_crop(mask, i, j, h, w, self.img_size, interpolation=Image.BICUBIC)

        # get mask before further image augment
        # mask = self.get_mask(raw_img, clean_img)

        if self.add_random_masks:
            mask = random_masks(mask.copy(), size=self.img_size[0], offset=10)
        mask = np.where(np.array(mask) > brightness_difference * 255, np.uint8(255), np.uint8(0))
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)

        mask = np.expand_dims(mask, -1)
        mask_t = to_tensor(mask)
        # mask_t = (mask_t > brightness_difference).float()

        # mask_t, _ = torch.max(mask_t, dim=0, keepdim=True)
        binary_mask = (1 - mask_t)  # valid positions are 1; holes are 0
        binary_mask = binary_mask.expand(3, -1, -1)
        clean_img = self.transformer(clean_img)
        corrupted_img = clean_img * binary_mask
        return corrupted_img, binary_mask, clean_img

    @staticmethod
    def get_mask(raw_pil, clean_pil):
        raw = raw_pil.convert("L")
        clean = clean_pil.convert("L")
        mask = ImageChops.difference(raw, clean)
        return mask


def random_masks(pil_img, size=512, offset=10):
    draw = ImageDraw.Draw(pil_img)
    # draw liens
    # can't use np.random because its not forkable under PyTorch's dataloader with multiprocessing
    reps = random.randint(1, 5)

    for i in range(reps):
        cords = np.array(random.choices(range(offset, size), k=4)).reshape(2, 2)
        cords[1] = np.clip(cords[1], a_min=cords[0] - 75, a_max=cords[0] + 75)

        width = random.randint(15, 20)
        draw.line(cords.reshape(-1).tolist(), width=width, fill=255)
    # # draw circles
    reps = random.randint(1, 5)
    for i in range(reps):
        cords = np.array(random.choices(range(offset, size - offset), k=2))
        cords.sort()
        ex = np.array(random.choices(range(20, 70), k=2)) + cords
        ex = np.clip(ex, a_min=offset, a_max=size - offset)
        draw.ellipse(np.concatenate([cords, ex]).tolist(), fill=255)
    return pil_img


class TestDataset(TextSegmentationData):
    def __init__(self, image_folder, max_images=False, image_size=(512, 512), random_crop=True):
        super(TestDataset, self).__init__(image_folder, False, False,
                                          max_images, image_size)
        self.transformer = Compose([ToTensor(),
                                    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
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
        self.eval_imgs = [glob.glob(img_folder + "/*.{}".format(i), recursive=True) for i in ['jpg', 'jpeg', 'png']]
        self.eval_imgs = list(chain.from_iterable(self.eval_imgs))
        assert resize % 8 == 0
        self.resize = resize
        self.transformer = Compose([ToTensor(),
                                    Normalize(mean=mean, std=std),
                                    ])

        print("Find {} test images. ".format(len(self.eval_imgs)))

    def __len__(self):
        return len(self.eval_imgs)

    def __getitem__(self, item):
        img_file = self.eval_imgs[item]
        img = Image.open(img_file).convert("RGB")
        return self.resize_pad_tensor(img), img_file

    def resize_pad_tensor(self, pil_img):
        origin = to_tensor(pil_img).unsqueeze(0)
        fix_len = self.resize
        long = max(pil_img.size)
        ratio = fix_len / long
        new_size = tuple(map(lambda x: int(x * ratio) // 8 * 8, pil_img.size))
        img = pil_img.resize(new_size, Image.BICUBIC)
        # img = pil_img
        img = self.transformer(img).unsqueeze(0)

        _, _, h, w = img.size()
        if fix_len > w:

            boarder_pad = (0, fix_len - w, 0, 0)
        else:

            boarder_pad = (0, 0, 0, fix_len - h)

        img = pad(img, boarder_pad, value=0)
        mask_resizer = self.resize_mask(boarder_pad, pil_img.size)
        return img, origin, mask_resizer

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
