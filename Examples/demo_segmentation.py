import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from Dataloader import EvaluateSet
from models.text_segmentation import TextSegament, XceptionTextSegment


def draw_bounding_box(img, mask, area_threshold=100):
    b, c = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in b:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(img, [hull], 0, (50, 128, 30), -1)

    return img


def process(eval_img, device='cpu'):
    (img, origin, unpadder), file_name = eval_img
    with torch.no_grad():
        out = model(img.to(device))

    prob = F.sigmoid(out)
    mask = prob > 0.5
    mask = torch.nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=1)(mask.float()).byte()
    mask = unpadder(mask)
    mask = mask.float().cpu()

    save_image(mask, file_name + ' _mask.jpg')
    origin_np = np.array(to_pil_image(origin[0]))
    mask_np = to_pil_image(mask[0]).convert("L")
    mask_np = np.array(mask_np, dtype='uint8')
    mask_np = draw_bounding_box(origin_np, mask_np, 500)
    mask_ = Image.fromarray(mask_np)
    mask_.save(file_name + "_contour.jpg")
    # ret, mask_np = cv2.threshold(mask_np, 127, 255, 0)
    # dst = cv2.inpaint(origin_np, mask_np, 1, cv2.INPAINT_NS)
    # out = Image.fromarray(dst)
    # out.save(file_name + ' _box.jpg')


model = XceptionTextSegment()
model.total_parameters()

# the model is trained with in-place batch norm, but the weights are compatible with torch's batch norm
model.load_state_dict(torch.load("checkpoints/text_seg_model_681epos.pt", map_location='cpu'))
model = model.cuda()

evalset = EvaluateSet(mean=[0.4935, 0.4563, 0.4544],
                      std=[0.3769, 0.3615, 0.3566],
                      img_folder='test_data',
                      resize=600)

a = time.time()
for i in evalset:
    process(i, 'cuda')

# with ThreadPool(cpu_count() - 1) as p:
#     p.map(process, evalset)
print("Runtime :{}".format(time.time() - a))
