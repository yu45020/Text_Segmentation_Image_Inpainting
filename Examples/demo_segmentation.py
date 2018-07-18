import re
import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from dataloader import EvaluateSet
from models.text_segmentation import TextSegament

model = TextSegament(width_mult=2)
model.total_parameters()

# the model is trained with in-place batch norm, but the weights are compatible with torch's batch norm
own = model.state_dict()
old = torch.load("checkpoints/text_seg_model_BCE_70epos.pt", map_location='cpu')
new_name = list(own)
assert len(new_name) == len(list(old))
new = {k: v for k, v in zip(new_name, old.values())}
model.load_state_dict(new)
model = model.eval()

evalset = EvaluateSet(mean=[0.4935, 0.4563, 0.4544],
                      std=[0.3769, 0.3615, 0.3566],
                      img_folder='images/test_imgs/',
                      resize=600)


def process(eval_img):
    (img, origin, unpadder), file_name = eval_img
    with torch.no_grad():
        out = model(img)

    prob = F.sigmoid(out)
    mask = prob > 0.5
    mask = unpadder(mask)
    origin[mask] = 1
    file_name = re.sub("eval_1", 'output', file_name)
    save_image(origin, file_name + '_out.jpg')
    # save_image(mask, file_name + ' _mask.jpg')


if __name__ == '__main__':
    a = time.time()
    with ThreadPool(cpu_count()) as p:
        p.map(process, evalset)
    print("Runtime :{}".format(time.time() - a))

