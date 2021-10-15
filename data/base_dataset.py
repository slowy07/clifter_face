import torch.utils.data as data
from PIL import Image
import torch.vision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return "BaseDataset"

    def initalize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w

    if opt.resize_or_crop == "resize_and_crop":
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == "scale_width_and_crop":
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


def get_transfrom(opt, params, method=Image.BICUBIC, normalize=True):
    transfrom_list = []
    if "resize" in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transfrom_list.append(transforms.Scale(osize, method))
