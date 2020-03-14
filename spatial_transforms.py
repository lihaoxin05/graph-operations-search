import random
import math
import numbers
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bx, target):
        for t in self.transforms:
            img, bx = t(img, bx, target)
        return img, bx

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    def __call__(self, pic, box, target):
        box = torch.from_numpy(np.array(box))
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255), box
        
    def randomize_parameters(self):
        pass


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, box, target):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, box

    def randomize_parameters(self):
        pass


class Scale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, box, target):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, box
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation), box
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation), box
        else:
            return img.resize(self.size, self.interpolation), box

    def randomize_parameters(self):
        pass


class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, box, target):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        
        box[:,0::2] = (np.clip(box[:,0::2] * w, x1, x1 + tw) - x1) / tw
        box[:,1::2] = (np.clip(box[:,1::2] * h, y1, y1 + th) - y1) / th
        
        return img.crop((x1, y1, x1 + tw, y1 + th)), box

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ['c', 'tl', 'tr', 'bl', 'br']

    def __call__(self, img, box, target):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.))
            y1 = int(round((image_height - th) / 2.))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == 'tr':
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))
        box[:,0::2] = (np.clip(box[:,0::2] * image_width, x1, x2) - x1) / self.size
        box[:,1::2] = (np.clip(box[:,1::2] * image_height, y1, y2) - y1) / self.size

        return img, box

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]


class RandomHorizontalFlip(object):

    def __init__(self, dataset):
        if dataset == 'SomethingSomethingV1':
            self.protect_class = [2, 4, 30, 41, 52, 66]
        elif dataset == 'SomethingSomethingV2':
            self.protect_class = [86, 87, 93, 94, 166, 167]

    def __call__(self, img, box, target):
        if self.p < 0.5 and target not in self.protect_class:
            flip_box = box.copy()
            flip_box[:,0] = 1.0 - box[:,2]
            flip_box[:,1] = 1.0 - box[:,3]
            flip_box[:,2] = 1.0 - box[:,0]
            flip_box[:,3] = 1.0 - box[:,1]
            return img.transpose(Image.FLIP_LEFT_RIGHT), flip_box
        return img, box

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):

    def __init__(self,
                 scales,
                 size,
                 interpolation=Image.BILINEAR,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img, box, target):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == 'tr':
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == 'br':
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))
        box[:,0::2] = (np.clip(box[:,0::2] * image_width, x1, x2) - x1) / crop_size
        box[:,1::2] = (np.clip(box[:,1::2] * image_height, y1, y2) - y1) / crop_size

        return img.resize((self.size, self.size), self.interpolation), box

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, box, target):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))
        box[:,0::2] = (np.clip(box[:,0::2] * image_width, x1, x2) - x1) / crop_size
        box[:,1::2] = (np.clip(box[:,1::2] * image_height, y1, y2) - y1) / crop_size

        return img.resize((self.size, self.size), self.interpolation), box

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()
