import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
import glob
import random

class RandomRotate(object):
    def __call__(self, data):
        dirct = random.randint(0, 3)
        for key in data.keys():
            data[key] = np.rot90(data[key], dirct).copy()
        return data

class RandomFlip(object):
    def __call__(self, data):
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.fliplr(data[key]).copy()
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.flipud(data[key]).copy()
        return data

class RandomCrop(object):
    def __init__(self, Hsize, Wsize):
        super(RandomCrop, self).__init__()
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data

class ToTensor(object):
    def __call__(self, data):

        for key in data.keys():
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).clone()

        return data


class Train_Loader(Dataset):
    def __init__(self, input_path, gt_path, crop_size):
        self.input_name_list = []
        self.gt_name_list = []
        #self.transform = transforms.Compose([RandomCrop(crop_size[0], crop_size[1]), RandomFlip(), RandomRotate, ToTensor()])
        self.transform = transforms.Compose([RandomCrop(crop_size[0], crop_size[1]), ToTensor()])

        input_name_path = sorted(glob.glob(os.path.join(input_path, "*")))
        gt_name_path = sorted(glob.glob(os.path.join(gt_path, "*")))
        for i in range(len(input_name_path)):
            self.input_name_list.append(input_name_path[i])
            self.gt_name_list.append(gt_name_path[i])

        assert len(self.input_name_list) == len(self.gt_name_list), "Missmatched Length!"

    def __len__(self):
        return len(self.input_name_list)

    def __getitem__(self, idx):

        input = cv2.imread(self.input_name_list[idx]).astype(np.float32) / 255
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_name_list[idx]).astype(np.float32) / 255
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        sample = {'input': input,
                  'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample
