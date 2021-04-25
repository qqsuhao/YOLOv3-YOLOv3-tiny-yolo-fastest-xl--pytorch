import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True      # 防止出现image file is truncated错误
# 如果图片出现损坏，会报image file is truncated错误

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    '''
    沿着四个方向进行填充，目的是把图片填充成方形
    '''
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)      # 沿着四个方向进行填充，目的是把图片填充成方形

    return img, pad


def resize(image, size):            # 使用最近邻插值缩放图片
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    '''
    加载文件夹中的数据，主要是在检测样例图片时用的，因为不需要加载标签
    '''
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))   # glob.glob() 是一个正则匹配函数
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'), 
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:      # 按行读取文件的所有内容
            self.img_files = file.readlines()

        # 存放图片的文件和存放标签的文件存放一个目录下，分别命名为images 和 labels
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size            # 图像尺寸
        self.max_objects = 100              # 目标数量上限
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32  # 图像最小尺寸
        self.max_size = self.img_size + 3 * 32  # 图像最大尺寸
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        
        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()     # .strip删除末尾的空格

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8) # 打开图像
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")         # 忽略警告
                boxes = np.loadtxt(label_path).reshape(-1, 5)       # 将label文件加载成5列的矩阵，第一列是类别， 后四列是目标框的位置和大小
        except Exception as e:
            # print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:          # 对图像进行变换
            try:
                img, bb_targets = self.transform((img, boxes))      # 对图像进行变换的同同时，需要对图像的标签为位置数据也同步进行变换
            except:
                print(f"Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        '''
        collate_fn就是在加载数据过程中，对批数据进行一个自定义的处理
        在这个程序中，使用collate_fn实现了：
        每次经过10个batch以后，都会随机地对batch中的图像重resize一次，从而使得训练数据中有一部分图像是经过拉伸或者压缩的，
        从而起到数据增强的作用。避免模型对于一些拉宽或者压扁了的图像识别不准确
        另外，这个函数还给batch中的数据加上了序号，这样一来bb_targets就是六列了。
        '''
        # batch中存放了批数据，batch中的元素是元组，元组中包含img_path, img, bb_targets这个三个成员
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
