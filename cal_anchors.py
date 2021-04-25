# -*- coding:utf8 -*-
# @TIME     : 2021/4/11 21:04
# @Author   : SuHao
# @File     : cal_anchors.py


from __future__ import division
from utils.datasets import *
from utils.augmentations import *
from utils.parse_config import *
from utils.transforms import *
import argparse
import torch
import tqdm

import os
import numpy as np
from sklearn.cluster import KMeans


def create_filepath_text(path):
    files = os.listdir(path)
    files = [os.path.join(path, i) for i in files]
    with open("train.txt", "w") as f:
        for i in files:
            f.write(i)
            f.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=100, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="data/bac.data", help="path to data config file")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)            # 配置文件包含程序涉及到的相关路径
    train_path = data_config["train"]    # 训练数据集路径
    valid_path = data_config["valid"]    # 验证数据集路径

    # read targets
    size = []
    with open(train_path, "r") as file:      # 按行读取文件的所有内容
        img_files = file.readlines()
    label_files = []
    for path in img_files:
        image_dir = os.path.dirname(path)
        label_dir = "labels".join(image_dir.rsplit("images", 1))
        assert label_dir != image_dir, \
            f"Image path must contain a folder named 'images'! \n'{image_dir}'"
        label_file = os.path.join(label_dir, os.path.basename(path))
        label_file = os.path.splitext(label_file)[0] + '.txt'
        label_files.append(label_file)

    for i in tqdm.tqdm(range(len(img_files))):
        label_path = label_files[i].rstrip()
        boxes = np.loadtxt(label_path).reshape(-1, 5)       # 将label文件加载成5列的矩阵，第一列是类别， 后四列是目标框的位置和大小
        size.append(boxes[:, 3:])


    X = np.concatenate(size, axis=0)
    model = KMeans(n_clusters=6)
    model.fit(X)
    centers = model.cluster_centers_
    centers = centers * opt.img_size
    centers = np.sort(centers, axis=0).astype("int32")
    print(centers)




