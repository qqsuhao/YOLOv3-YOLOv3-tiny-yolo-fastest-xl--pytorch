# -*- coding:utf8 -*-
# @TIME     : 2021/3/18 11:11
# @Author   : SuHao
# @File     : plot.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def plot_imgs_boxes(batch_i, imgs, targets, path="../cache/"):
    '''
    :param imgs: numpy()
    :param targets: numpy()
    :param path:
    :return:
    '''
    os.makedirs(path, exist_ok=True)
    num_samples = imgs.shape[0]
    img_size = imgs.shape[2]

    plt.figure(dpi=300)
    for i in range(num_samples):
        ax = plt.subplot(2, 4, i+1)
        ax.imshow(imgs[i].transpose(1,2,0))
        ax.set_axis_off()
        mask = targets[:, 0] == i
        for j in range(np.sum(mask)):
            w = targets[mask, :][j, 4] * img_size
            h = targets[mask, :][j, 5] * img_size
            x = targets[mask, :][j, 2] * img_size - w / 2
            y = targets[mask, :][j, 3] * img_size - h / 2
            ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor="red", linewidth=2, fill=False))
    plt.savefig(os.path.join(path, str(batch_i)+".jpg"))
    plt.close()