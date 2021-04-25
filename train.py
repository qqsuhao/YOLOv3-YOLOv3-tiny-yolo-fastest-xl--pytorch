# -*- coding:utf8 -*-
# @TIME     : 2021/3/18 11:11
# @Author   : SuHao
# @File     : train.py

from __future__ import division

from models.modelv5 import YOLOv3
from models.lossv5 import YOLOLoss_total
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--start_epoch", type=int, default=0, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="configs/yolov3-tiny-bac.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="data/bac.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  # , default="weights/yolo-fastest-xl.weights"
    parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=100, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")
    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored")
    opt = parser.parse_args()
    print(opt)

    logger = Logger(opt.logdir)

    # file
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # cuda
    cudnn.deterministic = False
    cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()
    opt.batch_size *= n_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)            # 配置文件包含程序涉及到的相关路径
    train_path = data_config["train"]    # 训练数据集路径
    valid_path = data_config["valid"]    # 验证数据集路径
    class_names = load_classes(data_config["names"])            # 各种类别的名称

    # Initiate model
    model = YOLOv3(opt.model_def).to(device)                   # 建立模型
    lr = float(model.hyperparams["learning_rate"])
    # model.apply(weights_init_normal)                            # 权值随机初始化

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))       # 加载pytorch格式的权重文件
            print("loading pth")
        else:
            model.load_darknet_weights(opt.pretrained_weights)          # 加载darknet格式的权重文件。以.weight为后缀
            print("loading weights")
    else:
        model.apply(weights_init_normal)

    # fix some parameters
    # for i in model.modules():
    #     if i.__class__.__name__ == "Conv2d":
    #         if i.out_channels == 255:
    #             for j in i.parameters():
    #                 j.requires_grad = True
    #         else:
    #             for j in i.parameters():
    #                 j.requires_grad = False
    #     else:
    #         for j in i.parameters():
    #             j.requires_grad = False


    if n_gpu > 0:
        model = torch.nn.DataParallel(model)

    # Get dataloader
    dataset = ListDataset(train_path, multiscale=opt.multiscale_training, img_size=opt.img_size, transform=AUGMENTATION_TRANSFORMS)
    # 加载数据集并进行数据增强；使用一个开源的数据增强库完成数据增强。
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        # pin_memory就是锁页内存，pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，
        # 这样将内存的Tensor转义到GPU的显存就会更快一些。
        collate_fn=dataset.collate_fn,      # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)            # Adam优化器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    scheduler.last_epoch = opt.start_epoch - 1

    scaler = amp.GradScaler(enabled=True)

    Loss = YOLOLoss_total(opt.model_def)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    num_warm = max(3*len(dataloader), 1e3)
    nbs = 48  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)
    accumulate = 1


    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        Loss.train()
        optimizer.zero_grad()
        start_time = time.time()
        loss = 0
        pbar = tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")
        for batch_i, (_, imgs, targets) in enumerate(pbar):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device)
            targets = targets.to(device)

            # if batches_done <= num_warm:
            #     xi = [0, num_warm]
            #     accumulate = max(1, np.interp(batches_done, xi, [1, nbs/opt.batch_size]).round())
            #     for x in optimizer.param_groups:
            #         x['lr'] = np.interp(batches_done, xi, [0.0, x['initial_lr'] * lf(epoch)])


            # Autocast
            # with amp.autocast(enabled=True):
            outputs = model(imgs)
            _, loss = Loss(outputs, targets)
            # print(loss.detach().to("cpu").item())
            # loss.backward()
            scaler.scale(loss).backward()

            if batches_done % opt.gradient_accumulations == 0:      # 梯度累加
                # Accumulates gradient before each step
                # optimizer.step()
                # optimizer.zero_grad()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'loss' : '{0:1.4f}'.format(loss.detach().to("cpu").item())})
            pbar.update(1)
            # -------------------------------------------------------------------
            #   Log progress
            # -------------------------------------------------------------------

            # log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            #
            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(Loss.Loss_list))]]]
            #
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics):
            #     formats = {m: "%.6f" for m in metrics}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in Loss.Loss_list]
            #     metric_table += [[metric, *row_metrics]]
            #
            # log_str += AsciiTable(metric_table).table
            # log_str += f"\nTotal loss {to_cpu(loss).item()}"
            #
            # Tensorboard logging
            # tensorboard_log = []
            # for j, yolo in enumerate(Loss.Loss_list):
            #     for name, metric in yolo.metrics.items():
            #         if name != "grid_size":
            #             tensorboard_log += [(f"train/{name}_{j+1}", metric)]
            # tensorboard_log += [("train/loss", to_cpu(loss).item())]
            # logger.list_of_scalars_summary(tensorboard_log, batches_done)
            logger.scalar_summary("loss", loss.detach().to("cpu").item(), batches_done)

            # # Determine approximate time left for epoch
            # epoch_batches_left = len(dataloader) - (batch_i + 1)
            # time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            # log_str += f"\n---- ETA {time_left}"
            #
            # if opt.verbose: print(log_str)

            model.module.seen += imgs.size(0)
            # if batch_i > 30: break

        scheduler.step()

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = evaluate(
                model,
                Loss,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=20,
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean()),
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
            else:
                print( "---- mAP not measured (no detections found by model)")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.module.state_dict() if n_gpu>0 else model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

    torch.save(model.module.state_dict() if n_gpu>0 else model.state_dict(), f"weights/yolov3.pth")

