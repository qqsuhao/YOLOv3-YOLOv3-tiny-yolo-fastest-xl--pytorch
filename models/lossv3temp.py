# -*- coding:utf8 -*-
# @TIME     : 2021/4/12
# @Author   : SuHao

# https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/yolo_loss.py

import math
import torch
import torch.nn as nn
from utils.parse_config import parse_model_config
import numpy as np

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou



class YOLOLoss(nn.Module):
    def __init__(self, config_path, yolo=0):
        '''
        :param config_path: 从cfg配置文件中解析各种参数
        :param yolo 为0,1,2 指定YOLO层
        '''
        super(YOLOLoss, self).__init__()
        self.module_defs = parse_model_config(config_path)                              # 解析网络配置文件
        self.hyperparams = self.module_defs.pop(0)              # 超参数
        self.anchors = []                                       # 存放anchor的长宽
        for module_def in self.module_defs:
            if module_def["type"] == "yolo":                          # 寻找YOLO块
                if module_def["mask"][0] == str(3*yolo):           # 寻找是否是对应序号的yolo块
                    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]       # Anchor的序号，yolov3中每个特征图有3个Anchor
                    anchors = [int(x) for x in module_def["anchors"].split(",")]
                    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                    self.anchors = [anchors[i] for i in anchor_idxs]         # 提取3个Anchor
                    break
        self.num_anchors = len(self.anchors)                    # anchor的数量
        self.num_classes = int(module_def["classes"])           # 类别的数量
        self.img_dim = int(self.hyperparams["width"])
        self.ignore_thres = float(module_def["ignore_thresh"])
        self.obj_scale = 1
        self.noobj_scale = 1
        self.metrics = {}
        self.grid_size = torch.tensor(0)  # grid size

        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()            # 最小均方误差
        self.bce_loss = nn.BCELoss()          # 二分类交叉熵


    def forward(self, pred, targets=None):
        # yolo层的输入是特征图，即pred是特征图，维度是样本数量*(（5+类别数量）*3)*13*13
        # Tensors for cuda support
        if targets is not None:
            FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor
            num_samples = pred.size(0)      # 样本数量
            import math
            grid_size = int(math.sqrt(pred.size(2) / self.num_anchors))       # 此时网格尺寸应该是13或26或52
            self.stride = self.img_dim / grid_size
            prediction = (
                pred.view(num_samples, self.num_anchors, grid_size, grid_size, self.num_classes + 5)        # pred的维度是n*(self.num_anchors*grid_size*grid_size)*85
                    .contiguous()
                #  当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            )
            pred_boxes = prediction[..., 0:4] / self.stride
            x = pred_boxes[..., 0] - pred_boxes[..., 0].floor()
            y = pred_boxes[..., 1] - pred_boxes[..., 1].floor()
            w = pred_boxes[..., 2]
            h = pred_boxes[..., 3]
            scaled_anchors = FloatTensor([(a_w/self.stride, a_h/self.stride) for a_w, a_h in self.anchors])    # 对Anchor的坐标位置进行缩放
            anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
            w = torch.log(w / anchor_w + 1e-16)
            h = torch.log(h / anchor_h + 1e-16)
            conf = prediction[..., 4]
            pred_cls = prediction[..., 5:]
            #  build target
            scaled_anchors = [(a_w/self.stride, a_h/self.stride) for a_w, a_h in self.anchors]
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                           grid_size, grid_size,
                                                                           self.ignore_thres)
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            #  losses.
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + \
                0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
            return loss
        else:
            return pred


    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


class YOLOLoss_total(nn.Module):
    def __init__(self, config_path):
        super(YOLOLoss_total, self).__init__()
        self.config_path = config_path


    def forward(self, x, targets=None):
        self.L = len(x)
        self.loss = 0
        self.Loss_list = []
        self.outputs = []
        self.outs = []
        self.params = {"anchors": [], "strides": []}
        if targets is not None:
            for i in range(self.L):
                self.Loss_list.append(YOLOLoss(self.config_path, yolo=self.L - 1 - i))
                loss = self.Loss_list[-1](x[i], targets)
                self.loss += loss
            return self.loss
        else:
            for i in range(self.L):
                self.Loss_list.append(YOLOLoss(self.config_path, yolo=self.L - 1 - i))
                output = self.Loss_list[-1](x[i], targets)
                self.outputs.append(output)
            self.outputs = torch.cat(self.outputs, 2)
            self.outputs = torch.squeeze(self.outputs, dim=1)
            return self.outputs