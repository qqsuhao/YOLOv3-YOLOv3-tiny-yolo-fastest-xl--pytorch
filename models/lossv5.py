# -*- coding=utf-8 -*-

import math
import torch
import torch.nn as nn
from utils.parse_config import parse_model_config


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(pred, targets, params):
    anchors = params["anchors"]
    strides = params["strides"]
    assert (len(anchors) == len(strides))
    num_anchor = len(anchors[0])

    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)
    # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])).to(device)

    BCEcls = nn.BCELoss().to(device)
    BCEobj = nn.BCELoss().to(device)

    cp, cn = smooth_BCE(eps=0.0)

    tcls, tbox, indices, anchors = build_targets(pred, targets, anchors, strides)

    n_scale = len(pred)
    balance = [4.0, 1.0] if n_scale == 2 else [4.0, 1.0, 0.4]

    for s, pred_s in enumerate(pred):  # each scale

        shape = pred_s.shape
        num_out = int(shape[1] / num_anchor)
        pred_s = pred_s.view(shape[0], num_anchor, num_out, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        b, a, gj, gi = indices[s]  # img, anchor, gridy, gridx
        tobj = torch.zeros_like(pred_s[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            ps = pred_s[b, a, gj, gi]  # N * (num_cls+5)

            # Regression (giou loss)
            # pxy = ps[:, :2].sigmoid() * 2. - 0.5
            # pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[s]
            pxy = ps[:, :2] * 2. - 0.5
            pwh = (ps[:, 2:4] * 2) ** 2 * anchors[s]
            pbox = torch.cat((pxy, pwh), dim=1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[s], x1y1x2y2=False, CIoU=True)

            # lbox += (1.0 - iou).mean()
            l1_loss = nn.functional.smooth_l1_loss(pbox, tbox[s], reduction='mean')
            lbox += l1_loss

            # Objectness
            iou_ratio = 0.5
            tobj[b, a, gj, gi] = (1.0 - iou_ratio) + iou_ratio * iou.detach().clamp(0).type(tobj.dtype)

            # Classification
            if num_out - 5 > 1:  # only if multiple classes
                t = torch.full_like(ps[:, 5:], cn, device=device)
                t[range(n), tcls[s]] = cp
                lcls += BCEcls(ps[:, 5:], t)
        lobj += BCEobj(pred_s[..., 4], tobj) * balance[s]  # obj loss

    lbox *= 0.05
    lobj *= 1.0 * (1.4 if n_scale == 3 else 1.)
    lcls *= 0.5
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls)).detach()


def build_targets(pred, targets, anchors, strides):
    '''
        pred     --->  [scale_1, scale2 ....]
        targets  --->  N*6 (image,class,x,y,w,h)
    '''
    tcls, tbox, indices, anch = [], [], [], []
    anchors = torch.Tensor(anchors).cuda()

    num_anchor, num_target = len(anchors[0]), targets.shape[0]

    ai = torch.arange(num_anchor, device=targets.device).float().view(num_anchor, 1).repeat(1, num_target)
    targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:, :, None]), 2)
    # targets.shape --> 3*N*7, 3是每个尺度的anchor个数(将gt复制3份)

    offset = 0.5 * torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()

    gain = torch.ones(7, device=targets.device)
    for s, stride in enumerate(strides):

        gain[2:6] = torch.tensor(pred[s].shape)[[3, 2, 3, 2]]  # (80,80,80,80)或 40或 20
        t = targets * gain  # 3*N*7

        anchor_s = anchors[s] / stride

        if num_target:
            r = t[:, :, 4:6] / anchor_s[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < 4.0
            t = t[j]

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1. < 0.5) & (gxy > 1.)).T
            l, m = ((gxi % 1. < 0.5) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        b, c = t[:, :2].long().T  # img batch_id, class // shape --> N
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchor_s[a])
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


class YOLOLoss(nn.Module):
    def __init__(self, config_path, yolo=0):
        '''
        :param config_path: 从cfg配置文件中解析各种参数
        :param yolo 为0,1,2 指定YOLO层
        '''
        super(YOLOLoss, self).__init__()
        self.module_defs = parse_model_config(config_path)  # 解析网络配置文件
        self.hyperparams = self.module_defs.pop(0)  # 超参数
        self.anchors = []  # 存放anchor的长宽
        for module_def in self.module_defs:
            if module_def["type"] == "yolo":  # 寻找YOLO块
                if module_def["mask"][0] == str(3 * yolo):  # 寻找是否是对应序号的yolo块
                    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]  # Anchor的序号，yolov3中每个特征图有3个Anchor
                    anchors = [int(x) for x in module_def["anchors"].split(",")]
                    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                    self.anchors = [anchors[i] for i in anchor_idxs]  # 提取3个Anchor
                    break
        self.num_anchors = len(self.anchors)  # anchor的数量
        self.num_classes = int(module_def["classes"])  # 类别的数量
        self.img_dim = int(self.hyperparams["width"])
        self.ignore_thres = float(module_def["ignore_thresh"])
        self.obj_scale = 1
        self.noobj_scale = 1
        self.metrics = {}
        self.grid_size = torch.tensor(0)  # grid size

        self.mse_loss = nn.MSELoss()  # 最小均方误差
        self.bce_loss = nn.BCELoss()  # 二分类交叉熵

    def forward(self, pred, targets=None):
        # yolo层的输入是特征图，即pred是特征图，维度是样本数量*(（5+类别数量）*3)*13*13
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor
        num_samples = pred.size(0)  # 样本数量
        import math
        grid_size = int(math.sqrt(pred.size(2) / self.num_anchors))  # 此时网格尺寸应该是13或26或52
        self.stride = self.img_dim / grid_size
        prediction = (
            pred.view(num_samples, self.num_anchors, grid_size, grid_size,
                      self.num_classes + 5)  # pred的维度是n*1*(self.num_anchors*grid_size*grid_size)*85
                .contiguous()
            #  当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
        )
        self.grid_size = grid_size
        self.grid_x = FloatTensor([i for j in range(self.grid_size) for i in range(self.grid_size)])\
            .view([1, 1, self.grid_size, self.grid_size, 1])
        self.grid_y = FloatTensor([j for j in range(self.grid_size) for i in range(self.grid_size)])\
            .view([1, 1, self.grid_size, self.grid_size, 1])
        pred_boxes = prediction[..., 0:4]
        pred_x = (pred_boxes[..., 0:1] / self.stride - self.grid_x + 0.5) / 2
        pred_y = (pred_boxes[..., 1:2] / self.stride - self.grid_y + 0.5) / 2

        pred_w = pred_boxes[..., 2:3]
        pred_h = pred_boxes[..., 3:4]
        scaled_anchors = FloatTensor(
            [(a_w, a_h) for a_w, a_h in self.anchors])  # 对Anchor的坐标位置进行缩放
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1, 1))
        pred_w = torch.sqrt(pred_w / anchor_w) / 2
        pred_h = torch.sqrt(pred_h / anchor_h) / 2
        prediction = torch.cat((pred_x, pred_y, pred_w, pred_h, prediction[..., 4:]), -1)
        prediction = prediction.permute(0, 1, 4, 2, 3)
        prediction = (prediction.contiguous()
                      .view(num_samples, self.num_anchors * (self.num_classes + 5), grid_size, grid_size)
                      )
        if targets is not None:
            return prediction, self.anchors, self.stride
        else:
            return pred


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
                output, anchors, stride = self.Loss_list[-1](x[i], targets)
                self.outs.append(output)
                self.params["anchors"].append(anchors)
                self.params["strides"].append(stride)
            loss, lossitems = compute_loss(self.outs, targets, self.params)
            return lossitems, loss
        else:
            for i in range(self.L):
                self.Loss_list.append(YOLOLoss(self.config_path, yolo=self.L - 1 - i))
                output = self.Loss_list[-1](x[i], targets)
                self.outputs.append(output)
            self.outputs = torch.cat(self.outputs, 2)
            self.outputs = torch.squeeze(self.outputs, dim=1)
            return self.outputs