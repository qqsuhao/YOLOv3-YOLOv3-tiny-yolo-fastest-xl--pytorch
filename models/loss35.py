import torch
import torch.nn as nn
from utils.utils import bbox_iou, bbox_wh_iou, to_cpu
from utils.parse_config import parse_model_config



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, stride):
    '''
    :param pred_boxes: 预测框的位置和长宽 (num_samples, self.num_anchors, grid_size, grid_size, 4)
    :param pred_cls: 预测类别的概率
    :param target: 真值
    :param anchors: Anchor，存在矩阵里
    :param ignore_thres: 默认设为0.5
    :return:
    '''
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    device = torch.device("cuda") if pred_boxes.is_cuda else torch.device("cpu")

    nB = pred_boxes.size(0)     # 样本数量
    nA = pred_boxes.size(1)     # Anchor的数量：3
    nC = pred_cls.size(-1)      # 类别数
    nG = pred_boxes.size(2)     # 13

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, requires_grad=False).to(device)            # 目标掩码，bool，用于存放该处格点是否有目标，0填充
    noobj_mask = torch.ones(nB, nA, nG, nG, requires_grad=False).to(device)            # 无目标掩码
    class_mask = torch.zeros(nB, nA, nG, nG, requires_grad=False).to(device)            # 类别掩码
    iou_scores = torch.zeros(nB, nA, nG, nG, requires_grad=False).to(device)            # IOU分数
    tx = torch.zeros(nB, nA, nG, nG, requires_grad=True).to(device)                    # 真值相对于网格点的偏离值
    ty = torch.zeros(nB, nA, nG, nG, requires_grad=True).to(device)                    # 真值相对于网格点的偏离值
    tw = torch.zeros(nB, nA, nG, nG, requires_grad=True).to(device)                    # 真值相对于网格点的偏离值
    th = torch.zeros(nB, nA, nG, nG, requires_grad=True).to(device)                    # 真值相对于网格点的偏离值
    tcls = torch.zeros(nB, nA, nG, nG, nC, requires_grad=False).to(device)              #存放类别


    # Convert to position relative to box
    # target中存放每个样本的真值，包括目标位置及其所属类别
    # traget一共6列，第一列表示样本序号，第二列表示标签，第三列到第六列表示位置坐标和长宽。
    # 因为一个样本可能包含多个目标，因此需要一个维度用于标记该目标框属于哪个样本
    target_boxes = target[:, 2:6] * nG  # 由于位置真值是以归一化值保存的，因此需要乘上特征图大小
    gxy = target_boxes[:, :2]           # 目标位置真值
    gwh = target_boxes[:, 2:]           # 目标长宽真值
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)             # 分别计算每个Anchor与真值之间的IOU，挑选最大的Anchor作为最优选项
    # Separate target values
    b, target_labels = target[:, :2].long().t()         # .long()进行数据类型转换；  .t()进行转置，而且是深拷贝
    gx, gy = gxy.t()            # 目标位置真值
    gw, gh = gwh.t()            # 目标长宽真值
    gi, gj = gxy.long().t()     #
    ########## TODO(arthur77wang):
    gi[gi < 0] = 0
    gj[gj < 0] = 0
    gi[gi > nG - 1] = nG - 1
    gj[gj > nG - 1] = nG - 1
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1             # 为1的元素表示对应位置有目标。且标出了对应的最佳anchor
    noobj_mask[b, best_n, gj, gi] = 0           # 为1的元素表示对应位置无目标

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0      # 通过门限设置对应位置是否有目标

    # Coordinates
    # 计算偏移量

    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = (gy - gy.floor() + 0.5) / 2
    # Width and height
    # tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    # th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    tw[b, best_n, gj, gi] = torch.sqrt(gw / anchors[best_n][:, 0]) / 2
    th[b, best_n, gj, gi] = torch.sqrt(gh / anchors[best_n][:, 1]) / 2


    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1          # 类别概率真值
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()       # 类别的掩码矩阵
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)   # 有目标的位置与对应的最佳anchor的iou分数

    tconf = obj_mask            # 置信度的真值
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf



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
        self.metrics = {}
        self.grid_size = torch.tensor(0)  # grid size
        self.lambda_xy = 10
        self.lambda_wh = 10
        self.lambda_conf = 1.0
        self.lambda_cls = 10

        self.mse_loss = nn.MSELoss()            # 最小均方误差
        self.bce_loss = nn.BCELoss()          # 二分类交叉熵


    def forward(self, pred, targets=None):
        if targets is not None:
            # yolo层的输入是特征图，即pred是特征图，维度是样本数量*(（5+类别数量）*3)*13*13
            # Tensors for cuda support
            FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if pred.is_cuda else torch.LongTensor
            num_samples = pred.size(0)      # 样本数量
            import math
            grid_size = int(math.sqrt(pred.size(2) / self.num_anchors))       # 此时网格尺寸应该是13或26或52
            self.grid_size = grid_size
            self.stride = self.img_dim / grid_size
            prediction = (
                pred.view(num_samples, self.num_anchors, grid_size, grid_size, self.num_classes + 5)        # pred的维度是n*(self.num_anchors*grid_size*grid_size)*85
                    .contiguous()
                #  当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            )
            self.grid_x = FloatTensor([i for j in range(self.grid_size) for i in range(self.grid_size)]) \
                .view([1, 1, self.grid_size, self.grid_size])
            self.grid_y = FloatTensor([j for j in range(self.grid_size) for i in range(self.grid_size)]) \
                .view([1, 1, self.grid_size, self.grid_size])
            pred_boxes = prediction[..., 0:4]
            pred_x = (pred_boxes[..., 0] / self.stride - self.grid_x + 0.5) / 2
            pred_y = (pred_boxes[..., 1] / self.stride - self.grid_y + 0.5) / 2
            pred_w = pred_boxes[..., 2]
            pred_h = pred_boxes[..., 3]
            scaled_anchors = FloatTensor([(a_w, a_h) for a_w, a_h in self.anchors])    # 对Anchor的坐标位置进行缩放
            anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
            pred_w = torch.sqrt(pred_w / anchor_w) / 2
            pred_h = torch.sqrt(pred_h / anchor_h) / 2
            pred_conf = prediction[..., 4]
            pred_cls = prediction[..., 5:]

            # 接下来的代码主要是为了进行性能评估
            scaled_anchors = FloatTensor([(a_w/self.stride, a_h/self.stride) for a_w, a_h in self.anchors])    # 对Anchor的坐标位置进行缩放
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=scaled_anchors,
                ignore_thres=self.ignore_thres,
                stride=self.stride,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(pred_x[obj_mask==1], tx[obj_mask==1])
            loss_y = self.mse_loss(pred_y[obj_mask==1], ty[obj_mask==1])
            loss_w = self.mse_loss(pred_w[obj_mask==1], tw[obj_mask==1])
            loss_h = self.mse_loss(pred_h[obj_mask==1], th[obj_mask==1])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask==1], obj_mask[obj_mask==1])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask==1], obj_mask[noobj_mask==1])
            loss_conf = loss_conf_obj + 100*loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask==1], tcls[obj_mask==1])
            total_loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                         loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                         loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
            # print("loss_x: ", loss_x.detach().to("cpu").item())
            # print("loss_y: ", loss_y.detach().to("cpu").item())
            # print("loss_w: ", loss_w.detach().to("cpu").item())
            # print("loss_h: ", loss_h.detach().to("cpu").item())
            # print("loss_conf: ", loss_conf_noobj.detach().to("cpu").item())
            # print("loss_conf: ", loss_conf_obj.detach().to("cpu").item())
            # print("loss_cls: ", loss_cls.detach().to("cpu").item())


            # Metrics               # 对模型的评估
            cls_acc = 100 * class_mask[obj_mask==1].mean()
            conf_obj = pred_conf[obj_mask==1].mean()
            conf_noobj = pred_conf[noobj_mask==1].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)       # 精确率
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)      # 召回率
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return pred, total_loss
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
        if targets is not None:
            for i in range(self.L):
                self.Loss_list.append(YOLOLoss(self.config_path, yolo=self.L - 1 - i))
                output, layer_loss = self.Loss_list[-1](x[i], targets)
                self.loss += layer_loss
                self.outputs.append(output)
            self.outputs = torch.cat(self.outputs, 2)
            self.outputs = torch.squeeze(self.outputs, dim=1)
            return self.outputs, self.loss
        else:
            for i in range(self.L):
                self.Loss_list.append(YOLOLoss(self.config_path, yolo=self.L - 1 - i))
                output = self.Loss_list[-1](x[i], targets)
                self.outputs.append(output)
            self.outputs = torch.cat(self.outputs, 2)
            self.outputs = torch.squeeze(self.outputs, dim=1)
            return self.outputs