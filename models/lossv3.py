import torch
import torch.nn as nn
from utils.utils import build_targets, to_cpu
from utils.parse_config import parse_model_config


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
        self.lambda_xy = 2
        self.lambda_wh = 2
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

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
            self.stride = self.img_dim / grid_size
            prediction = (
                pred.view(num_samples, self.num_anchors, grid_size, grid_size, self.num_classes + 5)        # pred的维度是n*(self.num_anchors*grid_size*grid_size)*85
                    .contiguous()
                #  当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            )
            pred_boxes = prediction[..., 0:4] / self.stride
            pred_x = pred_boxes[..., 0] - pred_boxes[..., 0].floor()
            pred_y = pred_boxes[..., 1] - pred_boxes[..., 1].floor()
            pred_w = pred_boxes[..., 2]
            pred_h = pred_boxes[..., 3]
            scaled_anchors = FloatTensor([(a_w/self.stride, a_h/self.stride) for a_w, a_h in self.anchors])    # 对Anchor的坐标位置进行缩放
            anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
            anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
            pred_w = torch.log(pred_w / anchor_w)
            pred_h = torch.log(pred_h / anchor_h)
            pred_conf = prediction[..., 4]
            pred_cls = prediction[..., 5:]

            # 接下来的代码主要是为了进行性能评估
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(pred_x[obj_mask==1], tx[obj_mask==1])
            loss_y = self.mse_loss(pred_y[obj_mask==1], ty[obj_mask==1])
            loss_w = self.mse_loss(pred_w[obj_mask==1], tw[obj_mask==1])
            loss_h = self.mse_loss(pred_h[obj_mask==1], th[obj_mask==1])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask==1], obj_mask[obj_mask==1])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask==1], obj_mask[noobj_mask==1])
            loss_conf = loss_conf_obj + 200*loss_conf_noobj
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