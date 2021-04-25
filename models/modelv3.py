# -*- coding:utf8 -*-
# @TIME     : 2021/3/18 10:27
# @Author   : SuHao
# @File     : model.py

import torch.nn as nn
from utils.parse_config import *
from utils.utils import *

def creat_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    根据cfg配置文件创建网络
    """
    hyperparams = module_defs.pop(0)            # 超参数
    output_filters = [int(hyperparams["channels"])]     # 输入层的输出通道数
    module_list = nn.ModuleList()               # 存放模块的列表
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":               # 根据配置文件创建卷积块，包含BN层+卷积层+激活函数层
            bn = int(module_def["batch_normalize"])             # 表示是否要进行BN
            filters = int(module_def["filters"])                # 输出通道数
            kernel_size = int(module_def["size"])               # 卷积核大小
            pad = (kernel_size - 1) // 2                        # 填充
            groups = int(module_def["groups"]) if "groups" in module_def.keys() else 1    # 分组卷积
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],             # 输入通道数：上一个网络模块的输出
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    groups=groups,
                    bias=not bn,                                # 如果要进行BN就没有偏执，如果不进行BN，就没有偏置
                    # 因为如果要进行BN，偏置会在BN的计算过程中抵消掉，不起作用，因此还不如直接取消偏置，减少参数量
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))       # 添加BN
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))          # 添加LeakyReLU

        elif module_def["type"] == "maxpool":                       # 最大池化层
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])          # 步长
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))            # 0填充
                # nn.ZeroPad2d沿着四个方向进行补零操作
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))     # 最大池化
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":                      # 上采样
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode='nearest')
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":                         # 融合层
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())           # 作者创建了一个空层，相关操作在后续

        elif module_def["type"] == "shortcut":              # 残差网络中的相加
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())         # 作者创建了一个空层，相关操作在后续

        ## 我自己加的dropout
        elif module_def["type"] == "dropout":
            drop = nn.Dropout(p=float(module_def["probability"]))
            modules.add_module(f"dropout_{module_i}", drop)

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]       # Anchor的序号，yolov3中每个特征图有3个Anchor
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]         # 提取3个Anchor
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["width"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)                         # 向列表中添加模块
        output_filters.append(filters)
    return hyperparams, module_list



class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.img_dim = img_dim


    def forward(self, inputs):
        self.grid_size = inputs.size(2)
        self.num_samples = inputs.size(0)
        inputs_view = inputs.view(self.num_samples, self.num_anchors, self.num_classes + 5,
                      self.grid_size, self.grid_size).permute(0, 1, 3, 4, 2).contiguous()

        FloatTensor = torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if inputs.is_cuda else torch.LongTensor
        # Get outputs
        # x,y,w,h都是预测的偏置值
        x = torch.sigmoid(inputs_view[..., 0])  # Center x
        y = torch.sigmoid(inputs_view[..., 1])  # Center y
        w = inputs_view[..., 2]  # Width
        h = inputs_view[..., 3]  # Height
        pred_conf = torch.sigmoid(inputs_view[..., 4])  # Conf   置信度
        pred_cls = torch.sigmoid(inputs_view[..., 5:])  # Cls pred.      每个类别对应的概率

        # 重新计算偏置
        self.stride = torch.floor_divide(self.img_dim, self.grid_size)             # 计算图像到特征图的缩放倍数 416/13=32
        # Calculate offsets for each grid
        g = self.grid_size
        # self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        # self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        '''
        使用上述方法求grid_x会导致dnn加载onnx出错，即使转onnx的时候并不会出错
        dnn内部似乎不支持arange和repeat这两种操作，因此我们使用列表推导的方式替代arange和repeat
        '''
        self.grid_x = FloatTensor([i for j in range(self.grid_size) for i in range(self.grid_size)])\
            .view([1, 1, self.grid_size, self.grid_size])
        # 有点像meshgrid的意思，先构造一个二维矩阵，每行都是从0到13的13个整数，然后view成一个四个维度的矩阵
        # 对应了yolo输出的第5个维度的第一层，即prediction[..., 0]
        self.grid_y = FloatTensor([j for j in range(self.grid_size) for i in range(self.grid_size)])\
            .view([1, 1, self.grid_size, self.grid_size])
        self.anchor_w = [self.anchors[i][0]/self.stride for i in range(self.num_anchors)]
        self.anchor_h = [self.anchors[i][1]/self.stride for i in range(self.num_anchors)]       # 列表self.anchor_h里的元素是tensor

        # 在偏置值的基础上得到预测的边框的位置
        # 源代码写作 x.data + self.grid_x, 转为onnx并使用dnn加载会出错
        X = FloatTensor()           # x 和 self.grid_x维度并不完全相同，为了转onnx成功，需要写成这样
        for i in range(self.num_anchors):
            X = torch.cat( (X, torch.add(x[:, i:i+1, :, :], self.grid_x)), 1)
        Y = FloatTensor()
        for i in range(self.num_anchors):
            Y = torch.cat( (Y, torch.add(y[:, i:i+1, :, :], self.grid_y)), 1)
        W = FloatTensor()
        expw = torch.exp(w)                    # 貌似DNN不支持使用exp，没有办法，我们不得不将exp替换为平方，对应地求解loss部分也需要进行替换
        # expw = w ** 2
        for i in range(self.num_anchors):
            W = torch.cat( (W, torch.mul(expw[:, i:i+1, :, :], self.anchor_w[i])), 1)
        H = FloatTensor()
        exph = torch.exp(h)
        # exph = h ** 2
        for i in range(self.num_anchors):
            H = torch.cat( (H, torch.mul(exph[:, i:i+1, :, :], self.anchor_h[i])), 1)

        '''
        这里源代码写作 W = torch.exp(w.data) * self.anchor_w
        但是源代码中的torch.exp(w.data)， self.anchor_w的唯独不相等，默认使用了广播乘法
        但是这样转出来的onnx不被dnn支持
        所以我们干脆将self.anchor_w作为一个普通list求解，然后通过torch.mul()求解
        此外对切片进行赋值也会导致转onnx出错，因此我们使用cat完成这一过程。
        '''
        outputs = torch.cat(
            (
                torch.mul(X, self.stride).view(self.num_samples, 1, -1, 1),
                torch.mul(Y, self.stride).view(self.num_samples, 1, -1, 1),
                torch.mul(W, self.stride).view(self.num_samples, 1, -1, 1),
                torch.mul(H, self.stride).view(self.num_samples, 1, -1, 1),
                pred_conf.view(self.num_samples, 1, -1, 1),
                pred_cls.view(self.num_samples, 1, -1, self.num_classes),
            ),
            -1,
        )       # 沿着倒数第一个维度将上述三个矩阵进行拼接
        # inputs 维度为n*(3*85)*g*g, outputs是预测结果维度是n*(3*g*g)*85
        return outputs


class YOLOv3(nn.Module):
    '''
    YOLOv3 模型
    '''
    def __init__(self, config_path):
        super(YOLOv3, self).__init__()
        self.module_defs = parse_model_config(config_path)                              # 解析网络配置文件
        #  self.hyperparams是一个字典
        #  self.module_list是存放网络结构的列表，其中的元素都是每个网络层或者网络结构对象或者nn.Sequence()
        self.hyperparams, self.module_list = creat_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], YOLOLayer)]
        # 单独拿出Yolo层，yolo-tiny有两个yolo层
        self.img_size = int(self.hyperparams["width"])
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)


    def forward(self, x):
        layer_outputs, yolo_outputs = [], []
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        yolo_outputs_2 = FloatTensor()   # 转换后的输出，预测的位置
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            if module_def["type"] in ["convolutional", "upsample", "maxpool", "dropout"]:
                x = module(x)
            elif module_def["type"] == "route":         # 融合层，特征图拼接
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])       # 残差模块
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                out = module(x)
                yolo_outputs.append(out)
            layer_outputs.append(x)
            # 每次经过一个模块，其输出保存在layer_output中，方便随时访问中间层的输出，便于route和shotcut操作
            # layer_outputs并不占用额外的内存，因为append只是浅拷贝
        # yolo_outputs.append(yolo_outputs_2)
        # 如果有三个yolo层，yolo_outputs则有4个元素，前三个是yolo层不经过转换的输出，维度分别为n*255*13*13
        # n*255*26*26和n*255*52*52, 第四个元素是经过位置转换的yolo层输出的拼接，维度n*(3*13*13+3*26*26+3*52*52)*85
        # 如果输入图像大小是416*416，则最多预测3*13*13+3*26*26+3*52*52个目标
        # yolo_outputs = torch.cat(yolo_outputs, 1)
        return yolo_outputs


    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        # 记载darkenet格式的权重；darknet是一个开源框架，其权重文件后缀为.weight
        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()






if __name__ == "__main__":
    '''
    测试model.py
    '''
    from utils.utils import *

    conifg_path = "../configs/yolo-fastest-xl.cfg"
    net = YOLOv3(conifg_path)
    # print(net)

    net.apply(weights_init_normal)
    net.eval()
    inputs = torch.rand(1, 3, 320, 320)*255
    outputs = net(inputs)
    print(outputs[0].size())


    # onnx
    save_path = "./yolo-fastest-xl.onnx"
    torch.onnx.export(net, inputs, save_path, input_names=["input"], output_names=["out0", "out1"],
                      verbose=True, opset_version=11)
    # load onnx
    import onnx
    import torch
    import cv2
    onnx_name = "./yolo-fastest-xl.onnx"
    model = onnx.load(onnx_name)
    #检查IR是否良好
    onnx.checker.check_model(model)

    # 添加 ExpLayer
    class ExpLayer(object):
        def __init__(self, params, blobs):
            super(ExpLayer, self).__init__()

        def getMemoryShapes(self, inputs):
            return inputs

        def forward(self, inputs):
            return [np.exp(inputs[0])]
    cv2.dnn_registerLayer('Exp', ExpLayer)

    # opencv dnn加载
    import numpy as np
    net = cv2.dnn.readNetFromONNX(onnx_name)
    img = inputs.numpy()
    img = img[0]
    img = img.transpose((1,2,0))
    img = img.astype('uint8')
    blob = cv2.dnn.blobFromImage(img, size=(320, 320))      # img 必须是uint8
    net.setInput(blob)
    out = net.forward("out0")
    print(out)

    # 检查opencv加载onnx以后的结果和原来的结果是否相同
    outputs = outputs[0].detach().numpy()
    print(outputs[0] - out[0])





