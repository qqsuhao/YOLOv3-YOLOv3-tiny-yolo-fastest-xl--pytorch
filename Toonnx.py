# -*- coding:utf8 -*-
# @TIME     : 2021/3/21 21:59
# @Author   : SuHao
# @File     : Toonnx.py


from models.modelv5 import YOLOv3
import onnx
import torch
import cv2
import numpy as np

conifg_path = "./configs/yolov3-tiny-bac.cfg"
weights_path = "./weights/tiny-bac.pth"
save_path = "./weights/tiny-bac.onnx"


net = YOLOv3(conifg_path)
# If specified we start from checkpoint
if weights_path:
    if weights_path.endswith(".pth"):
        net.load_state_dict(torch.load(weights_path))       # 加载pytorch格式的权重文件
        print("load_state_dict")
    else:
        net.load_darknet_weights(weights_path)          # 加载darknet格式的权重文件。以.weight为后缀
        print("load_darknet_weights")

net.eval()
inputs = torch.rand(1, 3, 320, 320)
torch.onnx.export(net, inputs, save_path, input_names=["input"], output_names=["outputs0", "outputs1"],
                  verbose=True, opset_version=11)
model = onnx.load(save_path)
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
net = cv2.dnn.readNetFromONNX(save_path)
img = inputs.numpy() * 255
img = img[0]
img = img.transpose((1, 2, 0))
img = img.astype('uint8')
blob = cv2.dnn.blobFromImage(img, size=(320, 320))      # img 必须是uint8
print(blob.shape)
net.setInput(blob)
out_blob = net.forward(net.getUnconnectedOutLayersNames())
print(out_blob[1].shape)

out = cv2.dnn.imagesFromBlob(out_blob[1])
print(out[0].shape)







