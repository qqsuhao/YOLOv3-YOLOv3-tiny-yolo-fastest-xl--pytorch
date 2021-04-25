# YOLOv3/YOLOv3-tiny/yolo-fastest-xl/-pytorch

## 仓库介绍

本仓库旨在实现YOLOv3/YOLOv3-tiny/yolo-fasetest-xl这三种版本的网络从训练，到评估，再到导出为onnx并使用opencv进行部署的全套流程。

## 对应博客

[https://blog.csdn.net/qq7835144/article/details/115112748](https://blog.csdn.net/qq7835144/article/details/115112748)

## 参考代码：

[https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

[https://github.com/BobLiu20/YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch)

[https://github.com/dog-qiuqiu/Yolo-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)

[https://github.com/muvanpersie/yolo-fastest-pytorch](https://github.com/muvanpersie/yolo-fastest-pytorch)

[https://github.com/bubbliiiing/yolo3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch)

## 仍然存在的问题

时至今日我无法在coco2014上复现出与https://github.com/eriklindernoren/PyTorch-YOLOv3相同的map结果，尽管大部分代码都是参考这个仓库的。此外这个仓库貌似不是很稳定，至今仍然在进行修改，我也没有精力把他每次修改以后的代码重新跑一遍。

为了解决无法复现结果的问题，我参考了大量其他的代码，主要是loss求解部分的代码。但是仍然没有复现出很高的map。我并没有精力把每个仓库的代码都跑一次，查看是否可以在coco2014上复现出55%左右的map，只是对比了他们的loss求解部分。

我不确定其他仓库的代码是否借鉴了https://github.com/eriklindernoren/PyTorch-YOLOv3，但是貌似目前yolov3的代码在求解loss方面，要么都是借鉴https://github.com/eriklindernoren/PyTorch-YOLOv3，要么都是借鉴https://github.com/ultralytics/yolov3。并且eriklindernoren最近更新的代码也开始借鉴ultralytics，因为eriklindernoren的代码风格与ultralytics极其相似。eriklindernoren抛弃了之前的loss求解的代码，转而使用ultralytics求解loss的方法。但是严格意义上eriklindernoren抛弃了之前的loss求解的代码才是YOLOv3的loss求解方法，而ultralytics求解loss的方法实际上属于YOLOv5的loss求解方法。

我在代码里添加了三个版本的model和loss。第一个版本是eriklindernoren早期的loss求解方法，也是网络上绝大多数博主或者仓库使用的loss求解方法。该版本命名为modelv3.py和lossv3.py。但是我在使用这个版本的代码进行训练时，得到的评估结果很糟糕，远不及eriklindernoren得到的55%的map。第二个版本是参考ultralytics的loss求解方法，该版本命名为modelv5.py和lossv5.py。实际上模型还是YOLOv3，只是loss求解是YOLOv5的，只是为了名称整齐，将文件命名为modelv5.py。eriklindernoren最新的代码和https://github.com/muvanpersie/yolo-fastest-pytorch应该都是参考的ultralytics，因为他们的代码风格，变量命名极其相似。我在这个版本的代码上进行训练得到了比较好的结果。如果测试时将置信度(thre_conf)调整至0.5，我训练出来的YOLOv3-tiny可以达到与官方权值文件相同的map，大约是15%左右。但是如果把置信度(thre_conf)调整至0.001，我的模型只能达到19%map，而官方权值却可以达到30%map。不过能达到这个程度我已经很开心了。

前两个版本的loss求解代码主要有两个区别，一个是loss求解时使用的损失函数不一样；另一个是求解xywh的偏置的方法不一样。modelv3.py和lossv3.py使用如下方法求解偏置：

```python
x = torch.sigmoid(inputs_view[..., 0])  # Center x
y = torch.sigmoid(inputs_view[..., 1])  # Center y
w = inputs_view[..., 2]  # Width
h = inputs_view[..., 3]  # Height
x = x + grid_x
y = y + grid_y
w = torch.exp(w) * anchors_w
h = torch.exp(h) * anchors_h
```

modelv5.py和lossv5.py使用如下方法求解偏置：

```python
x = torch.sigmoid(inputs_view[..., 0])  # Center x
y = torch.sigmoid(inputs_view[..., 1])  # Center y
w = torch.sigmoid(inputs_view[..., 2])  # Width
h = torch.sigmoid(inputs_view[..., 3])  # Height
x = x * 2.0 - 0.5 + grid_x
y = y * 2.0 - 0.5 + grid_y
w = (w * 2.0) ** 2 * anchors_w
h = (h * 2.0) ** 2 * anchors_h
```

为了搞清楚到底是哪一个区别导致modelv5.py和lossv5.py 与 modelv3.py和lossv3.py这个两个版本的代码差异如此巨大，创建了model35.py和loss35.py，即第三个版本的代码。在这个版本中，使用modelv5.py和lossv5.py 的偏执求解方法+使用modelv3.py和lossv3.py的损失函数。测试以后发现，这个版本的代码与第一个版本的代码表现相似，都很差。所以应该是损失函数差异导致第二个版本的代码比第一个版本好。



**综上，目前modelv5.py和lossv5.py这个版本的代码可以达到与官方权值文件相似的性能，但是仍然不是很完美，因为lossv5.py使用的loss实际上已经是YOLOv5的方法了。**



## 代码结构

|---- cache	           # 设置的缓存文件，后来没有使用，空文件夹，暂不删除

|---- checkpoints	# 存放训练过程中的权值文件，防止因意外导致训练中断丢失训练结果

|---- configs 		# 存放各种模型的cfg文件

|---- data		      # 存放数据集的相关文件以及数据集。

|---- logs			# 存放tensorboard记录的日志，方便观察各种指标在训练过程中的变化

​	|---- tiny-loss35	# 使用第三个版本的代码训练yolov3-tiny的日志

​	|---- tiny-lossv5	# t 使用第二个版本的代码训练yolov3-tiny的日志

|---- models		# 存放模型构建文件和损失函数求解文件

​	|---- model\*.py	\# 其中每个model\*.py可以单独运行以测试是否可以成功转出onnx并被opencv成功加载

​	|---- loss\*.py	# 对应model文件求解loss的代码

|---- ouputs		# 使用data/sample文件夹中的图片进行测试，并将结果保存在outputs中

|---- utils		# 存放一些YOLOv3需要使用到的额外的功能。

​	|---- augmentations.py			# 图像增强

​	|---- datasets.py					# 加载数据集

​	|---- logger.py			# 创建记录日志的对象

​	|---- parse_config.py		# 解析.cgf配置文件和.data配置文件

​	|---- plot.py		# 绘制样本的函数，便于把训练样本画出来，看一看增强后的数据是个什么样子

​	|---- transforms.py		# 加载数据集是使用的变换，与augmentations.py配套

​	|---- utils.py		# 存放其他函数，比如求解IOU，NMS。

|---- wieghts	# 存放权值文件

|---- cal_anchors.py	#	使用skleran的聚类求解数据集的anchors

|---- detect.py		# 使用训练好的模型对样本图像进行目标检测

|---- test.py		# 对训练好的模型进行评估

|---- Toonnx.py		# 将训练好的模型转换为onnx格式

|---- train.py		# 训练模型

----

## 训练策略

参考https://github.com/muvanpersie/yolo-fastest-pytorch，在代码中使用了混合精度训练+学习率阶段下降+多GPU并行训练。

## 自己的数据从训练到部署

- 因项目涉及，我搜集了一些菌落的照片并使用labelImg工具进行标注。分别将图片和标注文件存放在data/bac/images和data/bac/labels。然后在data文件夹里创建bac.data和bac.names文件。
- 需要自己生成一个txt文件存放对应数据集样本的存放路径。例如训练数据集存放在data/bac/images中，就需要生成一个txt文件（这一部分的代码我没有写成py脚本，需要读者自己实现这个功能了），存放所有样本的路径，例如train.txt

```
data/bac/images/0.jpg
data/bac/images/1.jpg
data/bac/images/10.jpg
data/bac/images/101.jpg
```

- bac.data文件里主要是一些路径设置，如下

```
classes= 1
train=data/bac/train.txt	# 只需要指定上述生成的txt文件的路径即可
valid=data/bac/train.txt	# 因为数据很少，就使用训练作为验证集，读者使用的时候可替换
names=data/bac.names
backup=backup/			# 不重要
eval=bac				# 不重要
```

- bac.names 文件主要是类别名称，我只做目标检测，不做分类，所以只有一类，名字设为0。**注意bac.names写完以后一定要在结尾回车空出一行，否则代码读不到文件全部内容**。

```
0

```

- 计算anchor。在新的数据集上需要重新计算anchors，使用cal_anchors.py计算anchros。
- 修改对应模型的cfg文件中yolo层的anchors。由于菌落的anchors都很小，所以我稍微修改了yolov3-tiny的结构，将其命名为**yolov3-tiny-bac.cfg**。
- 修改train.py中的相关路径和参数即可。
- 同理test.py和detect.py也是修改相关路径和参数即可。
- 菌落检测结果（因为训练集和测试集使用相同的样本，效果肯定好）

![output/142.png](output\142.png)

- 我自己修改后的模型yolov3-tiny-bac.cfg权重只有8M左右，使用Toonnx.p将模型转换为onnx格式。并测试使用opencv的DNN模块能否正确加载
- 参考[https://github.com/qqsuhao/yolo-fastest-xl-based-on-opencv-DNN-using-onnx](https://github.com/qqsuhao/yolo-fastest-xl-based-on-opencv-DNN-using-onnx)使用C++部署模型
- 我的github主页后续会放置部署在树莓派上的代码。

## 权重文件下载

链接：https://pan.baidu.com/s/1CYVyUCMUGHUO_Xba5ZNaxQ 
提取码：1234 
复制这段内容后打开百度网盘手机App，操作更方便哦