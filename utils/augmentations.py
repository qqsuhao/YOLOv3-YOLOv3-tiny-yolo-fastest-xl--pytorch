import imgaug.augmenters as iaa
from .transforms import *

class DefaultAug(ImgAug):
    '''
    下面的这些函数都可以看源码，有详细的参数解释
    '''
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),       # 随机去掉一些像素点, 即把这些像素点变成0。
            iaa.Sharpen((0.0, 0.2)),        # 锐化
            iaa.Affine(rotate=(-20, 20), translate_percent=(-0.2,0.2)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.AddToBrightness((-30, 30)),     # change brightness of images
            iaa.AddToHue((-20, 20)),            # 为图像的色调添加随机值。
            iaa.Fliplr(0.5),    # 水平镜面翻转。
        ], random_order=True)


AUGMENTATION_TRANSFORMS = transforms.Compose([
        AbsoluteLabels(),
        DefaultAug(),
        PadSquare(),
        RelativeLabels(),
        ToTensor(),
    ])
