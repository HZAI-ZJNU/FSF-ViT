"""
original code from WZMIAOMIAO:
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer
"""
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        w,h=img.size
        area = w*h
        img_numpy_from_PIL=np.array(img)
        img_2=img_numpy_from_PIL[int(h/4):int(h/4)+int(h/2),int(w/4):int(w/4)+int(w/2),:]


        sl=0.02
        sh=0.1
        r1=0.3
        mean = [0.4914, 0.4822, 0.4465]
        block_4 = []
        for i in range(4):
            while True:
                target_area = np.random.uniform(sl, sh) * area
                aspect_ratio = np.random.uniform(r1, 1 / r1)
                block_h = int(round(np.math.sqrt(target_area * aspect_ratio)))
                block_w = int(round(np.math.sqrt(target_area / aspect_ratio)))
                if block_h<h and block_w<w:
                    block_4.append([block_h,block_w])
                    break

        img_3=img_numpy_from_PIL.copy()
        decision_array = np.random.randint(0, 2, size=4)
        if decision_array[0] == 1:
            img_3[0:block_4[0][0], 0:block_4[0][1], :] = mean
        if decision_array[1] == 1:
            img_3[0:block_4[1][0], w - block_4[1][1]:, :] = mean
        if decision_array[2] == 1:
            img_3[h - block_4[2][0]:, 0:block_4[2][1], :] = mean
        if decision_array[3] == 1:
            img_3[h - block_4[3][0]:, w - block_4[3][1]:, :] = mean

        #变换图片转换为PIL
        img_2=Image.fromarray(img_2)
        img_3=Image.fromarray(img_3)

        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
            img_2=self.transform(img_2)
            img_3=self.transform(img_3)
        img=torch.stack([img,img_2,img_3],dim=0)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images =[t for t in images]
        images=torch.cat(images,dim=0)
        labels = torch.as_tensor(labels)

        labels=labels.repeat_interleave(3)

        return images, labels
