import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import re
from PIL import Image
from torchvision import datasets, transforms
from config import HOME

GESTURE_CLASSES = (0, 1, 2, 3, 4, 5)


class GestureDetection(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join("%s" % self.root, "Annotation", "%s.txt")
        self._imgpath = osp.join("%s" % self.root, "Image", "%s.jpg")
        self.ids, self.data, self.targets = self.read_data(root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        if img is None:
            img = cv2.imread(self.ids[index])
            print(self.ids[index])
            print("===============================\n")
        if img is None:
            print(self.ids[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = transforms.CenterCrop(min(img.height, img.width))(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def read_img_with_skin(self, path):
        frame = cv2.imread(path)
        frame = transforms.Resize(224)(Image.fromarray(frame))
        frame = np.array(frame)
        fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
        fgmask = fgbg.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1) # 膨胀
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
        (_, cr, _) = cv2.split(ycrcb)
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0)# 高斯滤波
        _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU图像二值化
        return skin

    def read_data(self, root):
        res = []
        target = []
        ids = []
        for _, dirs, _ in os.walk(root):
            for dir_c in dirs:
                for _, _, files in os.walk(root + dir_c):
                    for name in files:
                        res.append(cv2.imread(root + dir_c + "/" + name))
                        target.append(dir_c)
                        ids.append(self.read_img_with_skin(root + dir_c + "/" + name))
        return ids, res, target
