from config import HOME

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import datasets, transforms
from gesture import GestureDetection
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import math

def _get_eucledian_distance(start, end):
    return np.sqrt(np.sum(np.square(start[:] - end[:])))

frame = cv2.imread(HOME + "/test/1/28.jpg")

fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
fgmask = fgbg.apply(frame)
kernel = np.ones((5, 5), np.uint8)
fgmask = cv2.erode(fgmask, kernel, iterations=1) # 膨胀
res = cv2.bitwise_and(frame, frame, mask=fgmask)
ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
(_, cr, _) = cv2.split(ycrcb)
cr1 = cv2.GaussianBlur(cr, (5, 5), 0)# 高斯滤波
_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # OTSU图像二值化


skin ,contours,hierarchy = cv2.findContours(skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):# 获取轮廓
    cv2.drawContours(frame, contours, i, (255, 0, 0), 2)# 绘制轮廓
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 0), 2)


hull = cv2.convexHull(contour, True, returnPoints=False) # 获得凸包点 x, y坐标
defects = cv2.convexityDefects(contour, hull) # 计算轮廓的凹点
if defects is not None: # 重要!
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = contour[s][0] # 起点
        end = contour[e][0] # 终点
        far = contour[f][0] # 最远点

        a = _get_eucledian_distance(start, end)
        b = _get_eucledian_distance(start, far)
        c = _get_eucledian_distance(end, far)

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        cv2.line(frame, tuple(start), tuple(end), [255, 255, 0], 2)
        cv2.circle(frame, tuple(far), 5, [0, 0, 255], -1)

cv2.imshow("image", frame)
cv2.waitKey(10000)
cv2.destroyAllWindows()

