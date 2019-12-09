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

frame = cv2.imread(HOME + "/mytest/0_0.jpg")
frame = cv2.resize(frame, (600, 800))
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

# for i, contour in enumerate(contours):# 获取轮廓
#     cv2.drawContours(frame, contours, i, (255, 0, 0), 2)# 绘制轮廓
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 0), 2)

ci = 0
maxArea = -1
for i in range(len(contours)):  # 找到最大的轮廓（根据面积）
    temp = contours[i]
    area = cv2.contourArea(temp)  #计算轮廓区域面积
    if area > maxArea:
        maxArea = area
        ci = i

res_c = contours[ci]  #得出最大的轮廓区域

hull = cv2.convexHull(res_c, True, returnPoints=True) # 获得凸包点 x, y坐标
# drawing = np.zeros(frame.shape, np.uint8)
cv2.drawContours(frame, [res_c], 0, (0, 255, 255), 2)   #画出最大区域轮廓
cv2.drawContours(frame, [hull], 0, (0, 100, 255), 3)  #画出凸包轮廓

moments = cv2.moments(res_c)  # 求最大区域轮廓的各阶矩
center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
cv2.circle(frame, center, 8, (0,0,255), 10)   #画出重心

point_res = []   #寻找一些局部最优解
max = 0
count = 0
notice = 0
print(len(res_c))
for i in range(len(res_c)):
    temp = res_c[i]
    dist = (temp[0][0] -center[0])**2 + (temp[0][1] -center[1])**2 #计算重心到轮廓边缘的距离
    if dist > max:
        max = dist
        notice = i
    if dist != max:
        count = count + 1
    if count > len(res_c)/2/5:
            count = 0
            max = 0
            flag = False   #布尔值
            if center[1] < res_c[notice][0][1]:   #低于手心的点不算
                continue                        #continue结束当前循环进入下一个循环
            for j in range(len(point_res)):  #离得太近的不算
                if abs(res_c[notice][0][0]-point_res[j][0]) < 20 :
                    flag = True
                    break
            if flag :
                continue
            point_res.append(res_c[notice][0])


cnt = 0
dist = _get_eucledian_distance(point_res[:], center)
cv2.circle(frame, (res_c[notice][0][0],res_c[notice][0][1]), 8 , (255, 255, 0), -1) #画出指尖

cv2.imshow("image" + str(cnt), frame)
cv2.waitKey(5000)
cv2.destroyAllWindows()

