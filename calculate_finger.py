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


def _get_eucledian_distance__tuple(start, end):
    return np.sqrt(np.sum(np.square(np.array(start) - np.array(end))))


def __get_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    if angle2 > 180:
        angle2 = 360 - angle2
    return angle2


def calculate_finger(path):
    frame = cv2.imread(path)
    [o_height, o_width, _] = frame.shape
    height = 512
    width = int(o_width * height / o_height)
    frame = cv2.resize(frame, (width, height))
    fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)  # 膨胀
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    ycrcb = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
    _, skin = cv2.threshold(
        cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )  # OTSU图像二值化

    skin, contours, _ = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ci = 0
    maxArea = -1
    for i in range(len(contours)):  # 找到最大的轮廓（根据面积）
        temp = contours[i]
        area = cv2.contourArea(temp)  # 计算轮廓区域面积
        if area > maxArea:
            maxArea = area
            ci = i

    res_c = contours[ci]  # 得出最大的轮廓区域

    hull = cv2.convexHull(res_c, True, returnPoints=True)  # 获得凸包点 x, y坐标
    cv2.drawContours(frame, [res_c], 0, (0, 255, 255), 2)  # 画出最大区域轮廓

    moments = cv2.moments(res_c)  # 求最大区域轮廓的各阶矩
    hand_center = (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )
    cv2.circle(frame, hand_center, 8, (0, 0, 255), 10)  # 画出重心

    hull2 = cv2.convexHull(res_c, True, returnPoints=False)
    defects = cv2.convexityDefects(res_c, hull2)
    # dist = []

    r = 0xFFFF
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = tuple(res_c[s][0])
        end = tuple(res_c[e][0])
        far = tuple(res_c[f][0])
        r = min(r, depth)
        cv2.line(frame, start, end, [0, 255, 0], 2)

    between_fingers_point = []
    before = []  # j points before between
    after = []  # j points after between
    half_section_size = int(res_c.shape[0] / 20)
    # 选出可能是手指的点和前后half个
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        start = tuple(res_c[s][0])
        end = tuple(res_c[e][0])
        far = tuple(res_c[f][0])
        maybe = False
        AB = _get_eucledian_distance__tuple(start, far)
        AC = _get_eucledian_distance__tuple(start, hand_center)
        BC = _get_eucledian_distance__tuple(hand_center, far)
        BD = _get_eucledian_distance__tuple(far, end)
        AD = _get_eucledian_distance__tuple(start, end)

        maybe = False
        if AC > BC and BC >= 0.1 * r and BC <= 1.3 * r:
            maybe = True
        if AC <= BC and AC >= 0.1 * r and AC <= 1.3 * r:
            maybe = True
        if min(BC, AC) / max(BC, AC) <= 0.8:
            maybe = True
        if max(AB, BD) / min(AD, BD) >= 0.8:
            maybe = True

        if maybe:
            between_fingers_point.append(start)
            list_t = []
            for j in range(s - 3, s - half_section_size - 3, -3):
                list_t.append(res_c[j][0])
            before.append(list_t[:])
            list_t = []
            for j in range(s + 3, (s + half_section_size + 3) % res_c.shape[0], 3):
                list_t.append(res_c[j][0])
            after.append(list_t[:])

    # 角度小于等于60 认为是手指
    finger = []
    for i in range(0, len(between_fingers_point)):
        min_angle = 180
        for j in range(0, min(len(before[i]), len(after[i]))):
            min_angle = min(
                min_angle,
                __get_angle(
                    np.array(before[i][j] - between_fingers_point[i]),
                    np.array(after[i][j] - between_fingers_point[i]),
                ),
            )
        if min_angle <= 60:
            finger.append(between_fingers_point[i])

    # 排序，去除距离太近的点和边界点
    result = []
    last = None
    finger = sorted(finger, key=lambda x: finger[0])
    for i in range(0, len(finger)):
        print(finger[i])
        if (
            finger[i][0] >= width - half_section_size
            or finger[i][1] >= height - half_section_size
        ):
            continue
        if (
            last is not None
            and _get_eucledian_distance__tuple(finger[i], last) < half_section_size
        ):
            continue
        result.append(finger[i])
        last = finger[i]
        cv2.circle(frame, finger[i], 5, [190, 17, 200], -1)


# cv2.imshow("image" + str(len(result)), frame)
# cv2.waitKey(7000)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    calculate_finger(HOME + "/test/test/0/0.jpg")
