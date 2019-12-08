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
import os.path as osp
import numpy as np


data_tf = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

labels = [0, 1, 2, 3, 4, 5]

# rgb_image = data_tf(Image.fromarray(rgb_image))
# rgb_image = transforms.ToPILImage()(rgb_image)
# rgb_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_BGR2GRAY)
# Image.fromarray(rgb_image).show()


class GestureTest(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._annopath = osp.join("%s" % self.root, "Annotation", "%s.txt")
        self._imgpath = osp.join("%s" % self.root, "Image", "%s.jpg")
        self.ids, self.data = self.read_data(root)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        if img is None:
            img = cv2.imread(self.ids[index])
            print(self.ids[index])
            print("===============================\n")
        if img is None:
            print(self.ids[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img).convert('RGB')
        img = transforms.CenterCrop(min(img.height, img.width))(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.ids[index]

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
        ids = []
        for _, _, files in os.walk(root):
            for name in files:
                ids.append(root + "/" + name)
                res.append(self.read_img_with_skin(root + "/" + name))
        return ids, res


batch_size = 1
test_dataset = GestureTest(HOME + "/mytest", transform=data_tf)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = torch.load(HOME + "gesture-cnn_skin_gray_5.pth")
print("len is %s" % (len(test_dataset)))
# 模型评估
model.eval()

criterion = nn.CrossEntropyLoss()
for data in test_loader:
    img, name = data
    # img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
    else:
        img = Variable(img)
    out = model(img)
    _, pred = torch.max(out, 1)
    print("image name %s 可能是 %s " % (name, pred.item()))

# img = Image.fromarray(cv2.imread(HOME + "/mytest/3.jpg"))
# img = data_tf(img)
# img = Variable(img)
# if torch.cuda.is_available():
#     img = img.cuda()
# out = model(img)
# print("image name {%s} 可能是 {%s}".format(out))
