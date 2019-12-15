from config import HOME
from calculate_finger import calculate_finger
import os
import cv2

root = HOME + "/test/test/"
res = []
target = []
ids = []
for _, dirs, _ in os.walk(root):
    for dir_c in dirs:
        for _, _, files in os.walk(root + dir_c):
            for name in files:
                # res.append(cv2.imread(root + dir_c + "/" + name))
                target.append(dir_c)
                ids.append(root + dir_c + "/" + name)

acc = 0
for i in range(len(ids)):
    if str(calculate_finger(ids[i])) == target[i]:
        acc = acc + 1
    else:
        print(
            "id %s, calculate is %d, actual %s "
            % (ids[i], calculate_finger(ids[i]), target[i])
        )

print("right %d in all %d, acc is %.2f" % (acc, len(ids), acc / len(ids)))
