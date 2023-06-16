import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('paimeng.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret0, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# 确定背景区域
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# 确定前景区域
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret1, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# 查找未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# 标记标签
ret2, markers1 = cv2.connectedComponents(sure_fg)
markers = markers1+1
markers[unknown==255] = 0

markers3 = cv2.watershed(img,markers)
img[markers3 == -1] = [0,255,0]

# 显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('分水岭'), plt.axis('off')
plt.show()
