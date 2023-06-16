import cv2  
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('paimeng.jpg')

#图像转为灰度图
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#获取图像高度和宽度
height = grayImage.shape[0]
width = grayImage.shape[1]

#创建图像
result = np.zeros((height, width, 3), np.uint8)

#图像灰度线性变换
# for i in range(height):
#     for j in range(width):
#         point = 255 - grayImage[i,j] 
#         result[i,j] = np.uint8(point)

#图像灰度分段线性变换
# for i in range(height):
#     for j in range(width):
#         if (0 <= grayImage[i,j] < 100):
#             point = grayImage[i,j] + 30
#         elif (100 <= grayImage[i,j] < 150):
#             point = grayImage[i,j] + 10
#         else:
#             point = grayImage[i,j] - 30
#         result[i,j] = np.uint8(point)

#图像灰度非线性变换(开方)
# for i in range(height):
#     for j in range(width):
#         point = int(np.sqrt(grayImage[i,j])) + 200
#         result[i,j] = np.uint8(point)

#图像灰度非线性变换(sgn)
for i in range(height):
    for j in range(width):
        if (0 <= grayImage[i,j] < 100):
            point = 0
        elif (100 <= grayImage[i,j] < 150):
            point = 125
        else:
            point = 255
        result[i,j] = np.uint8(point)

#显示图像
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.show()