import cv2  
import numpy as np  
import matplotlib.pyplot as plt

def move(img, width, height, x, y):
    #平移参数
    M = np.float32([[1, 0, x], [0, 1, y]])
    #图像平移
    change = cv2.warpAffine(img, M, (width, height))
    return change

def rotation(img, width, height, angle):
    #旋转参数
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    #旋转图像
    change = cv2.warpAffine(img, M, (width, height))
    return change

def mirror(img, flipCode):
    #图像镜像
    change = cv2.flip(img, flipCode)
    return change


#读取图像
img = cv2.imread("paimeng.jpg") 
#图像尺寸
height = img.shape[0]
width = img.shape[1]
# cv2.imshow("img", img)

#平移距离x，y
x = 100
y = 100
change1 = move(img, width, height, x, y)
change = move(img, width, height, x, y)
# cv2.imshow("change1", change1)

#旋转角度
angle = 30
change2 = rotation(img, width, height, angle)
change = rotation(change, width, height, angle)
# cv2.imshow("change2", change2)

#水平镜像1，垂直镜像0，同时-1
flipCode = -1
change3 = mirror(img, flipCode)
change = mirror(change, flipCode)
# cv2.imshow("change3", change3)

#显示图像
plt.subplot(221),plt.imshow(cv2.cvtColor(change1, cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(222),plt.imshow(cv2.cvtColor(change2, cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(223),plt.imshow(cv2.cvtColor(change3, cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.subplot(224),plt.imshow(cv2.cvtColor(change, cv2.COLOR_BGR2RGB)),plt.axis('off')
plt.show()