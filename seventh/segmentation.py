import cv2
from matplotlib import pyplot as plt
import numpy as np

def diedai(img):
    img_array = np.array(img).astype(np.float32)#转化成数组
    I=img_array
    zmax=np.max(I)
    zmin=np.min(I)
    tk=(zmax+zmin)/2#设置初始阈值
    #根据阈值将图像进行分割为前景和背景，分别求出两者的平均灰度  zo和zb
    b=1
    m,n=I.shape
    while b==0:
        ifg=0
        ibg=0
        fnum=0
        bnum=0
        for i in range(1,m):
             for j in range(1,n):
                tmp=I(i,j)
                if tmp>=tk:
                    ifg=ifg+1
                    fnum=fnum+int(tmp)  
                else:
                    ibg=ibg+1
                    bnum=bnum+int(tmp)
        #计算前景和背景的平均值
        zo=int(fnum/ifg)
        zb=int(bnum/ibg)
        if tk==int((zo+zb)/2):
            b=0
        else:
            tk=int((zo+zb)/2)
    return tk


img = cv2.imread('paimeng.jpg', 0)
 
# Otsu法
ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 迭代法
img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)
img_1 = cv2.resize(gray,(200,200))

yvzhi=diedai(img_1)
ret1, th1 = cv2.threshold(img, yvzhi, 255, cv2.THRESH_BINARY)

# 显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
plt.title('原图'), plt.axis('off')
plt.subplot(132), plt.imshow(cv2.cvtColor(otsu, cv2.COLOR_BGR2RGB), cmap='gray'),
plt.title('大津法分割'), plt.axis('off')
plt.subplot(133), plt.imshow(th1,cmap='gray'),
plt.title('迭代法分割'), plt.axis('off')
plt.show()    
