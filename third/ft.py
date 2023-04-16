import numpy as np
import cv2
from matplotlib import pyplot as plt

np.seterr(divide = 'ignore', invalid = 'ignore')
img = cv2.imread("paimeng.jpg")

def magnitude_phase_split(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    #幅度谱
    magnitude = np.abs(dft_shift)
    #相位谱
    phase = np.angle(dft_shift)
    return magnitude,phase

def magnitude_phase_combine(img_m, img_p):
    img_new = img_m * np.e ** (1j * img_p)
    img_new = np.uint8(np.abs(np.fft.ifft2(img_new)))
    img_new = img_new / np.max(img_new) * 255
    return img_new

img_m, img_p = magnitude_phase_split(img)

img1_m = np.abs(img_m)
img1_p = img_p

img2_m = np.abs(img_m)
img2_p = 0

img3_m = 150
img3_p = img_p

img1 = magnitude_phase_combine(img1_m, img1_p)
img2 = magnitude_phase_combine(img2_m, img2_p)
img3 = magnitude_phase_combine(img3_m, img3_p)

plt.figure(figsize=(10,7),facecolor='black',edgecolor='black')
plt.subplot(221),plt.imshow(img, 'gray'),plt.title('Original Image')
plt.axis('off')
plt.subplot(222,),plt.imshow(img1, 'gray'),plt.title('Method 1')
plt.axis('off')
plt.subplot(223),plt.imshow(img2, 'gray'),plt.title('Method 2')
plt.axis('off')
plt.subplot(224),plt.imshow(img3, 'gray'),plt.title('Method 3')
plt.axis('off')
plt.show()
