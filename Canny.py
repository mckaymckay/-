import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img = 'rose.jpeg'
img = cv.imread(img, 0)  #以灰度值读入图像

#高斯平滑
blur = cv.GaussianBlur(img, (5, 5), 0)  #参数可调
#canny函数
edges = cv.Canny(blur, 60, 180)  #参数可调

#绘制图像
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('original'), plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray_r')
plt.title('edges'), plt.axis("off")
plt.show()
