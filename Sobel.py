import numpy as np
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt

original = 'rose.jpeg'
img = np.array(Image.open(original).convert("L"))  #打开图像并转化为灰度矩阵
# print(img)

# sobel算子
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#convolve2d()第一个参数是原图像矩阵，第二个参数为卷积算子，指定边界样式为boundary=‘symm’，
# 然后指定关键字参数mode=“same”(输出矩阵大小和原图像矩阵相同)。

# 计算x方向的卷积
img1_x = signal.convolve2d(img, sobel_x, boundary='symm', mode='same')
# 计算y方向的卷积
img1_y = signal.convolve2d(img, sobel_y, boundary='symm', mode='same')
#得到提督矩阵
img1_xy = np.sqrt(img1_x**2 + img1_y**2)
# 梯度矩阵归一到0-255
img1_xy = img1_xy * (255 / img.max())

#prewitt算子计算过程与sobel算子一样
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
img2_x = signal.convolve2d(img, prewitt_x, boundary='symm', mode="same")
img2_y = signal.convolve2d(img, prewitt_y, boundary='symm', mode="same")
img2_xy = np.sqrt(img2_x**2 + img2_y**2)
img2_xy = img2_xy * (255 / img2_xy.max())

# 绘制出图像
plt.subplot(1, 2, 1)
plt.imshow(np.absolute(img1_xy), cmap='gray_r')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np.absolute(img1_xy), cmap='gray_r')
plt.axis('off')
plt.show()