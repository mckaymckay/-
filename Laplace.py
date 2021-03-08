import numpy as np
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt

original = 'rose.jpeg'
img = np.array(Image.open(original).convert("L"))  #打开图像并转化为灰度矩阵


# 一般在进行Laplace运算之前，我们会先对图像进行先对图像进行模糊平滑处理，目的是去除图像中的高频噪声。

#定义高斯函数
def func(x,y,sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))


#高斯平滑处理
Gaussian = np.fromfunction(func,(3,3),sigma=3)
img = signal.convolve2d(img,Gaussian,mode="same")


#laplace算子
laplace = np.array([[0,1,0],
                    [1,-4,1],
                   [0,1,0]])
#laplace扩展算子
laplace2 = np.array([[1,1,1],
                    [1,-8,1],
                   [1,1,1]])
# Laplace算子是一个二阶导数的算子，它实际上是一个x方向二阶导数和y方向二阶导数的和的近似求导算子。

# 卷积
img_l1 = signal.convolve2d(img,laplace,boundary='symm',mode="same")
img_l2 = signal.convolve2d(img,laplace2,boundary='symm',mode="same")

# 将卷积结果转化成0~255
img_l1=(img_l1/float(img_l1.max()))*255
img_l2=(img_l2/float(img_l2.max()))*255


plt.subplot(1,2,1)
plt.imshow(np.absolute(img_l1),cmap='gray_r')
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(np.absolute(img_l2),cmap='gray_r')
plt.axis("off")
plt.show()
