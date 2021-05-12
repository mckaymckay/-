import cv2
import matplotlib.pyplot as plt
import numpy as np
'''
1.加载需要的库和图像并显示
'''
path = 'mai.jpg'
img = cv2.imread(path).astype(np.float32)/255
cv2.imshow('image', img)


'''
2.对图像的矩阵减平均值
'''
# 通过img.copy()方法，复制img的数据到mean_img
mean_img=img.copy()
print(mean_img.mean())

# 得出零平均值矩阵
mean_img -= mean_img.mean()
cv2.imshow('mean_img',mean_img)

'''
3，再对图像矩阵除于标准差
'''
std_img = mean_img.copy()
# 标准差
print(std_img.std())

# 零均值矩阵除于标准差，得出单位方差矩阵
std_img /= std_img.std()
cv2.imshow('std_img',std_img)


cv2.waitKey()