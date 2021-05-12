'''
python各类图像库的图像读写方式总结：
reference:https://www.cnblogs.com/skyfsm/p/8276501.html

opencv
PIL(pillow)
matplotlib.image
scipy.misc
skimage

'''
import cv2
import numpy as np
'''1.opencv
值得注意,opencv读进来的图片是一个numpy矩阵,彩色图片维度是(高度，宽度，通道数),数据类型是uint8
'''
# 1.读入图片：默认彩色图，cv2.IMREAD_GRAYSCALE灰度图，cv2.IMREAD_UNCHANGED包含alpha通道

cv_img = cv2.imread('mai.jpg')
# cv2.imshow('opencv', cv_img)
# print(cv_img.shape)  # (height,weight,channel)
# print(cv_img.size)  # 像素总数目=h*w*c
# print(cv_img.dtype) # unit8
# print(cv_img)

# 2.先读入彩色图，再转灰度图
cv_img2 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('cv_img2', cv_img2)

# 3.图片矩阵转换
print(cv_img.shape)  # h,w,c
cv_img3 = cv_img.transpose(2, 0, 1)
print(cv_img.shape)  # c,h,w

# 4.图片归一化
cv_img4 = cv_img.astype('float') / 255.0
print(cv_img4.dtype)  # float64
# print(cv_img4)

# 5.存储图片
# 得到的是全黑的图片，因为之前归一化了
# cv2.imwrite('img.jpg',cv_img4)
cv_img5 = cv_img4 * 255
# cv2.imwrite('cv_img5.jpg', cv_img5)

# 6.大坑之BGR
cv_img6 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
# cv2.imshow('', cv_img6)  # 变了变了

# 7.访问像素
print(cv_img[200, 200])
print(cv_img2[200, 200])
cv_img[200, 200] = [255, 255, 255]
print(cv_img[200, 200])

# ROI操作
cv_img8 = cv_img[100:200, 200:400]
cv2.imshow('roi',cv_img8)
'''2.'''

cv2.waitKey()