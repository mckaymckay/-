''' 
颜色统计与分布曲线绘制

# 颜色分布跟正态分布很相似, 颜色直方图其实是多个正态分布图的累加
# reference:http://www.1zlab.com/wiki/python-opencv-tutorial/image-statistic-draw-curves/

'''

from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread('mai.jpg')

# 🚗 1.绘制灰度图的统计直方图
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.hist(gray.ravel(),bins=256,range=[0,256])

# 🚗2.绘制RGB彩图的统计直方图
# 创建画布
fig, ax = plt.subplots()

# Matplotlib预设的颜色字符
bgrColor = ('b', 'g', 'r')

# 统计窗口间隔 , 设置小了锯齿状较为明显 最小为1 最好可以被256整除
bin_win = 4
# 设定统计窗口bins的总数
bin_num = int(256 / bin_win)
# 控制画布的窗口x坐标的稀疏程度. 最密集就设定xticks_win=1
xticks_win = 2

for cidx, color in enumerate(bgrColor):
    # cidx channel 序号
    # color r / g / b
    cHist = cv2.calcHist([img], [cidx], None, [bin_num], [0, 256])
    # 绘制折线图
    ax.plot(cHist, color=color)

# 设定画布的范围
ax.set_xlim([0, bin_num])
# 设定x轴方向标注的位置
ax.set_xticks(np.arange(0, bin_num, xticks_win))
# 设定x轴方向标注的内容
ax.set_xticklabels(list(range(0, 256, bin_win * xticks_win)), rotation=45)

# 显示画面
plt.show()
