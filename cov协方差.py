'''

reference:
https://www.cnblogs.com/pengkunfan/p/3947529.html
https://zhuanlan.zhihu.com/p/45140262
1.
NumPy函数的正态分布np.random.normal()
该函数需要平均值，标准差和分布的观察数作为输入

'''
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# 计算协方差矩阵
def plotDataAndCov(data):
    # print(A.shape)  #(300,2)
    ACov = np.cov(data, rowvar=False, bias=True)
    print('Covariance matrix:\n', ACov)  #(2,2)

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 10)

    ax0 = plt.subplot(1, 2, 1)

    # Choosing the colors
    cmap = sns.color_palette("GnBu", 10)
    sns.heatmap(ACov, cmap=cmap, vmin=0)

    ax1 = plt.subplot(1, 2, 2)

    # data can include the colors
    if data.shape[1] == 3:
        c = data[:, 2]
    else:
        c = "#0A98BE"
    ax1.scatter(data[:, 0], data[:, 1], c=c, s=40)

    # Remove the top and right axes from the data plot
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)


'''
 模拟数据
# 1.不相关的数据,a1 a2两个矢量标准差为1,第一个mean=1,第二个mean=2,是不相关的 
'''
a1 = np.random.normal(2, 1, 300)
a2 = np.random.normal(1, 1, 300)
A = np.array([a1, a2]).T  #转置后：（300，2）
# print(A[:10,:])
# sns.distplot(A[:,0], color="#53BB04")
# sns.distplot(A[:,1], color="#0A98BE")
# plotDataAndCov(A)
'''
# 2.相关数据
在散点图上可以看到两个维度之间的相关性。我们可以看到可以绘制一条线并用于从x预测y，反之亦然。
'''
np.random.seed(1234)
b1 = np.random.normal(3, 1, 300)
b2 = b1 + np.random.normal(7, 1, 300) / 2.
B = np.array([b1, b2]).T
# plotDataAndCov(B)
# plt.show()
# plt.close()
'''
预处理:
其中X'是标准化的数据集, X是原始数据集, x' 是平均值, 并且σ是标准偏差.

A:平均归一化:减去平均值
B:标准化或归一化:标准化用于将所有特征放在相同的比例上

'''


# A:X'=(X-x')
def center(X):
    newX = X - np.mean(X, axis=0)
    return newX

B_centered = center(B)
print('before:\n\n')
plotDataAndCov(B)
plt.show()
plt.close()
print('after:\n\n')
# plotDataAndCov(B_centered)
# plt.show()
# plt.close()


# B:X'=(X-x')/σ
def standardize(X):
    newX = center(X) / np.std(X, axis=0)
    return newX
np.random.seed(1234) 
c1 = np.random.normal(3, 1, 300) 
c2 = c1 + np.random.normal(7, 5, 300)/2. 
C = np.array([c1, c2]).T 
plotDataAndCov(C) 
plt.xlim(0, 15) 
plt.ylim(0, 15) 
plt.show() 
plt.close()
# 可以看到x和y的尺度不同。另请注意，由于比例差异，相关性似乎较小。现在让我们标准化它
CStandardized = standardize(C) 
plotDataAndCov(CStandardized) 
plt.show() 
plt.close()
# 可以看到比例相同，并且数据集根据两个轴以零为中心。
# 现在，看一下协方差矩阵。您可以看到每个坐标的方差 - 左上角单元格和右下角单元格 - 等于1。
# 这个新的协方差矩阵实际上是相关矩阵。两个变量（c1和c2）之间的Pearson相关系数是0.54220151