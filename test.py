
# 颜色正态分布
import numpy as np
import seaborn as sns
x,y=np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
print(x)

if __name__=='__main__':
    # 随机生成100个符合正态分布的数
    x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
    # 颜色渐变 ： 深 =》 浅 ，as_cmap - 返回颜色字典，而不是颜色列表
    pal = sns.dark_palette("green",as_cmap=True)
    # x,y,颜色
    sns.kdeplot(x, y, cmap=pal)