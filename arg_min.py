'''
numpy.argmin(a, axis=None, out=None)[source]

给出axis方向最小值的下标
返回：index_array : ndarray of ints下标组成的数组，shape与输入数组a去掉axis的维度相同。
返回axis轴向上最小值的索引构成的矩阵。

reference：https://blog.csdn.net/The_Time_Runner/article/details/89927488
'''
import numpy as np
a=np.arange(6).reshape(2,3)
print(a)
print(np.argmin(a))
print(np.argmin(a,axis=0))
print(np.argmin(a,axis=1))

b=np.arange(6)
b[4]=0
print(b)
print(np.argmin(b))
print(np.argmin(b,axis=0))

c=np.random.randint(24,size=[2,3,2])
print(c)
print(np.argmin(c,axis=0),'*')
print(np.argmin(c,axis=1),'@')
print(np.argmin(c,axis=2),'##')