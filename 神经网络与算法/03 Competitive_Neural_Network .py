#!/usr/bin/env python
# coding: utf-8

# # Chapter4 竞争学习神经网络
# 
# >姓名：潘冯谱(2018级本科)
# >
# >学校：华东理工大学
# >
# >专业：数学与应用数学
# 
# 修改于2020-11-09

# ## 习题4.7
# 
# 竞争学习中采用的是典型学习规则 Winner-Take-All，该算法分为三步骤：
# - 向量归一化
# - 寻找获胜神经元（点积法测量相似性）$$W(t)^TX^p = \max_{j\in\{1,2,\cdots,m\}}\{W_j(t)^TX^p\}$$
# - 网络输出与权值调整$$W(t+1) = W(t) + \eta(t,n)[X^p - W(t)]$$
# 
# ### 编程目标：
# 
# 采用胜者为王学习算法训练一个竞争网络，将下面的输入模式分为两类：$$X^1=(1,-1)^T，X^2=(1,1)^T，X^3=(-1,-1)^T$$.
# - 1) $\eta=0.5$，初始权值矩阵为：$$W=\begin{bmatrix}\sqrt{2}&0\\0&\sqrt{2}\end{bmatrix}$$将输入模式按顺序训练一遍并图示训练结果，观察模式如何聚类。
# - 2) 如果输入模式的顺序改变，训练结果是否改变？请解释原因。
# - 3) 令$\eta=0.25$，重做1)，这种改变对训练有何影响？

# ### 输入模式

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt
plt.xlabel('X(1)')
plt.ylabel('X(2)')

x1 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y1 = np.sqrt(abs(2-x1**2))
x2 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y2 = -np.sqrt(abs(2-x1**2))
plt.scatter(x1,y1,s=0.1,c="#A9A9A9")
plt.scatter(x2,y2,s=0.1,c="#A9A9A9")

plt.scatter([1,1,-1],[-1,1,-1],c="#ff1212")
plt.axis('equal')
plt.show()


# ### 按顺序训练

# In[2]:


def iterator(w,x,t):
    net1 = np.dot(w[0],x)
    net2 = np.dot(w[1],x)
    if(net1 < net2):
        w[1] = w[1] + t*(x-w[1])
        s = np.dot(w[1],w[1])
        w[1] = math.sqrt(2/s)*w[1]
        ww = w[1]
        n = 1
        return ww,n
    else:
        w[0] = w[0] + t*(x-w[0])
        s = np.dot(w[0],w[0])
        w[0] = math.sqrt(2/s)*w[0]
        ww = w[0]
        n = 0
        return ww,n

x1 = np.array([1,-1])
x2 = np.array([1,1])
x3 = np.array([-1,-1])
x = [x1,x2,x3]
w1 = np.array([math.sqrt(2),0])
w2 = np.array([0,math.sqrt(2)])
w = [w1,w2]
t = 0.5
k = 0
for j in range(1):
    for i in range(3):
        ww,n = iterator(w,x[i],t)
        print('第{}步训练：'.format(3*k+i+1),'x{}'.format(i+1))
        print('权值w({})竞争胜利！'.format(n),'更新为===>',ww)
    k += 1


# 从训练结果可以看出：$X^1$,$X^3$属于同一类,$X^2$属于另一类。
# 
# ### 图示训练结果，观察模式如何聚类

# In[3]:


import math
import numpy as np
import matplotlib.pyplot as plt
#Mac字体库中默认没有'SimHei'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('X(1)')
plt.ylabel('X(2)')

x_1 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y_1 = np.sqrt(abs(2-x_1**2))
x_2 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y_2 = -np.sqrt(abs(2-x_1**2))

plt.scatter(x_1,y_1,s=0.1,c="#A9A9A9")
plt.scatter(x_2,y_2,s=0.1,c="#A9A9A9")

plt.scatter([0.27589938],[-1.38703985],c="#ff1212",marker='^',label='w1')
plt.scatter([0.5411961],[1.30656296],c="#0000FF",marker='^',label='w2')

plt.scatter([1,-1],[-1,-1],c="#ff1212",label='类I')
plt.scatter([1],[1],c="#0000FF",label='类II')
plt.legend(loc=4)
plt.axis('equal')
plt.show()


# ### 改变输入模式的顺序

# In[4]:


x = [x2,x3,x1]

t = 0.5
k = 0
for j in range(1):
    for i in range(3):
        ww,n = iterator(w,x[i],t)
        print('第{}步训练：'.format(3*k+i+1))
        print('权值w({})竞争胜利！'.format(n),'更新为===>',ww)
    k += 1


# In[5]:


x = [x3,x2,x1]

t = 0.5
k = 0
for j in range(1):
    for i in range(3):
        ww,n = iterator(w,x[i],t)
        print('第{}步训练：'.format(3*k+i+1))
        print('权值w({})竞争胜利！'.format(n),'更新为===>',ww)
    k += 1


# In[6]:


x = [x2,x1,x3]

t = 0.5
k = 0
for j in range(1):
    for i in range(3):
        ww,n = iterator(w,x[i],t)
        print('第{}步训练：'.format(3*k+i+1))
        print('权值w({})竞争胜利！'.format(n),'更新为===>',ww)
    k += 1


# ### 改变输入模式的顺序，并没有改变训练结果。一部分原因是聚类的结果与初始权值的关系相当大，还有一部分是输入模式的数量和分布所造成的。

# ### 学习率$\eta$=0.25

# In[7]:


x = [x1,x2,x3]

t = 0.25
k = 0
for j in range(1):
    for i in range(3):
        ww,n = iterator(w,x[i],t)
        print('第{}步训练：'.format(3*k+i+1))
        print('权值w({})竞争胜利！'.format(n),'更新为===>',ww)
    k += 1


# In[8]:


plt.xlabel('X(1)')
plt.ylabel('X(2)')

x_1 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y_1 = np.sqrt(abs(2-x_1**2))
x_2 = np.arange(-math.sqrt(2),math.sqrt(2),0.00001)
y_2 = -np.sqrt(abs(2-x_1**2))

plt.scatter(x_1,y_1,s=0.1,c="#A9A9A9")
plt.scatter(x_2,y_2,s=0.1,c="#A9A9A9")

plt.scatter([-0.17215904],[-1.40369557],c="#ff1212",marker='^',label='w1')
plt.scatter([0.96251325],[1.03613139],c="#0000FF",marker='^',label='w2')

plt.scatter([1,-1],[-1,-1],c="#ff1212",label='类I')
plt.scatter([1],[1],c="#0000FF",label='类II')
plt.legend(loc=4)
plt.axis('equal')
plt.show()


# In[ ]:




