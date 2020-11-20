#!/usr/bin/env python
# coding: utf-8

# # 第二次编程作业-人工神经网络模型基础
# 
# >姓名：潘冯谱(2018级本科)
# >
# >学校：华东理工大学
# >
# >专业：数学与应用数学
# 
# 修改于2020-10-26

# ## 习题3.14 BP神经网络
# 
# 1986年由Rumelhart和McClelland为首的科学家提出的概念，是一种按照误差逆向传播算法训练的多层前馈神经网络，是应用最广泛的神经网络。
# 
# [简单了解神经网络](https://www.cnblogs.com/maybe2030/p/5597716.html#_label2)
# 
# [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
# 
# ### 编程目标：
# <img src="https://images2015.cnblogs.com/blog/853467/201606/853467-20160630141449671-1058672778.png" width="300">
# 
# BP网络结构如图所示，初始权值（见代码）。网络的输入模式为$X = (-1,1,3)^T$，期望输出为$d = (0.95,0.05)^T$。试对单次训练过程进行分析，求出：
# 
# - 隐含层权值矩阵$V$和输出层权值矩阵$W$；
# - 各层净输入和输出：$net^y$、$Y$和$net^o$、$O$，其中上标$y$代表隐含层，$o$代表输出层；
# - 各层输出的一阶导数$f'(net^y)$和$f'(net^o)$；
# - 各层误差信号$\delta^o$和$\delta^y$；
# - 各层权值调整量$\Delta V$和$\Delta W$；
# - 调整后的权值矩阵$V$和$W$；

# ## 输出层转移函数为$f(net)=net$

# In[1]:


import numpy as np
import math

x0 = [-1,1,3]

d = np.array([0,0.95,0.05])

w1 = [[3,1,-2],[-1,2,0]]
w2 = [[-2,1,0],[3,1,-2]]
t = 1
net1 = [-1,0,0]
net2 = [0,0,0]
delta1 = [0,0,0]


def sigmod(x):      # 单极性sigmod
    x = 1/(1+math.exp(-1*x))
    return x

def sigmod2(x):      # 双极性sigmod
    x = (1-math.exp(-1*x))/(1+math.exp(-1*x))
    return x
    
def identity(x):    # 恒等函数
    return x

# 输入样本数据前馈传播
for i in range(2):
    net1[i+1] = sigmod(np.dot(w1[i],x0))
print('输入层到隐含层：',net1)

for i in range(2):
    net2[i+1] = np.dot(w2[i],net1)
print('隐含层到输出层：',net2)
net1[0] = 0

# 误差反向传播
# 输出层的权值调整
delta2 = (net2 - d)*net1
total_error = np.dot(delta2,delta2)
print(total_error)

w2[0] = w2[0] + t*delta2[1]
w2[1] = w2[1] + t*delta2[2]
print(w2)

# 隐含层的权值调整
dg = net1*(1-np.array(net1))
delta1[0]= np.dot(delta2,w1[0])
delta1[1]= np.dot(delta2,w1[1])
w1[0] = w1[0] + t*delta1[0]*dg[1]*x0[1]
w1[1] = w1[1] + t*delta1[1]*dg[2]*x0[2]
print(w1)


# ## 训练BP神经网络模型（权值矩阵）

# In[2]:


import numpy as np
import math

x0 = [-1,1,3]

d = np.array([0,0.95,0.05])

w1 = [[3,1,-2],[-1,2,0]]
w2 = [[-2,1,0],[3,1,-2]]
t = 1
net1 = [-1,0,0]
net2 = [0,0,0]
delta1 = [0,0,0]
error = []

def sigmod(x):      # 单极性sigmod
    x = 1/(1+math.exp(-1*x))
    return x

def sigmod2(x):      # 双极性sigmod
    x = (1-math.exp(-1*x))/(1+math.exp(-1*x))
    return x

def identity(x):    # 恒等函数
    return x

epoch = 0
total_error = 20
while total_error > 1e-5:
        # 输入样本数据前馈传播
    for i in range(2):
        net1[i+1] = sigmod(np.dot(w1[i],x0))
    print('输入层到隐含层：',net1)

    for i in range(2):
        net2[i+1] = np.dot(w2[i],net1)
    print('隐含层到输出层：',net2)
    net1[0] = 0

    # 误差反向传播
    # 输出层的权值调整
    delta2 = (net2 - d)*net1
    total_error = np.dot(delta2,delta2)

    w2[0] = w2[0] - t*delta2[1]
    w2[1] = w2[1] - t*delta2[2]
    print(w2)

    # 隐含层的权值调整
    dg = net1*(1-np.array(net1))
    delta1[0]= np.dot(delta2,w1[0])
    delta1[1]= np.dot(delta2,w1[1])
    w1[0] = w1[0] - t*delta1[0]*dg[1]*x0[1]
    w1[1] = w1[1] - t*delta1[1]*dg[2]*x0[2]
    print(w1)
    error.append(total_error)
    epoch +=1
    print('epoch=',epoch,'total_error=',total_error)


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(np.arange(epoch)+1,error)
plt.legend(['Training'])
plt.show()


# In[4]:


epoch

