#!/usr/bin/env python
# coding: utf-8

# # 第一次编程作业-人工神经网络模型基础
# 
# >姓名：潘冯谱(2018级本科)
# >
# >学校：华东理工大学
# >
# >专业：数学与应用数学
# 
# 修改于2020-10-25

# ## 习题2.4 [Hebb学习规则](https://www.sciencedirect.com/science/article/pii/S0361923099001823)
# 
# 1949年，[D.O.Hebb](http://cogprints.org/1652/1/hebb.html)最早提出了关于神经网络学习机理的“突触修正”的假设。该假设指出，当神经元的突触前膜电位与突触后膜电位同时为正，突触传导增强；反之，当突触前膜电位与后膜电位正负相反时，突触传导减弱。即当神经元 i 与神经元 j 同时处于兴奋时，连接强度应增强。根据该假设定义的权值调整方法，称Hebb学习规则。
# 在Hebb学习规则中，学习信号简单地等于神经元的输出：$$r=f(W_j^T X)$$
# 权向量的调整公式为：$$\Delta W_j = \eta f(W_j^T X)X$$
# 权向量中，每个分量的调整由下式确定：$$\Delta w_{ij} = \eta f(W_j^T X)x_i = \eta o_jx_i$$
# 
# 
# ### 编程目标：
# 双输入单输出神经网络，初始权向量$W(0)=(1,-1)^T$，学习率$\eta = 1$，4个输入向量为$X^1 = (1,-2)^T$，$X^2 = (0,1)^T$，$X^3 = (2,3)^T$，$X^4 = (1,1)^T$，若采用Hebb学习规则，对以下两种情况求第四步训练后的权向量：
# - 神经元采用离散型转移函数$f(net) = sgn(net)$;
# - 神经元采用双极性连续型转移函数$f(net) = \frac{1 - e^{-net}}{1 + e^{-net}}$

# In[1]:


import numpy as np
import math

def activation_sgn(i,w,x,t):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    if(net >= 0):
        f = 1
    else:
        f = -1
    w = w + t*f*x
    print('f(net)=',f,'权向量：',w)
    return w
# 为展示效果S型函数的一切输出值保留4位小数
def activation_sigmod(i,w,x,t):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    f = (1-math.exp(-1*net))/(1+math.exp(-1*net))
    w = w + t*f*x
    print('f(net)=',round(f,4),'权向量：',[round(i,4) for i in w])
    return w

x1 = np.array([1,-2])
x2 = np.array([0,1])
x3 = np.array([2,3])
x4 = np.array([1,1])
x = [x1,x2,x3,x4]
w = [1,-1]
t = 1

print('神经元采用离散型转移函数：')
for i in range(4):
    w = activation_sgn(i+1,w,x[i],t)
    
print('\n神经元采用双极性连续型转移函数：')
w = np.array([1,-1])
for i in range(4):
    w = activation_sigmod(i+1,w,x[i],t)


# ### What I learnt：
# * Hebb学习规则的调整公式表明，权值调整量与输入输出的乘积成正比。在这种情况下，Hebb学习规则需预先设置权饱和值，防止输入和输出正负始终一致时出现权值的无约束增长。
# * 要求权值初始化，即在学习开始前(t=0)，先对$W_j(0)$赋予0附近的小随机数。
# * Hebb学习规则代表了一类纯前馈、无监督学习方法。
# * 比较两种权值调整结果可以看出，两种转移函数下的权值调整方向是一致的，但采用连续转移函数时，权值调整力度减弱。

# ## 习题2.5 [Perceptron](https://www.britannica.com/technology/perceptrons#ref1009913)学习规则
# 
# 1958年，Frank Rosenblatt 首次定义了一个具有单层计算单元的神经网络结构，称为[Perceptron(感知机)](https://forum.huawei.com/enterprise/en/perceptron/thread/624123-100504)。
# 
# 感知机的学习信号等于神经元的期望输出与实际输出之差：$$r = d_j - o_j$$
# 其中，$d_j$为期望输出，$o_j = f(W_j^T X)$。感知机采用符号转移函数，表达式为：
# \begin{equation}
#  f(W_j^T X) = sgn(W_j^T X) = \left\{
# \begin{aligned}
# 1 & & W_j^T X\geq 0 \\
# -1 & & W_j^T X<0
# \end{aligned}
# \right.
# \end{equation}
# 权向量的调整公式为：$$\Delta W_j = \eta [d_j - sgn(W_j^T X)]X$$
# 权向量中，每个分量的调整由下式确定：$$\Delta w_{ij} = \eta [d_j - sgn(W_j^T X)]x_i$$
# 
# 
# ### 编程目标：
# 转移函数$f(net) = sgn(net)$，学习率$\eta = 1$，初始权向量$W(0)=(0,1,0)^T$，两对输入样本为$X^1 = (2,1,-1)^T$，$d^1 = -1$；$X^2 = (0,-1,-1)^T$，$d^2 = 1$，试用感知机学习规则对以上样本进行反复训练，直至网络输出误差为 0，写出每一训练步的净输入net(t)

# In[2]:


import numpy as np

def activation_sgn(i,w,x,t,d):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    if(net >= 0):
        f = np.array([1])
    else:
        f = np.array([-1])
    dt = d-f
    w = w + t*dt*x
    print('f(net)=',f,'权向量：',w,'输出误差：',dt)
    return w,dt

x1 = np.array([2,1,-1])
d1 = np.array([-1])
x2 = np.array([0,-1,-1])
d2 = np.array([1])
x = [x1,d1,x2,d2]
w = [0,1,0]
t = 1
dt = 1
j = 0
while dt != 0 :
    for i in range(2):
        w,dt = activation_sgn(2*j+i+1,w,x[2*i],t,x[2*i+1])
        if(dt == 0):
            break
    j += 1


# ### What I learnt：
# * 当实际输出与期望输出相同时，权值不需要调整（没有学习到新知识）。
# * 转移函数规定为符号函数，权值调整公式可以进一步简化。
# * 感知机学习规则只适用于二进制神经元，初始权值可以取任意值。
# * 感知机学习规则代表了一类有监督学习方法。

# ## 习题2.6 $\delta$ 学习规则
# 
# 1986年，[McClelland 和 Rumelhart](https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12155)在神经网络训练中引入了$\delta（Delta）$规则。该规则亦可称为连续感知机学习规则。
# 
# 在$\delta$学习规则中，学习信号很容易由对神经元实际输出与期望输出的均方误差求误差梯度得到：$$r = [d_j - f(W_j^T X)]f'(W_j^T X) = (d_j - o_j)f'(net)$$
# 权向量的调整公式为：$$\Delta W_j = \eta (d_j - o_j)f'(net_j)X$$
# 权向量中，每个分量的调整由下式确定：$$\Delta w_{ij} = \eta (d_j - o_j)f'(net_j)x_i$$
# 
# 
# ### 编程目标：
# 神经网络采用双极性Sigmoid函数，学习率$\eta = 0.25$，初始权向量$W(0)=(1,0,1)^T$，两对输入样本为$X^1 = (2,0,-1)^T$，$d^1 = -1$；$X^2 = (1,-2,-1)^T$，$d^2 = 1$，试用$\delta$ 学习规则进行训练，写出每一训练的训练结果。

# In[3]:


import numpy as np
import math

def activation_sigmod(i,w,x,t,d):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    f = (1-math.exp(-1*net))/(1+math.exp(-1*net))
    f = np.array([f])
    dt = d-f
    w = w + t*dt*(0.5)*(1-(f ** 2))*x
    print('f(net)=',[round(i,4) for i in f],'权向量：',[round(i,4) for i in w],'输出误差：',[round(i,4) for i in dt])
    return w,dt

x1 = np.array([2,0,-1])
d1 = np.array([-1])
x2 = np.array([1,-2,-1])
d2 = np.array([1])
x = [x1,d1,x2,d2]
w = [1,0,1]
t = 0.25
dt = 1
j = 0
while abs(dt) >= 1 :
    for i in range(2):
        w,dt = activation_sigmod(2*j+i+1,w,x[2*i],t,x[2*i+1])
        if(abs(dt) <= 1):
            break
    j += 1


# ### What I learnt：
# * $\delta$ 学习规则可推广到多层前馈网络中，初始权值可以取任意值。
# * $\delta$ 学习规则要求转移函数可导，只适用于有监督学习中的连续转移函数，如Sigmoid函数。

# ## 习题2.7 LMS学习规则
# 
# 1962年，Bernard Widrow 和 Marcian Hoff 提出了 Widrow-Hoff 学习规则。因它能时神经元实际输出与期望输出之间的平方差最小，故也称最小均方规则(LMS)。
# 
# 在LMS学习规则的学习信号为：$$r = d_j - W_j^TX$$
# 权向量的调整公式为：$$\Delta W_j = \eta (d_j - W_j^TX)X$$
# 权向量中，每个分量的调整由下式确定：$$\Delta w_{ij} = \eta (d_j - W_j^TX)x_i$$
# 
# 
# ### 编程目标：
# 神经网络数据同习题2.6，试用 Widrow-Hoff 学习规则进行训练，写出每一训练的训练结果。

# In[4]:


import numpy as np
import math

def activation_sigmod(i,w,x,t,d):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    f = net
    f = np.array([f])
    dt = d-f
    w = w + t*dt*x
    print('f(net)=',[round(i,4) for i in f],'权向量：',[round(i,4) for i in w],'输出误差：',[round(i,4) for i in dt])
    return w,dt

x1 = np.array([2,0,-1])
d1 = np.array([-1])
x2 = np.array([1,-2,-1])
d2 = np.array([1])
x = [x1,d1,x2,d2]
w = [1,0,1]
t = 0.25
dt = 1
j = 0
while abs(dt) >= 1 :
    for i in range(2):
        w,dt = activation_sigmod(2*j+i+1,w,x[2*i],t,x[2*i+1])
        if(abs(dt) <= 1):
            break
    j += 1


# ### What I learnt：
# * LMS学习规则可以看作是 $\delta$学习规则的一个特殊情况（$f(W_j^T X) = W_j^T X$）。
# * LMS学习规则不需要对转移函数求导，不仅学习速度较快，而且精度较高，初始权值可以取任意值。

# ## 习题3.2 简单分类问题
# ![](https://img-blog.csdnimg.cn/20201025192523496.bmp#pic_center)
# ![](https://img-blog.csdnimg.cn/20201025192523580.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI0MzczMDA3,size_16,color_FFFFFF,t_70#pic_center)
# 
# - （a）权值：1，-0.5；阈值：0
# - （b）权值：0，1；阈值：0.5
# - （c）权值：0.5，0.5；阈值：-0.75

# ## 拓展练习
# 使用感知机神经网络解决线性分类问题

# In[5]:


import numpy as np
import math

def activation_sigmod(i,w,x,t,d):
    net = np.dot(w,x)
    print('第{}步训练后-->'.format(i),'net=',net)
    f = (1-math.exp(-1*net))/(1+math.exp(-1*net))
    f = np.array([f])
    dt = d-f
    w = w + t*dt*(0.5)*(1-(f ** 2))*x
    print('f(net)=',[round(i,4) for i in f],'权向量：',[round(i,4) for i in w],'输出误差：',[round(i,4) for i in dt])
    return w,dt

x1 = np.array([0,1,-1])
d1 = np.array([-1])
x2 = np.array([-1,1,-1])
d2 = np.array([-1])
x3 = np.array([-1,0,-1])
d3 = np.array([-1])
x4 = np.array([-1,-1,-1])
d4 = np.array([-1])
x5 = np.array([0,-1,-1])
d5 = np.array([1])
x6 = np.array([1,-1,-1])
d6 = np.array([1])
x7 = np.array([1,0,-1])
d7 = np.array([1])
x8 = np.array([1,1,-1])
d8 = np.array([1])
x = [x1,d1,x2,d2,x3,d3,x4,d4,x5,d5,x6,d6,x7,d7,x8,d8]
w = [0,0,0]
t = 0.15
dt = 1
j = 0
while abs(dt) >= 1e-1 :
    for i in range(8):
        w,dt = activation_sigmod(2*j+i+1,w,x[2*i],t,x[2*i+1])
        if(abs(dt) <= 1e-1):
            break
    j += 1


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
for k in range(4):
    plt.scatter(x[2*k][0],x[2*k][1],c="#ff1212",label='类I')
for j in range(4):
    plt.scatter(x[2*j+8][0],x[2*j+8][1],c="#0000FF",label='类II')
x = np.arange(-0.6,1,0.4)
y = 2.098/0.8881*x + 0.0006/0.8881
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-')


# ## 习题3.5 线性二分类问题
# 
# ![](https://img-blog.csdnimg.cn/20201025192523676.bmp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI0MzczMDA3,size_16,color_FFFFFF,t_70#pic_center)
# 
# ### 编程目标：
# 已知以下样本分属于两类：
# - $X^1 = (5,1)^T$，$X^2 = (7,3)^T$，$X^3 = (3,2)^T$，$X^4 = (5,4)^T$
# - $X^5 = (0,0)^T$，$X^6 = (-1,-3)^T$，$X^7 = (-2,3)^T$，$X^8 = (-3,0)^T$
# 
# 1.判断两类样本是否线性可分。
# 
# 2.试确定一直线，并使该直线方程与两类样本中心连线相垂直，且过中点。
# 
# 3.设计一单节点感知机，如用上述直线方程作为其分类判决方程 net = 0，写出感知机的权值与阈值。
# 
# 4.用上述感知机对以下3个样本进行分类：$$X = (4,2)^T, X = (0,5)^T, X = (\frac{36}{13},0)^T$$

# ### 目标可视化

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('X(1)')
plt.ylabel('X(2)')
plt.scatter([5,7,3,5],[1,3,2,4],c="#ff1212",label='类I')
plt.scatter([0,-1,-2,-3],[0,-3,3,0],c="#0000FF",label='类II')
plt.legend(loc=4)
plt.show()


# ### 判断两类样本是否线性可分

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
plt.xlabel('X(1)')
plt.ylabel('X(2)')
x1 = [5,7,3,5]
y1 = [1,3,2,4]
x2 = [0,-1,-2,-3]
y2 = [0,-3,3,0]
plt.scatter(x1,y1,c="#ff1212",label='类I')
plt.scatter(x2,y2,c="#0000FF",label='类II')
plt.legend(loc=4)

x = np.arange(5)
y = -7/4*x + 4
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-')
plt.show()


# ### 确定决策直线垂直平分两类样本中心连线

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
plt.xlabel('X(1)')
plt.ylabel('X(2)')
x1 = np.array([5,7,3,5])
y1 = np.array([1,3,2,4])
x2 = np.array([0,-1,-2,-3])
y2 = np.array([0,-3,3,0])
plt.scatter(x1,y1,c="#ff1212",label='类I')
plt.scatter(x2,y2,c="#0000FF",label='类II')
plt.legend(loc=4)
[x1,y1,x2,y2] = [x1.mean(),y1.mean(),x2.mean(),y2.mean()]
x = np.arange(5)
y = (x2-x1)/(y1-y2)*(x-(x1+x2)/2) + (y1+y2)/2
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, '-')
plt.scatter(x1,y1,c="#F0E68C",marker='^',label='Center—I')
plt.scatter(x2,y2,c="#4B0082",marker='^',label='Center—II')
plt.axis('equal')
plt.show()
print('直线方程：','{}X(1) - X(2) + {} = 0'.format(m,round(b,1)))
b = b/(m-1)
a = -1/(m-1)
m = m/(m-1)
print('感知机权值：','[{},{}]'.format(round(m,2),round(a,2)),'感知机阈值：{}'.format(-round(b,2)))


# ### 感知机分类

# In[10]:


import numpy as np

def classification(i,w,x):
    net = np.dot(w,x)
    if(net <= 0):
        if(net == 0):
            print('ERROR')
        else:
            print('样本{}属于 II 类'.format(x))
    else:
        print('样本{}属于 I 类'.format(x))

x1 = np.array([-1,4,2])
x2 = np.array([-1,0,5])
x3 = np.array([-1,36/13,0])
x = [x1,x2,x3]
w = [1.61,0.72,0.28]

for i in range(3):
    classification(i+1,w,x[i])

