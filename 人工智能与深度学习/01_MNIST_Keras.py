#!/usr/bin/env python
# coding: utf-8

# # 利用Keras库对MNIST数据集进行研究
# 
# >姓名：潘冯谱(2018级本科)
# >
# >学校：华东理工大学
# >
# >专业：数学与应用数学
# 
# 修改于 2020-10-22
# 
# ### MNIST数据集
# 
# [官方文档](http://yann.lecun.com/exdb/mnist/)
# 
# The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
# 
# It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.

# ### 多层感知机网络
# 
# ![](http://mp.ofweek.com/Upload/News/Img/member20047/201905/wx_article_20190526215523_1v9KVb.jpg)

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# ## 导入MNIST数据集

# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# 从Keras的数据集中导入MNIST数据集，分别构造训练集、测试集的特征和标签
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('训练集中共有{}个样本'.format(len(X_train)))
print('测试集中共有{}个样本'.format(len(X_test)))


# ## 可视化图像

# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示中文标签

import numpy as np


# In[10]:


# 训练数字图像的大小
X_train[0].shape


# In[12]:


# 打印训练集数据集中样本的图像
i = 10
plt.imshow(X_train[i])
plt.title('真实标签：'+str(y_train[i]))
plt.show()


# In[13]:


# 绘制像素热力图
def visualize_input(img, ax):
    # 先绘制数字的大图，然后对784个像素每个标注灰度值
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')
i = 6  # 绘制数据集中索引为5的灰度图像
fig = plt.figure(figsize = (10,10))  # 设定图的总大小
ax = fig.add_subplot(111)
# 调用绘图函数
visualize_input(X_train[i], ax)


# In[14]:


import matplotlib.cm as cm


# In[15]:


# 绘制前16张图像
for i in range(16):
    plt.style.use({'figure.figsize':(15,15)})  # 设置图像大小
    plt.subplot(1,4,i%4+1)  #
    plt.imshow(X_train[i])
    title = '真实标签：{}'.format(str(y_train[i]))
    plt.title(title)
    plt.xticks([])  #关闭刻度线显示
    plt.yticks([])  #
    plt.axis('off')  #关闭坐标轴显示
    if i%4 == 3:
        plt.show()


# In[16]:


# 绘制前16张图像的灰度图
for i in range(16):
    plt.style.use({'figure.figsize':(15,15)})  # 设置图像大小
    plt.subplot(1,4,i%4+1)  #
    plt.imshow(X_train[i], cmap='gray')
    title = '真实标签：{}'.format(str(y_train[i]))
    plt.title(title)
    plt.xticks([])  #关闭刻度线显示
    plt.yticks([])  #
    plt.axis('off')  #关闭坐标轴显示
    if i%4 == 3:
        plt.show()


# ## 预处理——归一化

# In[17]:


# 缩放像素值[0,255] --> [0,1]
# 神经网络模型对输入数据幅度敏感，进行数据归一化预处理可以让后续神经网络模型收敛速度更快，效果更好
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# ## 预处理——标签独热向量编码
# 
# ### One-Hot-Encoding
# 在构造损失函数时，用One-Hot编码向量计算交叉熵损失函数

# In[20]:


from keras.utils import np_utils

# 将训练集和测试集的标签转化为One-Hot编码向量
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# ## 构建MLP多层感知机神经网络
# 
# Dense 密集层，也就是全连接层
# 
# Dropout 丢弃层，防止过拟合
# 
# Flatten 将28*28的图像拉平成784维的长向量

# In[22]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Flatten(input_shape=(28,28)))

# 512个神经元的全连接神经网络，激活函数为Relu,Dropout率为0.2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
          
# 512个神经元的全连接神经网络，激活函数为Relu,Dropout率为0.2
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
          
# 输出层，10个神经元，softmax输出，对应图像为10个数字的概率
model.add(Dense(10, activation='softmax'))


# In[23]:


model.summary()


# In[24]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ## 模型训练前，在测试集上进行评估
# 
# 第一个数字是交叉熵损失，第二个数字是小数形式的准确率

# In[25]:


model.evaluate(X_test, y_test, verbose=1)


# ## 训练模型
# 
# 训练的时候，每步传入128个样本，总共训练10轮，也就是完整遍历10遍训练集，每轮训练后在验证集上进行测试
# 
# 验证集是从60,000张原始训练集中抽选出来的1/5样本，也就是12,000张样本，原始训练集中剩下的48,000个样本作为训练集
# 
# 使用ModekCheckpoint及时存储验证集上准确率最高的最优模型，存储为当前目录下的"mnist.model.best.hdf5"文件，早停可以防止过拟合
# 
# [Keras_ModekCheckpoint官方文档](https://keras.io/api/callbacks/#modekcheckpoint)

# In[26]:


from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

hist = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[checkpointer], verbose=1, shuffle=True)


# ## 可视化训练过程的信息

# In[27]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['accuracy'])
    plt.plot(network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
plot_history(hist)


# ## 加载训练过程中验证集上准确率最高的最优模型

# In[28]:


model.load_weights('mnist.model.best.hdf5')


# ## 在测试集上评估模型

# In[29]:


model.evaluate(X_test, y_test, verbose=1)


# ## 对一些样本进行预测

# In[30]:


i = 6
img_test = X_test[i].reshape(-1,28,28)
prediction = model.predict(img_test)[0]


# In[31]:


prediction


# In[34]:


np.argmax(prediction)


# In[35]:


i = 10

plt.imshow(X_test[i])
img_test = X_test[i].reshape(-1,28,28)
prediction = model.predict(img_test)[0]

title = '真实标签：{}\n预测标签：{}'.format(np.argmax(y_test[i]), np.argmax(prediction))
plt.title(title)
plt.show()

plt.bar(range(10), prediction)
plt.title('预测概率分布')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.show()

