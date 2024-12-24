# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import seaborn as sb
import os
import tifffile
from PIL import Image,ImageDraw
train_dir="train"
train_transforms = transforms.Compose([transforms.ToTensor()])
# 使用预处理格式加载图像
train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
# 创建三个加载器，分别为训练，验证，测试，将训练集的batch大小设为64，即每次加载器向网络输送64张图片，随机打乱
trainloader = torch.utils.data.DataLoader(train_data,batch_size = 32,shuffle = True)

print("数据加载完毕")

for ii, (inputs, labels) in enumerate(trainloader):
    print(inputs.shape)
    print(labels.shape)
    inputs=inputs.reshape(32,3,50,50,50)
    print(ii)



