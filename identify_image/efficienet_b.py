from efficientnet_pytorch_3d import EfficientNet3D
import torch
from torchvision import datasets,transforms,models
import numpy as np
import os
import argparse
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
from torchsummary import summary
import torch.nn as nn
import torch
from torch import Tensor
gpus = [2,1]
train_dir = 'train2'
valid_dir = 'valid2'
test_dir = 'test2'

train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))])
# 之后对张量的三个颜色通道数值的均值和标准差进行标准化处理，使神经网络在训练中得到最好的效果。

test_valid_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

# 使用预处理格式加载图像
train_data = datasets.ImageFolder(train_dir,transform = train_transforms)
test_data = datasets.ImageFolder(test_dir,transform = test_valid_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform = test_valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data,batch_size = 32,shuffle = True,
                                          drop_last=False,num_workers=4,pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data,batch_size = 32,
                                         drop_last=False,num_workers=4,pin_memory=True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size = 32,
                                          drop_last=False,num_workers=4,pin_memory=True)

print("数据加载完毕")

model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 3}, in_channels=3)

# summary(model, input_size=(3, 50, 50, 50))
print(model)

def accuracy_test(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    nnn=0
    model.to('cuda:2')  # 将模型放入GPU计算，能极大加快运算速度
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    running_loss = 0.0
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            tmp = images.shape[0]
            images = images.reshape(tmp, 3, 50, 50, 50)
            images, labels = images.to('cuda:2'),labels.to('cuda:2')

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            nnn+=1

            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
    print('Loss : {:.4f}'.format(running_loss / nnn))
    valid_loss_3_32.append(1.0*running_loss/nnn)
    print('the accuracy is {:.4f}'.format(correct / total))
    valid_accc_3_32.append(1.0*correct/total)

def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer):
    epochs = epochs  # 设置学习次数
    print_every = print_every
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    # model.to('cuda:1')

    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0]).to('cuda:2')

    for e in range(epochs):
        # if(e>0):
        #     break
        running_loss = 0
        nnn = 0
        steps = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            model.train()
            steps+=1
            # print("开始处理第" + str(steps) + "批图像")
            tmp = inputs.shape[0]
            inputs=inputs.reshape(tmp,3,50,50,50)

            inputs, labels = inputs.to('cuda:2'), labels.to('cuda:2')
            # inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()  # 优化器梯度清零

            # 前馈及反馈
            outputs = model(inputs)  # 数据前馈，正向传播
            loss = criterion(outputs, labels)  # 输出误差
            loss.backward()  # 误差反馈
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()

            if steps % print_every == 0:
                # test the accuracy
                # print(total)
                print('EPOCHS : {}-{}/{}'.format(e + 1, nnn + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss / print_every))
                X_3_32.append(str(e+1)+"-"+str(nnn+1))
                train_loss_3_32.append(1.0*running_loss/print_every)
                print("验证集：")
                accuracy_test(model, validloader, criterion)
                running_loss=0
                nnn+=1


loss_func=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.RMSprop(model.parameters(),lr=0.0001)
X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]
valid_accc_3_32.append(0.3333)
print("模型构建完毕")

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(model,trainloader,20,120,loss_func,optimizer)

print("测试集最终概率：")
accuracy_test(model, testloader, loss_func)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

np.save("X_32_effi_b",X_3_32_2)
np.save("train_loss_effi_b",train_loss_3_32_2)
np.save("valid_loss_effi_b",valid_loss_3_32_2)
np.save("valid_accc_effi_b",valid_accc_3_32_2)

path_state_dict = "./model_state_dict_effi_b.pkl"
net_state_dict = model.state_dict()
torch.save(net_state_dict, path_state_dict)
