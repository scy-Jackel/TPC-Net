import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import seaborn as sb
import torch.nn as tnn
import os

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=3):

        super().__init__()

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 3, 3),
                                    stride=(conv1_t_stride, 1, 1),
                                    padding=(conv1_t_size // 2, 1, 1),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out,
                                    output_size=(1, 1,
                                                 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
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

trainloader = torch.utils.data.DataLoader(train_data,batch_size = 32,shuffle = True,drop_last=True)
testloader = torch.utils.data.DataLoader(test_data,batch_size = 32,drop_last=True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size = 32,drop_last=True)

print("数据加载完毕")

X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]

myDenseNet=generate_model(264)

print("模型构建完毕")
print(myDenseNet)

loss_func=torch.nn.NLLLoss()

optimizer=torch.optim.RMSprop(myDenseNet.parameters(),lr=0.0001)

def accuracy_test(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    model.to('cuda:0')  # 将模型放入GPU计算，能极大加快运算速度
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    running_loss = 0.0
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            tmp = images.shape[0]
            images = images.reshape(tmp, 3, 50, 50, 50)
            images, labels = images.to('cuda:0'),labels.to('cuda:0')

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # 将预测及标签两相同大小张量逐一比较各相同元素的个数
    print('Loss : {:.4f}'.format(running_loss / total))
    valid_loss_3_32.append(1.0*running_loss/total)
    print('the accuracy is {:.4f}'.format(correct / total))
    valid_accc_3_32.append(1.0*correct/total)

def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer):
    epochs = epochs  # 设置学习次数
    print_every = print_every
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    model.to('cuda:0')

    for e in range(epochs):
        # if(e>0):
        #     break
        running_loss = 0
        total = 0
        nnn = 0
        steps = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            model.train()
            steps+=1
            # print("开始处理第" + str(steps) + "批图像")
            tmp = inputs.shape[0]
            inputs=inputs.reshape(tmp,3,50,50,50)
            inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
            optimizer.zero_grad()  # 优化器梯度清零

            # 前馈及反馈
            outputs = model(inputs)  # 数据前馈，正向传播
            loss = criterion(outputs, labels)  # 输出误差
            loss.backward()  # 误差反馈
            optimizer.step()  # 优化器更新参数

            running_loss += loss.item()
            total += labels.size(0)

            if steps % print_every == 0:
                # test the accuracy
                # print(total)
                print('EPOCHS : {}-{}/{}'.format(e + 1, nnn + 1, epochs),
                      'Loss : {:.4f}'.format(running_loss / total))
                X_3_32.append(str(e+1)+"-"+str(nnn+1))
                train_loss_3_32.append(1.0*running_loss/total)
                print("验证集：")
                accuracy_test(model, validloader, criterion)
                running_loss=0
                total=0
                nnn+=1

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(myDenseNet,trainloader,20,60,loss_func,optimizer)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

np.save("X_32_DenseNet",X_3_32_2)
np.save("train_loss_32_DenseNet",train_loss_3_32_2)
np.save("valid_loss_32_DenseNet",valid_loss_3_32_2)
np.save("valid_accc_32_DenseNet",valid_accc_3_32_2)

print("测试集最终概率：")
accuracy_test(myDenseNet, testloader, loss_func)

path_state_dict = "./model_state_dict_32_DenseNet.pkl"

net_state_dict = myDenseNet.state_dict()
torch.save(net_state_dict, path_state_dict)
