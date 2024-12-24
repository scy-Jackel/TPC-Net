# %matplotlib inline
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

# # device='cuda:0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
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

class myAlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)

        )
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

class myAlexNet2(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5 * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            # nn.Softmax(dim=1)

        )
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

class myAlexNet3(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5 * 5, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            # nn.Softmax(dim=1)

        )
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

class myAlexNet4(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5 * 5, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            # nn.Softmax(dim=1)

        )
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

def accuracy_test(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    nnn=0
    model.to('cuda:1')  # 将模型放入GPU计算，能极大加快运算速度
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    running_loss = 0.0
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            tmp = images.shape[0]
            images = images.reshape(tmp, 3, 50, 50, 50)
            images, labels = images.to('cuda:1'),labels.to('cuda:1')

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
    # valid_loss_3_32.append(1.0*running_loss/nnn)
    print('the accuracy is {:.4f}'.format(correct / total))
    # valid_accc_3_32.append(1.0*correct/total)

path_state_dict = "./model_state_dict_32_AlexNet_256.pkl"
state_dict_load = torch.load(path_state_dict)
net_new256 = myAlexNet()
net_new256.load_state_dict(state_dict_load)
loss_func = torch.nn.CrossEntropyLoss()
accuracy_test(net_new256, testloader, loss_func)

path_state_dict = "./model_state_dict_32_AlexNet_512.pkl"
state_dict_load = torch.load(path_state_dict)
net_new512 = myAlexNet2()
net_new512.load_state_dict(state_dict_load)
loss_func = torch.nn.CrossEntropyLoss()
accuracy_test(net_new512, testloader, loss_func)

path_state_dict = "./model_state_dict_32_AlexNet_1024.pkl"
state_dict_load = torch.load(path_state_dict)
net_new1024 = myAlexNet3()
net_new1024.load_state_dict(state_dict_load)
loss_func = torch.nn.CrossEntropyLoss()
accuracy_test(net_new1024, testloader, loss_func)


