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

class myAlexNet1(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5 * 5, 512),
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

class myAlexNet3(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 72, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(72),
            nn.ReLU(inplace=True),
            nn.Conv3d(72, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(192, 288, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(288),
            nn.ReLU(inplace=True),
            nn.Conv3d(288, 288, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(288),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(288, 288, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(288),
            nn.ReLU(inplace=True),
            nn.Conv3d(288, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(192 * 5 * 5 * 5, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1536,768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(768, num_classes),
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
            nn.Conv3d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),
        )
        # self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5 * 5, 2048),
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
    valid_loss_3_32.append(1.0*running_loss/nnn)
    print('the accuracy is {:.4f}'.format(correct / total))
    valid_accc_3_32.append(1.0*correct/total)

def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer):
    epochs = epochs  # 设置学习次数
    print_every = print_every
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    model.to('cuda:1')

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
            inputs, labels = inputs.to('cuda:1'), labels.to('cuda:1')
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

myAlexNet=myAlexNet4()
print(myAlexNet)
myAlexNet=myAlexNet1()
print(myAlexNet)
myAlexNet=myAlexNet2()
print(myAlexNet)
myAlexNet=myAlexNet3()
print(myAlexNet)

myAlexNet=myAlexNet4()
loss_func=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.RMSprop(myAlexNet.parameters(),lr=0.0001)
X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]
valid_accc_3_32.append(0.3333)
print("模型构建完毕")
print(myAlexNet)

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(myAlexNet,trainloader,20,120,loss_func,optimizer)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

print("测试集最终概率：")
accuracy_test(myAlexNet, testloader, loss_func)

np.save("X_32_AlexNet_2048_2",X_3_32_2)
np.save("train_loss_32_AlexNet_2048_2",train_loss_3_32_2)
np.save("valid_loss_32_AlexNet_2048_2",valid_loss_3_32_2)
np.save("valid_accc_32_AlexNet_2048_2",valid_accc_3_32_2)

path_state_dict = "./model_state_dict_32_AlexNet_2048_2.pkl"

net_state_dict = myAlexNet.state_dict()
torch.save(net_state_dict, path_state_dict)

print("---")

myAlexNet=myAlexNet1()
loss_func=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.RMSprop(myAlexNet.parameters(),lr=0.0001)
X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]
valid_accc_3_32.append(0.3333)
print("模型构建完毕")
print(myAlexNet)

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(myAlexNet,trainloader,20,120,loss_func,optimizer)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

print("测试集最终概率：")
accuracy_test(myAlexNet, testloader, loss_func)

np.save("X_32_AlexNet_512_2",X_3_32_2)
np.save("train_loss_32_AlexNet_512_2",train_loss_3_32_2)
np.save("valid_loss_32_AlexNet_512_2",valid_loss_3_32_2)
np.save("valid_accc_32_AlexNet_512_2",valid_accc_3_32_2)

path_state_dict = "./model_state_dict_32_AlexNet_512_2.pkl"

net_state_dict = myAlexNet.state_dict()
torch.save(net_state_dict, path_state_dict)

print("---")

myAlexNet=myAlexNet2()
loss_func=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.RMSprop(myAlexNet.parameters(),lr=0.0001)
X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]
valid_accc_3_32.append(0.3333)
print("模型构建完毕")
print(myAlexNet)

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(myAlexNet,trainloader,20,120,loss_func,optimizer)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

print("测试集最终概率：")
accuracy_test(myAlexNet, testloader, loss_func)

np.save("X_32_AlexNet_1024_2",X_3_32_2)
np.save("train_loss_32_AlexNet_1024_2",train_loss_3_32_2)
np.save("valid_loss_32_AlexNet_1024_2",valid_loss_3_32_2)
np.save("valid_accc_32_AlexNet_1024_2",valid_accc_3_32_2)

path_state_dict = "./model_state_dict_32_AlexNet_1024_2.pkl"

net_state_dict = myAlexNet.state_dict()
torch.save(net_state_dict, path_state_dict)

print("---")

myAlexNet=myAlexNet3()
loss_func=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.RMSprop(myAlexNet.parameters(),lr=0.0001)
X_3_32=[]
train_loss_3_32=[]
valid_loss_3_32=[]
valid_accc_3_32=[]
valid_accc_3_32.append(0.3333)
print("模型构建完毕")
print(myAlexNet)

# print("测试集初始概率：")
# accuracy_test(myAlexNet, testloader, loss_func)

deep_learning(myAlexNet,trainloader,20,120,loss_func,optimizer)

X_3_32_2=np.array(X_3_32)
train_loss_3_32_2=np.array(train_loss_3_32)
valid_loss_3_32_2=np.array(valid_loss_3_32)
valid_accc_3_32_2=np.array(valid_accc_3_32)

print("测试集最终概率：")
accuracy_test(myAlexNet, testloader, loss_func)

np.save("X_32_AlexNet_1536_2",X_3_32_2)
np.save("train_loss_32_AlexNet_1536_2",train_loss_3_32_2)
np.save("valid_loss_32_AlexNet_1536_2",valid_loss_3_32_2)
np.save("valid_accc_32_AlexNet_1536_2",valid_accc_3_32_2)

path_state_dict = "./model_state_dict_32_AlexNet_1536_2.pkl"

net_state_dict = myAlexNet.state_dict()
torch.save(net_state_dict, path_state_dict)
