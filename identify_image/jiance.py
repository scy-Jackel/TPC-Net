import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import math
import seaborn as sb
import torch.nn as tnn
import os
import tifffile
from sklearn.metrics import confusion_matrix

# y_pred_ouAlexNet = [] # ['2','2','3','1','4'] # 类似的格式
# y_true_ouAlexNet = [] # ['0','1','2','3','4'] # 类似的格式
# y_pred_AlexNet = [] # ['2','2','3','1','4'] # 类似的格式
# y_true_AlexNet = [] # ['0','1','2','3','4'] # 类似的格式
# y_pred_myAlexNet = [] # ['2','2','3','1','4'] # 类似的格式
# y_true_myAlexNet = [] # ['0','1','2','3','4'] # 类似的格式

train_dir = 'train2'
valid_dir = 'test5'
test_dir = 'test6'
right1=[]
right2=[]
over1=[]
over2=[]
less1=[]
less2=[]
res1=[]
res2=[]

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
            nn.Linear(2048, 1024),
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

class myAlexNet2(nn.Module):
    def __init__(self, num_classes=3):
        super(myAlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),

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

            nn.Conv3d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(),

            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(1024, 512),
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

# def predict(img, model2, topk=3):
#     ''' 预测图片.
#     '''
#     model2.eval()
#     img = img.unsqueeze(0)  # 将图片多增加一维
#     result = model2(img).topk(topk)
#     probs = []
#     classes = []
#     a = result[0]  # 返回TOPK函数截取的排名前列的结果列表a
#     b = result[1].tolist()  # 返回TOPK函数截取的排名前列的概率索引列表b
#     # print(a)
#     # print(b)
#
#     for i in a[0]:
#         probs.append(torch.exp(i).tolist())  # 将结果转化为实际概率
#     for n in b[0]:
#         classes.append(str(n + 1))  # 将索引转化为实际编号
#     print(classes)
#     # print(probs)
#     sum=probs[0]+probs[1]+probs[2]
#     probs[0]=probs[0]/sum
#     probs[1] = probs[1] / sum
#     probs[2] = probs[2] / sum
#     print(probs)
#     return classes[0]

path_state_dict = "model_state_dict_32_AlexNet_xiugai_32.pkl"
state_dict_load = torch.load(path_state_dict,map_location=torch.device('cpu'))
net_new = myAlexNet1()

path_state_dict2 = "model_state_dict_32_AlexNet_xiugai_22.pkl"
state_dict_load2 = torch.load(path_state_dict2,map_location=torch.device('cpu'))
net_new2 = myAlexNet2()

path_state_dict3 = "model_state_dict_32_AlexNet_xiugai_42.pkl"
state_dict_load3 = torch.load(path_state_dict3,map_location=torch.device('cpu'))
net_new3 = myAlexNet3()

net_new.load_state_dict(state_dict_load)
print("加载后: ", net_new.features[0].weight[0, ...])
net_new2.load_state_dict(state_dict_load2)
print("加载后: ", net_new2.features[0].weight[0, ...])
net_new3.load_state_dict(state_dict_load3)
print("加载后: ", net_new3.features[0].weight[0, ...])
print("模型加载完毕")
# print("加载后: ", net_new.features[0].weight[0, ...])
def accuracy_test1(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    nnn=0
    model.to('cpu')  # 将模型放入GPU计算，能极大加快运算速度
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    running_loss = 0.0
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            tmp = images.shape[0]
            images = images.reshape(tmp, 3, 50, 50, 50)
            images, labels = images.to('cpu'),labels.to('cpu')

            outputs = model(images)
            # loss = criterion(outputs, labels)
            # running_loss += loss.item()
            nnn+=1

            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for iii in range(labels.size(0)):
                print(predicted[iii].item())
                res1.append(predicted[iii].item())
                tmp0 = math.exp(outputs.data[iii][0].item())
                tmp1 = math.exp(outputs.data[iii][1].item())
                tmp2 = math.exp(outputs.data[iii][2].item())
                tmp=tmp0+tmp1+tmp2
                tmp0 = tmp0 / tmp
                tmp1 = tmp1 / tmp
                tmp2 = tmp2 / tmp
                right1.append(tmp0)
                over1.append(tmp1)
                less1.append(tmp2)
                print(tmp0)
                print(tmp1)
                print(tmp2)

def accuracy_test2(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    nnn=0
    model.to('cpu')  # 将模型放入GPU计算，能极大加快运算速度
    # model = tnn.DataParallel(model)
    # model = model.cuda()
    running_loss = 0.0
    with torch.no_grad():  # 使用验证集时关闭梯度计算
        for data in dataloader:
            images, labels = data
            tmp = images.shape[0]
            images = images.reshape(tmp, 3, 50, 50, 50)
            images, labels = images.to('cpu'),labels.to('cpu')

            outputs = model(images)
            # loss = criterion(outputs, labels)
            # running_loss += loss.item()
            nnn+=1

            _, predicted = torch.max(outputs.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for iii in range(labels.size(0)):
                print(predicted[iii].item())
                res2.append(predicted[iii].item())
                tmp0 = math.exp(outputs.data[iii][0].item())
                tmp1 = math.exp(outputs.data[iii][1].item())
                tmp2 = math.exp(outputs.data[iii][2].item())
                tmp=tmp0+tmp1+tmp2
                tmp0 = tmp0 / tmp
                tmp1 = tmp1 / tmp
                tmp2 = tmp2 / tmp
                right2.append(tmp0)
                over2.append(tmp1)
                less2.append(tmp2)


# tiff=tifffile.imread("lunwen/3-1-2.tif")
# res=tiff.reshape(50,2500,3)
# img = Image.fromarray(np.uint8(res))
# img.save("lunwen/3-1-2.png")

# accuracy_test1(net_new,validloader)
# accuracy_test1(net_new,testloader)
#
# accuracy_test2(net_new2,validloader)
# accuracy_test2(net_new2,testloader)
#
# accuracy_test3(net_new3,validloader)
# accuracy_test3(net_new3,testloader)
accuracy_test1(net_new,validloader)
accuracy_test2(net_new,testloader)
# accuracy_test2(net_new,testloader)

np.save("res1",res1)
np.save("res2",res2)
np.save("right1",right1)
np.save("right2",right2)
np.save("over1",over1)
np.save("over2",over2)
np.save("less1",less1)
np.save("less2",less2)

