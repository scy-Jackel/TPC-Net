import numpy as np
import matplotlib.pyplot as plt

# x=['0',
#    '1-4','1-8','1','2-4','2-8','2',
#    '3-4','3-8','3','4-4','4-8','4',
#    '5-4','5-8','5','6-4','6-8','6',
#    '7-4','7-8','7','8-4','8-8','8',
#    '9-4','9-8','9','10-4','10-8','10',
#    '11-4', '11-8', '11', '12-4', '12-8', '12',
#    '13-4', '13-8', '13', '14-4', '14-8', '14',
#    '15-4', '15-8', '15', '16-4', '16-8', '16',
#    '17-4', '17-8', '17', '18-4', '18-8', '18',
#    '19-4', '19-8', '19', '20-4', '20-8', '20'
#    ]
x_=np.load("model/X_32_AlexNet_256.npy")
# y1_=np.load("model/valid_accc_32_AlexNet_256_2.npy")
y2_=np.load("valid_loss_effi_s.npy")
y3_=np.load("model/valid_loss_32_AlexNet_1024_2.npy")
y4_=np.load("model/valid_loss_32_AlexNet_512_2.npy")

y5_=np.load("train_loss_effi_s.npy")
y6_=np.load("model/train_loss_32_AlexNet_1024_2.npy")
y7_=np.load("model/train_loss_32_AlexNet_512_2.npy")
x=[]
# y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
y7=[]
# # x.append('0')
# # y1.append(0.3333)
# # y2.append(0.3333)
# # y3.append(0.3333)
for i in range(x_.shape[0]):
        if(i%2==0):
                x.append(x_[i])
# # flag=1
# # for i in range(y1_.shape[0]):
# #     if(i==0):
# #         y1.append(y1_[i])
# #     elif(i==y1_.shape[0]-1):
# #         # y1.append(y1_[i])
# #         continue
# #     else:
# #         if(flag==1):
# #             y1.append(max(y1_[i],y1_[i+1]))
# #             flag=0
# #         else:
# #             flag=1
flag=1
for i in range(y2_.shape[0]-1):
        if(flag==1):
            y2.append(min(y2_[i], y2_[i + 1]))
            flag=0
        else:
            flag=1
flag=1
for i in range(y3_.shape[0]-6):
        if(flag==1):
            y3.append(min(y3_[i], y3_[i + 1]))
            flag=0
        else:
            flag=1
flag=1
for i in range(y4_.shape[0]-6):
        if(flag==1):
            y4.append(min(y4_[i], y4_[i + 1]))
            flag=0
        else:
            flag=1

flag=1
for i in range(y5_.shape[0]):
        if(flag==1):
            y5.append(min(y5_[i], y5_[i + 1]))
            flag=0
        else:
            flag=1
flag=1
for i in range(y6_.shape[0]-6):
        if(flag==1):
            y6.append(min(y6_[i], y6_[i + 1]))
            flag=0
        else:
            flag=1
flag=1
for i in range(y7_.shape[0]-6):
        if(flag==1):
            y7.append(min(y7_[i], y7_[i + 1]))
            flag=0
        else:
            flag=1
# x.append('last')
# y1.append(0.9097)
# y2.append(0.9009)
# y3.append(0.8938)
# y1_t=np.load("model/valid_accc_32_AlexNet_2.npy")
# y1=[]
# y1.append(y1_t[0])
# for i in range(60):
#     max=0
#     for j in range(4):
#         if(y1_t[1+j+i*4]>max):
#             max=y1_t[1+j+i*4]
#     y1.append(max)
#
# y2_t=np.load("model/valid_accc_16_AlexNet_2.npy")
# y2=[]
# y2.append(0.345)
# for i in range(60):
#     max=0
#     for j in range(4):
#         if(y2_t[j+i*4]>max):
#             max=y2_t[1+j+i*4]
#     y2.append(max)
#
# y3_t=np.load("model/valid_accc_64_AlexNet_2.npy")
# y3=[]
# y3.append(0.337)
# for i in range(60):
#     max=0
#     for j in range(4):
#         if(y3_t[j+i*4]>max):
#             max=y3_t[1+j+i*4]
#     y3.append(max)
# x=x[401:]
# y1=y1[401:600]
plt.figure(figsize=(16, 9), dpi=100)
# plt.plot(x, y1, c='red',label="bc=256")
plt.plot(x, y2, c=(118.0/255,218.0/255,145.0/255),label="efficientnet_valid")
# plt.plot(x, y3, c=(248.0/255,149.0/255,136.0/255),label="ouAlexNet1/2_valid")
# plt.plot(x, y4, c=(248.0/255,203.0/255,127.0/255),label="ouAlexNet1/4_valid")

plt.plot(x, y5, '-.',c=(118.0/255,218.0/255,145.0/255),label="efficientnet_train")
# plt.plot(x, y6, '-.',c=(248.0/255,149.0/255,136.0/255),label="ouAlexNet1/2_train")
# plt.plot(x, y7, '-.',c=(248.0/255,203.0/255,127.0/255),label="ouAlexNet1/4_train")
plt.legend(loc='best')
# plt.scatter(x, y1, c='red')
# plt.scatter(x, y2, c='green')
# plt.scatter(x, y3, c='blue')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("epoch", fontdict={'size': 24})
plt.ylabel("loss", fontdict={'size': 24})
plt.xticks(np.arange(0, 61, step=3))
# for a, b in zip(x, y1):
#     plt.text(a, round(b,3), round(b,3), ha='center', va='bottom', fontsize=8, c='red')
# for a, b in zip(x, y2):
#     plt.text(a, round(b,3), round(b,3), ha='center', va='bottom', fontsize=8, c='green')
# for a, b in zip(x, y3):
#     plt.text(a, round(b,3), round(b,3), ha='center', va='bottom', fontsize=8, c='blue')
plt.show()
# y1=np.array(y1)
# y2=np.array(y2)
# y3=np.array(y3)
# np.save("model/valid_accc_32_AlexNet_256",y1)
# np.save("model/valid_accc_32_AlexNet_512",y2)
# np.save("model/valid_accc_32_AlexNet_1024",y3)
print(y2)
print(y3)
print(y4)
print(y2[39])
print(y3[39])
print(y4[39])