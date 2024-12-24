import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# y1_=np.load("model/valid_accc_32_AlexNet_256_2.npy")
y2_=np.load("train_loss_effi_s.npy")
y3_=np.load("valid_loss_effi_s.npy")

y2=[]
y3=[]

# for i in range(y1_.shape[0]):
#     if(i==0):
#         y1.append(y1_[i])
#     elif(i==y1_.shape[0]-1):
#         # y1.append(y1_[i])
#         continue
#     else:
#         if(flag==1):
#             y1.append(max(y1_[i],y1_[i+1]))
#             flag=0
#         else:
#             flag=1
flag=1
for i in range(y2_.shape[0]):
    # if (i == 0):
    #     y2.append(y2_[i])
    # elif (i == y2_.shape[0] - 1):
    #     y2.append(y2_[i])
    #     # continue
    # else:
        if(flag==1):
            y2.append(min(y2_[i], y2_[i + 1]))
            flag=0
        else:
            flag=1
flag=1
for i in range(y3_.shape[0]-1):
    # if (i == 0):
    #     y3.append(y3_[i])
    # elif (i == y3_.shape[0] - 1):
    #     y3.append(y3_[i])
    #     # continue
    # else:
        if(flag==1):
            y3.append(min(y3_[i], y3_[i + 1]))
            flag=0
        else:
            flag=1
# flag=1
# for i in range(y4_.shape[0]):
#     if (i == 0):
#         y4.append(y4_[i])
#     elif (i == y4_.shape[0] - 1):
#         y4.append(y4_[i])
#         # continue
#     else:
#         if(flag==1):
#             y4.append(max(y4_[i], y4_[i + 1]))
#             flag=0
#         else:
#             flag=1

data=[]
for i in range(len(y2)):
    data.append([y2[i],y3[i]])

df=pd.DataFrame(data,columns=['ouAlexNet','EfficientNet'],dtype=float)

print(df)
df.to_excel("999.xlsx")