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

# path="ttt"
# ress=np.zeros((32,50,50,50,3))
# ress2=np.zeros((32,50,2500,3))
# with os.scandir(path) as it:
#     num=0
#     for iiiii in it:
#         tiff = tifffile.imread(iiiii.path)
#         ress[num]=tiff
#         res=tiff.reshape(50,2500,3)
#         ress2[num]=res
#         img = Image.fromarray(np.uint8(res))
#         img.save("ttt2/1/" + iiiii.name[0:-4] + ".png")
#         print(num)
#         num += 1
# print("图像生成完毕")
# ress2=torch.tensor(ress2)
# ress2=torch.transpose(ress2,2,3)
# ress2=torch.transpose(ress2,1,2)
# ress2=ress2.reshape((32,3,50,50,50))
# print("张量转换完毕")
#
# for a in range(32):
#     for b in range(3):
#         for c in range(50):
#             for d in range(50):
#                 for e in range(50):
#                     print(ress2[a][b][c][d][e])
#                     print("...")
#                     print(ress[a][c][d][e][b])
#
#
path="ttt2/1"
with os.scandir(path) as it:
    for iii in it:
        pic1=plt.imread(iii.path)
        pic2=plt.imread("train/1/"+iii.name)
        print("pic1")
        print(pic1)
        print("pic2")
        print(pic2)
        print("pic1-pic2")
        print(pic1-pic2)

