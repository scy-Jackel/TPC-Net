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
import shutil

res=[]
path="2D_img/2-over"
path2="2D_img/2-over-c"
with os.scandir(path) as it:
    for iii in it:
        if(iii.name[-6]=='p'):
            tmp=str(iii.name[0:-12])+str(iii.name[-9:-4]+".png")
            print(tmp)
            res.append(tmp)
        else:
            tmp=str(iii.name[0:-7]) + ".png"
            print(tmp)
            res.append(tmp)
with os.scandir(path2) as it:
    for iii in it:
        tmp=str(iii.name[0:-10])+str(iii.name[-7:-4]+".png")
        print(tmp)
        res.append(tmp)

ttt=np.array(res)

print("结束")

path="train/2"
path_2="train2/2"
path_3="train2/3"
with os.scandir(path) as it:
    for iii in it:
        print(iii.path)
        if(res.__contains__(iii.name)):
            shutil.copy(iii.path,path_2)
        else:
            shutil.copy(iii.path,path_3)

path="valid/2"
path_2="valid2/2"
path_3="valid2/3"
with os.scandir(path) as it:
    for iii in it:
        print(iii.path)
        if(res.__contains__(iii.name)):
            shutil.copy(iii.path,path_2)
        else:
            shutil.copy(iii.path,path_3)

path="test/2"
path_2="test2/2"
path_3="test2/3"
with os.scandir(path) as it:
    for iii in it:
        print(iii.path)
        if(res.__contains__(iii.name)):
            shutil.copy(iii.path,path_2)
        else:
            shutil.copy(iii.path,path_3)



