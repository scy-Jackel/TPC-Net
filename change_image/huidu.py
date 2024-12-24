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
import math
huidu0=[]
huidu1=[]
huidu2=[]
huidu3=[]
huidu4=[]
huidu5=[]
huidu6=[]
huidu7=[]
huidu8=[]

huidu02=[]
huidu12=[]
huidu22=[]
huidu32=[]
huidu42=[]
huidu52=[]
huidu62=[]
huidu72=[]
huidu82=[]
path1="xiujian/tiff"
path2="xiujian/tiff-xiujian"
with os.scandir(path1) as it:
    for iii in it:
        tiff=tifffile.imread(iii.path)
        for z in range(26,43):
            for x in range(16,33):
                for y in range(16,33):
                    tmp=math.sqrt((z-34)**2+(x-24)**2+(y-24)**2)
                    if (tmp <= 8):
                        huidu8.append(tiff[z][x][y][0])
                    if (tmp <= 7):
                        huidu7.append(tiff[z][x][y][0])
                    if (tmp <= 6):
                        huidu6.append(tiff[z][x][y][0])
                    if (tmp <= 5):
                        huidu5.append(tiff[z][x][y][0])
                    if (tmp <= 4):
                        huidu4.append(tiff[z][x][y][0])
                    if (tmp <= 3):
                        huidu3.append(tiff[z][x][y][0])
                    if (tmp <= 2):
                        huidu2.append(tiff[z][x][y][0])
                    if (tmp <= 1):
                        huidu1.append(tiff[z][x][y][0])
                    if (tmp <= 0):
                        huidu0.append(tiff[z][x][y][0])
        print(iii.path)
with os.scandir(path2) as it:
    for iii in it:
        tiff=tifffile.imread(iii.path)
        for z in range(26,43):
            for x in range(16,33):
                for y in range(16,33):
                    tmp=math.sqrt((z-34)**2+(x-24)**2+(y-24)**2)
                    if (tmp <= 8):
                        huidu82.append(tiff[z][x][y][0])
                    if (tmp <= 7):
                        huidu72.append(tiff[z][x][y][0])
                    if (tmp <= 6):
                        huidu62.append(tiff[z][x][y][0])
                    if (tmp <= 5):
                        huidu52.append(tiff[z][x][y][0])
                    if (tmp <= 4):
                        huidu42.append(tiff[z][x][y][0])
                    if (tmp <= 3):
                        huidu32.append(tiff[z][x][y][0])
                    if (tmp <= 2):
                        huidu22.append(tiff[z][x][y][0])
                    if (tmp <= 1):
                        huidu12.append(tiff[z][x][y][0])
                    if (tmp <= 0):
                        huidu02.append(tiff[z][x][y][0])
        print(iii.path)

huidu0=np.array(huidu0)
huidu1=np.array(huidu1)
huidu2=np.array(huidu2)
huidu3=np.array(huidu3)
huidu4=np.array(huidu4)
huidu5=np.array(huidu5)
huidu6=np.array(huidu6)
huidu7=np.array(huidu7)
huidu8=np.array(huidu8)


huidu02=np.array(huidu02)
huidu12=np.array(huidu12)
huidu22=np.array(huidu22)
huidu32=np.array(huidu32)
huidu42=np.array(huidu42)
huidu52=np.array(huidu52)
huidu62=np.array(huidu62)
huidu72=np.array(huidu72)
huidu82=np.array(huidu82)

print(huidu8.shape)

print(huidu0.mean())
print(huidu02.mean())
print(huidu1.mean())
print(huidu12.mean())
print(huidu2.mean())
print(huidu22.mean())
print(huidu3.mean())
print(huidu32.mean())
print(huidu4.mean())
print(huidu42.mean())
print(huidu5.mean())
print(huidu52.mean())
print(huidu6.mean())
print(huidu62.mean())
print(huidu7.mean())
print(huidu72.mean())
print(huidu8.mean())
print(huidu82.mean())

np.save("hudu0",huidu0)
np.save("hudu02",huidu02)
np.save("hudu1",huidu1)
np.save("hudu12",huidu12)
np.save("hudu2",huidu2)
np.save("hudu22",huidu22)
np.save("hudu3",huidu3)
np.save("hudu32",huidu32)
np.save("hudu4",huidu4)
np.save("hudu42",huidu42)
np.save("hudu5",huidu5)
np.save("hudu52",huidu52)
np.save("hudu6",huidu6)
np.save("hudu62",huidu62)
np.save("hudu7",huidu7)
np.save("hudu72",huidu72)
np.save("hudu8",huidu8)
np.save("hudu82",huidu82)

