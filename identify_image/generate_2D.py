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
import shutil
# tiff2=tifffile.imread("xiujian/tiff-xiujian2-rgb/3-15443-2-2.tif")
# tiff3=tifffile.imread("xiujian/tiff-xiujian2-rgb/3-15443-3-2.tif")
# res2=tiff2.reshape(50,2500,3)
# res3=tiff3.reshape(50,2500,3)
# img2 = Image.fromarray(np.uint8(res2))
# img3 = Image.fromarray(np.uint8(res3))
# img2.save("2.png")
# img3.save("3.png")
# path="xiujian/tiff-xiujian"
# path2="xiujian/tiff-xiujian2"
# path3="xiujian/tiff-xiujian-rgb"
# path4="xiujian/tiff-xiujian2-rgb"
#
path5="lesstif/lesstif_rgb"
path6="lesstif/lesstif_ping"
# path7="xiujian/tiff-xiujian-rgb"
# path8="xiujian/tiff-xiujian-rgb-ping"
# with os.scandir(path) as it:
#     for iii in it:
#         name=iii.name
#         if(int(name[-5])==2)&(int(name[-7])==1):
#             shutil.move(iii.path, path3)
#         elif (int(name[-5]) == 1) & (int(name[-7]) > 1):
#             shutil.move(iii.path, path2)
#         elif (int(name[-5]) == 2) & (int(name[-7]) > 1):
#             shutil.move(iii.path, path4)

# tiff=tifffile.imread("lunwen/3-1-2.tif")
# res=tiff.reshape(50,2500,3)
# img = Image.fromarray(np.uint8(res))
# img.save("lunwen/3-1-2.png")

with os.scandir(path5) as it:
    for iii in it:
        tiff = tifffile.imread(iii.path)
        res=tiff.reshape(50,2500,3)
        img = Image.fromarray(np.uint8(res))
        name=path6+"/"+iii.name[0:-4]+".png"
        img.save(name)
# with os.scandir(path7) as it:
#     for iii in it:
#         tiff = tifffile.imread(iii.path)
#         res=tiff.reshape(50,2500,3)
#         img = Image.fromarray(np.uint8(res))
#         name=path8+"/"+iii.name[0:-4]+".png"
#         img.save(name)