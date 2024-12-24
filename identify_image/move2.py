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

path1="/home/gaoruichen/IMG_final/train2/2-c"
path2="/home/gaoruichen/IMG_final/train2/2"
path3="/home/gaoruichen/IMG_final/train2/3-c"
path4="/home/gaoruichen/IMG_final/train2/3"

with os.scandir(path1) as it:
    for iii in it:
        shutil.move(iii.path,path2)
        print(iii.path)

with os.scandir(path3) as it:
    final=0
    for iii in it:
        if(final>4732):
            break
        shutil.move(iii.path,path4)
        final+=1
        print(iii.path)