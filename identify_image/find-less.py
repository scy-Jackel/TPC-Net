from torch import optim
from torch.autograd import Variable
from torchvision import datasets,transforms,models
import seaborn as sb
import os
import tifffile
from PIL import Image,ImageDraw
import shutil
import numpy as np
res=[]
path="/home/gaoruichen/IMG_final/2D_img/3-less"
with os.scandir(path) as it:
    for iii in it:
        tmp=iii.name
        if tmp[-6]=='p':
            ttt=tmp[0:-12]+".copy.tif"
            res.append(ttt)

    res=np.array(res)
    np.save("ttt2",res)