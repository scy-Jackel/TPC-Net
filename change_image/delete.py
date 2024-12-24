import os
import shutil
import os
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import numpy as np
import tifffile

path="/home/gaoruichen/IMG_final/train2/2"
with os.scandir(path) as it:
    for iii in it:
        strstr="copy"
        if(strstr in iii.name):
            os.remove(iii.path)
