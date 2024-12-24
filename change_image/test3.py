import cv2 as cv
import numpy as np
import random
import tifffile
from PIL import Image,ImageDraw
import os

path="data7-o-new"
with os.scandir(path) as it:
    for iiiii in it:
        img = tifffile.imread(iiiii.path)
        res = img.reshape(50, 2500, 3)
        imgg = Image.fromarray(np.uint8(res))
        imgg.save("data7-o-big/" + iiiii.name[0:-4] + ".png")
        print(iiiii.name)