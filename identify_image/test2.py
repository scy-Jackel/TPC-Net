import tifffile
import os
import numpy as np
from PIL import Image,ImageDraw
path1=r"train2/1"
path2=r"train/1"
path3=r"train2/2"
path4=r"train/2"

path5=r"valid2/1"
path6=r"valid/1"
path7=r"valid2/2"
path8=r"valid/2"

path09=r"test2/1"
path10=r"test/1"
path11=r"test2/2"
path12=r"test/2"

with os.scandir(path1) as it:
    for iiiii in it:
        tiff=tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path2+"/"+iiiii.name[0:-4]+".png")
        print(iiiii.path)

with os.scandir(path3) as it:
    for iiiii in it:
        tiff = tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path4 + "/" + iiiii.name[0:-4] + ".png")
        print(iiiii.path)

with os.scandir(path5) as it:
    for iiiii in it:
        tiff = tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path6 + "/" + iiiii.name[0:-4] + ".png")
        print(iiiii.path)

with os.scandir(path7) as it:
    for iiiii in it:
        tiff = tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path8 + "/" + iiiii.name[0:-4] + ".png")
        print(iiiii.path)

with os.scandir(path09) as it:
    for iiiii in it:
        tiff = tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path10 + "/" + iiiii.name[0:-4] + ".png")
        print(iiiii.path)

with os.scandir(path11) as it:
    for iiiii in it:
        tiff = tifffile.imread(iiiii.path)
        res = tiff.reshape(50, 2500, 3)
        img = Image.fromarray(np.uint8(res))
        img.save(path12 + "/" + iiiii.name[0:-4] + ".png")
        print(iiiii.path)


