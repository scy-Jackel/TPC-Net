import os
import shutil
import os
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import numpy as np
import tifffile
import random

tiff=tifffile.imread("data_3D_final/2-87.tif")
for a in range(-2, 3):
    for b in range(-2, 3):
        for c in range(-2, 3):
            tmp=random.random()*50+100
            tiff[34+a][24+b][24+c]=(tmp,tmp,tmp)

tifffile.imsave("data_3D_final/2-87.tif",tiff)
tiff1=tiff
img_res_xy = np.zeros((50, 50, 3))
img_res_zy = np.zeros((50, 50, 3))
img_res_zx = np.zeros((50, 50, 3))
for x in range(50):
    for y in range(50):
        for z in range(50):
            if (tiff1[z][x][y][0] == 0) & (tiff1[z][x][y][1] == 100) & (tiff1[z][x][y][2] == 0):
                img_res_xy[x][y][0] = 0
                img_res_xy[x][y][1] = 100
                img_res_xy[x][y][2] = 0
                break
            elif (tiff1[z][x][y][0] > img_res_xy[x][y][0]):
                img_res_xy[x][y][0] = tiff1[z][x][y][0]
                img_res_xy[x][y][1] = tiff1[z][x][y][1]
                img_res_xy[x][y][2] = tiff1[z][x][y][2]
for z in range(50):
    for y in range(50):
        for x in range(50):
            if (tiff1[z][x][y][0] == 0) & (tiff1[z][x][y][1] == 100) & (tiff1[z][x][y][2] == 0):
                img_res_zy[z][y][0] = 0
                img_res_zy[z][y][1] = 100
                img_res_zy[z][y][2] = 0
                break
            elif (tiff1[z][x][y][0] > img_res_zy[z][y][0]):
                img_res_zy[z][y][0] = tiff1[z][x][y][0]
                img_res_zy[z][y][1] = tiff1[z][x][y][1]
                img_res_zy[z][y][2] = tiff1[z][x][y][2]
for z in range(50):
    for x in range(50):
        for y in range(50):
            if (tiff1[z][x][y][0] == 0) & (tiff1[z][x][y][1] == 100) & (tiff1[z][x][y][2] == 0):
                img_res_zx[z][x][0] = 0
                img_res_zx[z][x][1] = 100
                img_res_zx[z][x][2] = 0
                break
            elif (tiff1[z][x][y][0] > img_res_zx[z][x][0]):
                img_res_zx[z][x][0] = tiff1[z][x][y][0]
                img_res_zx[z][x][1] = tiff1[z][x][y][1]
                img_res_zx[z][x][2] = tiff1[z][x][y][2]
name8_xy = "data_2D_final/2-87-xy.tif"
img_xy = Image.fromarray(np.uint8(img_res_xy))
img_xy.save(name8_xy)
name8_zy = "data_2D_final/2-87-zy.tif"
img_zy = Image.fromarray(np.uint8(img_res_zy))
img_zy.save(name8_zy)
name8_zx = "data_2D_final/2-87-zx.tif"
img_zx = Image.fromarray(np.uint8(img_res_zx))
img_zx.save(name8_zx)
