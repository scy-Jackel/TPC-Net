import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
from libtiff import TIFF
import tifffile
import os

path1=r"data6-3"
path2=r"data7-3"
with os.scandir(path1) as it:
    for iii in it:
        ppp1=path1+"/"+iii.name
        tiff1=tifffile.imread(ppp1)
        ppp2=path2+"/"+iii.name
        tiff2=tifffile.imread(ppp2)
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    if (tiff2[i][j][k][0]==0)&(tiff2[i][j][k][1]==100)&(tiff2[i][j][k][2]==0):
                        if i+10<50:
                            tiff1[i+10][j][k]=(0,100,0)


        ppp3="data_o/"+iii.name
        tifffile.imsave(ppp3,tiff1)

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

        name8_xy = "data_2D/" + iii.name[0:-4]+"-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "data_2D/" + iii.name[0:-4]+"-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "data_2D/" + iii.name[0:-4]+"-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)
        print(iii.name)




