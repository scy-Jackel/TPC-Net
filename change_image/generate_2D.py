import os
import shutil
import os
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import numpy as np
import tifffile

path="test"
with os.scandir(path) as it:
    for iii in it:
        imm=Image.open(iii.path)
        tiff=np.array(imm)
        print(tiff)
        tiff1=tiff.reshape((50,50,50,3))
        ppp3="test_3D/"+iii.name[0:-4]+".tif"
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

        name8_xy = "test_2D/"+iii.name[0:-4]+"-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "test_2D/"+iii.name[0:-4]+"-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "test_2D/"+iii.name[0:-4]+"-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)




