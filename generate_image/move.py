import os
import shutil
import os
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
from libtiff import TIFF
import tifffile
import os
# path1="data_2D/1-1-xy.tif"
# path2="data_2D/1-ok"
# shutil.move(path1, path2)
# path=r"data_o/1-ok"
# with os.scandir(path) as it:
#     for iii in it:
#         name1 = "data_2D/" + iii.name[0:-4] + "-xy.tif"
#         name2 = "data_2D/" + iii.name[0:-4] + "-zx.tif"
#         name3 = "data_2D/" + iii.name[0:-4] + "-zy.tif"
#         shutil.move(name1,"data_2D/1-ok")
#         shutil.move(name2, "data_2D/1-ok")
#         shutil.move(name3, "data_2D/1-ok")
#         print(iii.name)
path1="D:/valid2/1"
path2="D:/valid2/1-2D"
path3="D:/valid2/1-3D"
with os.scandir(path1) as it:
    for iiiii in it:
        img = Image.open(iiiii.path)
        img2 = np.array(img)
        img3=img2.reshape((50,50,50,3))
        tifffile.imwrite(path3+"/"+iiiii.name,img3)
        img=img3

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] >0) & (img[z][x][y][2] == 0):
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] >0) & (img[z][x][y][2] == 0):
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] >0) & (img[z][x][y][2] == 0):
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = path2+"/" + iiiii.name[0:-4] + "-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = path2+"/" + iiiii.name[0:-4] + "-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = path2+"/" + iiiii.name[0:-4] + "-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)
        print(iiiii.name)

