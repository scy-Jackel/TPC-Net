import cv2 as cv
import numpy as np
import random
import tifffile
from PIL import Image,ImageDraw
import os

path1="valid2/2"
path2="valid2/3"
path3="test2/2"
path4="test2/3"
with os.scandir(path1) as it:
    final_num=0
    for iiiii in it:
        if(final_num>322):
            break
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="valid2/2-c/"+iiiii.name[0:-4]+"(m).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

with os.scandir(path2) as it:
    final_num=0
    for iiiii in it:
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="valid2/3-c/"+iiiii.name[0:-4]+"(m1).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

with os.scandir(path2) as it:
    final_num=1
    for iiiii in it:
        if(final_num>135):
            break
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="valid2/3-c/"+iiiii.name[0:-4]+"(m2).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

with os.scandir(path3) as it:
    final_num=0
    for iiiii in it:
        if(final_num>307):
            break
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/2-over-c2/"+iiiii.name[0:-4]+"(m)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="test2/2-c/"+iiiii.name[0:-4]+"(m).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

with os.scandir(path4) as it:
    final_num=0
    for iiiii in it:
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m1)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="test2/3-c/"+iiiii.name[0:-4]+"(m1).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

with os.scandir(path2) as it:
    final_num=1
    for iiiii in it:
        if(final_num>165):
            break
        imm=Image.open(iiiii.path)
        img=np.array(imm)
        # print(img)
        img=img.reshape(50,50,50,3)
        img_z, img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # for iiii in range(2):
        M = cv.getRotationMatrix2D((img_w/2-1,img_h/2-1), (final_num%3+1)*90, 1)
        rrr = random.random()/2.0 + 1
        #	进行旋转
        img_tmp=np.zeros((50,50,50,3))
        for iii in range(img_z):
            rotated = cv.warpAffine(img[iii], M, (img_w, img_h))
            blank = np.zeros([img_h, img_w, img_c], img.dtype)
            dst = cv.addWeighted(rotated, rrr, blank, 1-rrr, 0)
            dst[dst>255]=255
            img_tmp[iii] = dst

        img=img_tmp

        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                flag = 1
                for z in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = max(img[z][x][y][1],img_res_xy[x][y][1])
                        img_res_xy[x][y][2] = 0
                    elif (img[z][x][y][0] > img_res_xy[x][y][0]) & (flag==1):
                        img_res_xy[x][y][0] = img[z][x][y][0]
                        img_res_xy[x][y][1] = img[z][x][y][1]
                        img_res_xy[x][y][2] = img[z][x][y][2]
        for z in range(50):
            for y in range(50):
                flag=1
                for x in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        falg=0
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = max(img[z][x][y][1],img_res_zy[z][y][1])
                        img_res_zy[z][y][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zy[z][y][0]) & (flag==1):
                        img_res_zy[z][y][0] = img[z][x][y][0]
                        img_res_zy[z][y][1] = img[z][x][y][1]
                        img_res_zy[z][y][2] = img[z][x][y][2]
        for z in range(50):
            for x in range(50):
                flag=1
                for y in range(50):
                    if (img[z][x][y][0] == 0) & (img[z][x][y][1] != 0) & (img[z][x][y][2] == 0):
                        flag=0
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = max(img[z][x][y][1],img_res_zx[z][x][1])
                        img_res_zx[z][x][2] = 0
                        break
                    elif (img[z][x][y][0] > img_res_zx[z][x][0]) & (flag==1):
                        img_res_zx[z][x][0] = img[z][x][y][0]
                        img_res_zx[z][x][1] = img[z][x][y][1]
                        img_res_zx[z][x][2] = img[z][x][y][2]

        name8_xy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = "2D_img/3-less-c2/"+iiiii.name[0:-4]+"(m2)-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        img = img.reshape(50, 2500, 3)
        imggg=Image.fromarray(np.uint8(img))
        name_final="test2/3-c/"+iiiii.name[0:-4]+"(m2).png"
        imggg.save(name_final)

        print(iiiii.name)
        final_num+=1

