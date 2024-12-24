import os
import shutil
import os
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import numpy as np
import itk
import argparse
import tifffile
from PIL import Image,ImageDraw
import os
import pandas as pd

names=['n','type','x','y','z','radius','parent','seg_id','level','mode','timestamp','TFresindex']
result1=pd.read_table('lesstif/03_new.txt',names=names,index_col='n',sep=' ')
result2=pd.read_table('lesstif/03_new.txt',names=names,sep=' ')
# print(result2)
path1="lesstif/lesstif"
path2="lesstif/lesstif_rgb"
path3="lesstif/tiff3"
path4="lesstif/tiff3_2D"
path5="lesstif/tiff4"
path6="lesstif/tiff4_2D"
path7="lesstif/tiff5"
path8="lesstif/tiff5_2D"
path9="lesstif/tmp"

with os.scandir(path1) as it:
    for iiiii in it:
        print("开始处理"+iiiii.name)
        tiff2=tifffile.imread(iiiii.path)
        # tiff2:原始(50,50,50,3)黑白图像
        tiff1=tifffile.imread(path2+"/"+iiiii.name)
        # tiff1:原始(50,50,50,3)带有标记点（未追踪完成）图像

        #根据标记点所在像素值判断分割阈值
        # res_gray=[]
        # for i in range(50):
        #     for j in range(50):
        #         for k in range(50):
        #             if (tiff1[i][j][k][0] == 0) & (tiff1[i][j][k][1] == 100) & (tiff1[i][j][k][2] == 0):
        #                 res_gray.append(tiff2[i][j][k][0])
        #
        # res_gray=np.array(res_gray)
        # res_gray=np.sort(res_gray)
        # print(res_gray)
        # yuzhi_m=res_gray[int(0.05*res_gray.shape[0])]
        # print(yuzhi_m)
        yuzhi_m=tiff2.mean()+tiff2.var()
        print(yuzhi_m)

        tiff3=tiff2
        # tiff3:图像分割之后(50,50,50,3)图像
        tiff3[tiff3<yuzhi_m]=0
        tiff3[tiff3>=yuzhi_m]=255

        tiff4=np.zeros((50,50,50))
        # tiff4:图像分割之后(50,50,50)图像
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    tiff4[i][j][k]=tiff3[i][j][k][0]

        tifffile.imsave(path3+"/"+iiiii.name,tiff4)
        img_res_xy = np.zeros((50, 50))
        img_res_zy = np.zeros((50, 50))
        img_res_zx = np.zeros((50, 50))
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if (tiff4[z][x][y] > img_res_xy[x][y]):
                        img_res_xy[x][y] = tiff4[z][x][y]
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    if (tiff4[z][x][y] > img_res_zy[z][y]):
                        img_res_zy[z][y] = tiff4[z][x][y]
        for z in range(50):
            for x in range(50):
                for y in range(50):
                    if (tiff4[z][x][y] > img_res_zx[z][x]):
                        img_res_zx[z][x] = tiff4[z][x][y]

        name8_xy = path4+"/"+iiiii.name[0:-4]+"-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = path4+"/"+iiiii.name[0:-4]+"-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = path4+"/"+iiiii.name[0:-4]+"-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        print("分割后图像保存完毕")
        #将分割后图像转成itk格式，并调用3D细化算法
        Dimension = 3
        PixelType = itk.UC
        ImageType = itk.Image[PixelType, Dimension]
        region = itk.ImageRegion[Dimension]()
        start = itk.Index[Dimension]()
        start.Fill(0)
        region.SetIndex(start)
        size = itk.Size[Dimension]()
        size[0] = 50
        size[1] = 50
        size[2] = 50
        region.SetSize(size)
        image = ImageType.New()
        image.SetRegions(region)
        image.Allocate()
        image.FillBuffer(itk.NumericTraits[PixelType].ZeroValue())
        pixelIndex = itk.Index[Dimension]()
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    pixelIndex[0] = k
                    pixelIndex[1] = j
                    pixelIndex[2] = i
                    image.SetPixel(pixelIndex, int(tiff4[i][j][k]))
        # print(image)
        thickness_map = itk.BinaryThinningImageFilter3D.New(image)
        itk.imwrite(thickness_map, path9+"/"+iiiii.name)

        tiff5=tifffile.imread(path9+"/"+iiiii.name)
        # tiff5:3D细化后的(50,50,50)图像
        tiff5=tiff5*255

        tifffile.imsave(path5+"/"+iiiii.name,tiff5)
        img_res_xy = np.zeros((50, 50))
        img_res_zy = np.zeros((50, 50))
        img_res_zx = np.zeros((50, 50))
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if (tiff5[z][x][y] > img_res_xy[x][y]):
                        img_res_xy[x][y] = tiff5[z][x][y]
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    if (tiff5[z][x][y] > img_res_zy[z][y]):
                        img_res_zy[z][y] = tiff5[z][x][y]
        for z in range(50):
            for x in range(50):
                for y in range(50):
                    if (tiff5[z][x][y] > img_res_zx[z][x]):
                        img_res_zx[z][x] = tiff5[z][x][y]

        name8_xy = path6+"/"+iiiii.name[0:-4]+"-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = path6+"/"+iiiii.name[0:-4]+"-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = path6+"/"+iiiii.name[0:-4]+"-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)

        # 补充标记点
        end5=[]
        end1=[]
        flag=-1
        num=0
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    if(tiff5[i][j][k]==255):
                        end5.append((i,j,k))
                    if (tiff1[i][j][k][0] == 0) & (tiff1[i][j][k][1] == 100) & (tiff1[i][j][k][2] == 0):
                        end1.append((i,j,k))
                        if (i==34)&(j==24)&(k==24):
                            flag=num
                        else:
                            num+=1

        end1=np.array(end1)
        iii=end1.shape[0]
        dis1=np.zeros((iii,iii))
        for a in range(iii):
            for b in range(iii):
                if(a==b):
                    dis1[a][b]=7300
                else:
                    dis1[a][b]=(end1[a][0]-end1[b][0])**2+(end1[a][1]-end1[b][1])**2+(end1[a][2]-end1[b][2])**2
        end1_new=[]
        for i in range(iii):
            end1_new.append(list(end1[flag]))
            dis1[:,flag]=7300
            flag= np.argmin(dis1[flag])
        print(end1)
        print(end1_new)

        dis_be=np.zeros((len(end5),3))
        for ii in range(len(end5)):
            for jj in range(3):
                dis_be[ii][jj]=(end5[ii][0]-end1_new[jj][0])**2+(end5[ii][1]-end1_new[jj][1])**2+(end5[ii][2]-end1_new[jj][2])**2

        jud=[]
        for ii in range(len(end5)):
            if(dis_be[ii][0]>dis_be[ii][1]):
                dis_be[ii][0]=dis_be[ii][1]=dis_be[ii][2]=7300
                jud.append(False)
            elif (dis_be[ii][0] > dis_be[ii][2]):
                dis_be[ii][0] = dis_be[ii][1] = dis_be[ii][2] = 7300
                jud.append(False)
            else:
                jud.append(True)

        print(dis_be)
        start=dis_be.argmin(axis=0)[0]
        print(start)

        end5=np.array(end5)
        iii=end5.shape[0]
        dis5=np.zeros((iii,iii))
        for a in range(iii):
            for b in range(iii):
                if(a==b):
                    dis5[a][b]=7300
                elif not ((jud[a])&(jud[b])):
                    dis5[a][b]=7300
                else:
                    dis5[a][b]=(end5[a][0]-end5[b][0])**2+(end5[a][1]-end5[b][1])**2+(end5[a][2]-end5[b][2])**2

        end5_new=[]
        for i in range(iii):
            end5_new.append(list(end5[start]))
            dis5[:,start]=7300
            tmp=start
            start= np.argmin(dis5[start])
            if(dis5[tmp][start]>64):
                break
        print(end5)
        print(end5_new)

        tiff6=tiff1
        # 最终生成的补充之后的(50,50,50,3)图像
        for iii in range(len(end5_new)):
            xxx=end5_new[iii][0]
            yyy=end5_new[iii][1]
            zzz=end5_new[iii][2]
            tiff6[xxx][yyy][zzz]=(0,100,0)

        zzz=end5_new[-1][0]
        print(xxx)
        xxx=end5_new[-1][1]
        print(yyy)
        yyy=end5_new[-1][2]
        print(zzz)
        num=int(iiiii.name[3:8])
        xt=result1.loc[num]['x']
        yt=result1.loc[num]['y']
        zt=result1.loc[num]['z']
        num2=result2['n'].shape[0]
        result2.loc[num2]=[num2+1,0,xt+(xxx-24),yt+(yyy-24),zt+(zzz-34),1,num,0,0,0,0,0]

        tifffile.imsave(path7+"/"+iiiii.name,tiff6)
        img_res_xy = np.zeros((50, 50, 3))
        img_res_zy = np.zeros((50, 50, 3))
        img_res_zx = np.zeros((50, 50, 3))
        for x in range(50):
            for y in range(50):
                for z in range(50):
                    if (tiff6[z][x][y][0] == 0) & (tiff6[z][x][y][1] == 100) & (tiff6[z][x][y][2] == 0):
                        img_res_xy[x][y][0] = 0
                        img_res_xy[x][y][1] = 100
                        img_res_xy[x][y][2] = 0
                        break
                    elif (tiff6[z][x][y][0] > img_res_xy[x][y][0]):
                        img_res_xy[x][y][0] = tiff6[z][x][y][0]
                        img_res_xy[x][y][1] = tiff6[z][x][y][1]
                        img_res_xy[x][y][2] = tiff6[z][x][y][2]
        for z in range(50):
            for y in range(50):
                for x in range(50):
                    if (tiff6[z][x][y][0] == 0) & (tiff6[z][x][y][1] == 100) & (tiff6[z][x][y][2] == 0):
                        img_res_zy[z][y][0] = 0
                        img_res_zy[z][y][1] = 100
                        img_res_zy[z][y][2] = 0
                        break
                    elif (tiff6[z][x][y][0] > img_res_zy[z][y][0]):
                        img_res_zy[z][y][0] = tiff6[z][x][y][0]
                        img_res_zy[z][y][1] = tiff6[z][x][y][1]
                        img_res_zy[z][y][2] = tiff6[z][x][y][2]
        for z in range(50):
            for x in range(50):
                for y in range(50):
                    if (tiff6[z][x][y][0] == 0) & (tiff6[z][x][y][1] == 100) & (tiff6[z][x][y][2] == 0):
                        img_res_zx[z][x][0] = 0
                        img_res_zx[z][x][1] = 100
                        img_res_zx[z][x][2] = 0
                        break
                    elif (tiff6[z][x][y][0] > img_res_zx[z][x][0]):
                        img_res_zx[z][x][0] = tiff6[z][x][y][0]
                        img_res_zx[z][x][1] = tiff6[z][x][y][1]
                        img_res_zx[z][x][2] = tiff6[z][x][y][2]

        name8_xy = path8+"/"+iiiii.name[0:-4]+"-xy.tif"
        img_xy = Image.fromarray(np.uint8(img_res_xy))
        img_xy.save(name8_xy)
        name8_zy = path8+"/"+iiiii.name[0:-4]+"-zy.tif"
        img_zy = Image.fromarray(np.uint8(img_res_zy))
        img_zy.save(name8_zy)
        name8_zx = path8+"/"+iiiii.name[0:-4]+"-zx.tif"
        img_zx = Image.fromarray(np.uint8(img_res_zx))
        img_zx.save(name8_zx)
        print("补充后图像生成完毕")

result2.to_csv("lesstif/03_new3.txt",sep=' ',index=0,header=0)