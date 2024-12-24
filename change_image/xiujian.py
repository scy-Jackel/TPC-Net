import os
import shutil
import os
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import numpy as np
import tifffile
import math
#2-87
#3-109

res=[]
flag=-1
num=0
tiff=tifffile.imread("data_3D_final/2-115.tif")
for i in range(50):
    for j in range(50):
        for k in range(50):
            if (tiff[i][j][k][0] == 0) & (tiff[i][j][k][1] == 100) & (tiff[i][j][k][2] == 0):
                res.append((i,j,k))
                if (i==34)&(j==24)&(k==24):
                    flag=num
                else:
                    num+=1

res=np.array(res)
iii=res.shape[0]
dis=np.zeros((iii,iii))
for a in range(iii):
    for b in range(iii):
        if(a==b):
            dis[a][b]=7300
        else:
            dis[a][b]=(res[a][0]-res[b][0])**2+(res[a][1]-res[b][1])**2+(res[a][2]-res[b][2])**2
res_new=[]
for i in range(iii):
    res_new.append(list(res[flag]))
    dis[:,flag]=7300
    flag= np.argmin(dis[flag])
print(res)
print(res_new)

tiff2=tifffile.imread("data6/2-115.tif")

# yuzhi_N=np.mean(tiff2)+np.var(tiff2)
tiff3=tiff2.flatten()
tiff3=np.sort(tiff3)
yuzhi_H=tiff3[int(0.99*tiff3.shape[0])]
# print(yuzhi_H)
res_near=[]
for i_near in range(iii):
    tmp=[]
    for a in range(-1,2):
        for b in range(-1,2):
            for c in range(-1,2):
                # if math.sqrt( a**2+b**2+c**2 )<=2:
                if abs(a)+abs(b)+abs(c)<3:
                    x=res_new[i_near][0]+a
                    y=res_new[i_near][1]+b
                    z=res_new[i_near][2]+c
                    if (x>=0)&(x<50)&(y>=0)&(y<50)&(z>=0)&(z<50):
                        tmp.append(x*2500+y*50+z)
    print(tmp)
    res_near.append(tmp)

biao=0
while(biao<15):
    if(biao>iii-1):
        break
    rec = []
    for i in range(biao,15+biao):
        if(i>iii-1):
            break
        rec = np.union1d(rec, res_near[i])
    print(rec)
    res_mean=[]
    for i in range(rec.shape[0]):
        a=int(rec[i]/2500)
        # print(a)
        rec[i]=rec[i]-a*2500
        b=int(rec[i]/50)
        # print(b)
        c=int(rec[i]-b*50)
        # print(c)
        res_mean.append(tiff2[a][b][c][0])
    res_mean=np.array(res_mean)
    yuzhi_B=np.mean(res_mean)
    # print(yuzhi_B)
    print(yuzhi_B)
    print(yuzhi_H)
    yuzhi=min(yuzhi_B,yuzhi_H)
    print(yuzhi)
    print(tiff2[res_new[biao][0]][res_new[biao][1]][res_new[biao][2]][0])
    print("---")
    if(tiff2[res_new[biao][0]][res_new[biao][1]][res_new[biao][2]][0]<yuzhi):
        tiff[res_new[biao][0]][res_new[biao][1]][res_new[biao][2]]=tiff2[res_new[biao][0]][res_new[biao][1]][res_new[biao][2]]
    else:
        break
    biao+=1

tifffile.imsave("data_xiujian/2-115-xj.tif",tiff)
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
name8_xy = "data_xiujian/2-115-xj-xy.tif"
img_xy = Image.fromarray(np.uint8(img_res_xy))
img_xy.save(name8_xy)
name8_zy = "data_xiujian/2-115-xj-zy.tif"
img_zy = Image.fromarray(np.uint8(img_res_zy))
img_zy.save(name8_zy)
name8_zx = "data_xiujian/2-115-xj-zx.tif"
img_zx = Image.fromarray(np.uint8(img_res_zx))
img_zx.save(name8_zx)










