import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
import tifffile
import os
import random
import math

# out=np.load("data.npz")
# ooo=out["arr_0"]
# str_zz = []
# str_zz=np.array(str_zz)
# for z in ooo:
#     tmp1 = str(z)[0:-2]
#     # print(tmp1)
#     tmp2 = ''
#     for i in range(5 - len(tmp1)):
#         tmp2 = tmp2 + '0'
#     tmp2 = tmp2 + tmp1+".npz"
#     # print(tmp2)
#     str_zz=np.append(str_zz,tmp2)
#
# print(str_zz)

names=['n','type','x','y','z','radius','parent','seg_id','level','mode','timestamp','TFresindex']
path=r"swc"
#读取swc文件
final_num=3
result1=pd.read_table('swc/03_new.txt',names=names,index_col='n',sep=' ')
print(result1)
result2=pd.read_table('swc/03_new.txt',names=names,sep=' ')
# print(result2)
endend=result2['n'].shape[0]
endend2=endend
result1['z'] = result1['z'] / 2.0
result1['x'] = result1['x'] / 0.3
result1['y'] = result1['y'] / 0.3
#获取终端点标号
tmp1=list(result2['n'])
tmp2=list(result2['parent'])
res=list(set(tmp1)-set(tmp2))
print("该文件共有"+str(len(res))+"个终端点")
print(res)
#获取终端点x,y,z坐标值
tip_z=[]
tip_x=[]
tip_y=[]
for tmp in res:
    tip_z.append(round(result1.loc[int(tmp)]['z']))
    tip_x.append(round(result1.loc[int(tmp)]['x']))
    tip_y.append(round(result1.loc[int(tmp)]['y']))

# print(res)
# print(len(res))
#遍历每一个终端点
iii=len(res)-1
print("开始生成第"+str(iii+1)+"个终端点图像")
res_z=[]
res_x=[]
res_y=[]
# now = result1.loc[int(res[iii])]['parent']
x = round(result1.loc[int(res[iii])]['x'])
y = round(result1.loc[int(res[iii])]['y'])
z = round(result1.loc[int(res[iii])]['z'])

# rannum=random.uniform(10,20)
# rannum=12
# print(rannum)
#
# tmp1=math.pow(px-tip_x[iii],2)+math.pow(py-tip_y[iii],2)+math.pow(pz-tip_z[iii],2)
# tmp2=math.sqrt(tmp1)
# rr_x = round(tip_x[iii] + 1.0 * rannum / tmp2 * (tip_x[iii] - px))
# rr_y = round(tip_y[iii] + 1.0 * rannum / tmp2 * (tip_y[iii] - py))
# rr_z = round(tip_z[iii] + 1.0 * rannum / tmp2 * (tip_z[iii] - pz))
#
# tmpi=result2['n'].shape[0]
# result2.loc[tmpi]=[tmpi+1,0,rr_x*0.3,rr_y*0.3,rr_z*2,1,int(res[iii]),0,0,0,0,0]
# result2.to_csv('swc/03_new.txt',sep=' ',index=0,header=0)
# print("文件改动完毕")

#获取以终端点为中心的50*50*50像素坐标值
count = np.arange(-24, 26)
for i in count:
    res_z.append(z + i)
    res_x.append(x + i)
    res_y.append(y + i)

# if (res_z[0]<10)|(res_z[-1]>10261):
#     print("第"+str(iii+1)+"个端点超出范围，跳过")
#     continue
# if (res_x[0]<0)|(res_x[-1]>21167):
#     print("第"+str(iii+1)+"个端点超出范围，跳过")
#     continue
# if (res_y[0]<0)|(res_y[-1]>36399):
#     print("第"+str(iii+1)+"个端点超出范围，跳过")
#     continue

#获取z坐标对应的z轴切片图像文件名
str_z=[]
for z in res_z:
    tmp1 = str(z)
    tmp2 = 'D:/fmost/mouseID_210254-15257/Red/Green/Origin/'
    for i in range(5 - len(tmp1)):
        tmp2 = tmp2 + '0'
    tmp2 = tmp2 + tmp1 + '.tif'
    str_z.append(tmp2)

# imin = res_x[0]
# jmin = res_y[0]

#遍历并MIP
res_max_xy = np.zeros((50, 50))
res_max_zy = np.zeros((50, 50))
res_max_zx = np.zeros((50, 50))
res_thr = np.zeros(((50, 50, 50)))
res_rgb = np.zeros(((50, 50, 50, 3)))

zzz=0
for pic in str_z:
    img = tifffile.imread(pic)
    imin = res_x[0]
    jmin = res_y[0]
    for i in res_x:
        for j in res_y:
            res_thr[zzz][i - imin][j - jmin] = img[j][i]
            res_rgb[zzz][i - imin][j - jmin][0] = img[j][i]
            res_rgb[zzz][i - imin][j - jmin][1] = img[j][i]
            res_rgb[zzz][i - imin][j - jmin][2] = img[j][i]
            # if (res_max[i - imin][j - jmin] < img[j][i]):
            #     res_max[i - imin][j - jmin] = img[j][i]
    print("第"+str(iii)+"个结点"+"第"+str(zzz+1)+"部分已处理完毕")
    zzz=zzz+1
# res_max_xy = np.max(res_thr, axis=0)
# res_max_zy = np.max(res_thr, axis=1)
# res_max_zx = np.max(res_thr, axis=2)

# res_max_xy = res_max_xy / 4096.0 * 256.0
# res_max_zy = res_max_zy / 4096.0 * 256.0
# res_max_zx = res_max_zx / 4096.0 * 256.0

res_thr = res_thr / 4096.0 * 256.0
res_rgb = res_rgb / 4096.0 * 256.0

range_all = res_rgb.max() - res_rgb.min()
res_rgb = (res_rgb - res_rgb.min()) / range_all * 255.0

name6=str(final_num)+'-'+str(iii + 1)+"-1.tif"
tifffile.imwrite(name6, res_rgb)
print("三维RGB图像生成完毕")

# im3=Image.open(name2)
#
iz = 34
ix = 24
iy = 24

points = []
nnn = 0
points.append([int(res[iii]), int(iz), int(ix), int(iy)])
nnn = nnn + 1

img = tifffile.imread(name6)
img[int(iz), int(ix), int(iy)] = [0, 100, 0]
# if(nnn==0):
#     now=res[iii]
#     tx = round(result1.loc[int(now)]['x'])
#     ty = round(result1.loc[int(now)]['y'])
#     tz = round(result1.loc[int(now)]['z'])+10
# else:
now = result1.loc[int(res[iii])]['parent']
tx = round(result1.loc[int(now)]['x'])
ty = round(result1.loc[int(now)]['y'])
tz = round(result1.loc[int(now)]['z'])+10
while (1):
    if ((tx > res_x[-1]) | (tx < res_x[0]) | (ty > res_y[-1]) | (ty < res_y[0]) | (tz > res_z[-1]) | (
            tz < res_z[0])):
        points.append([int(now),int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])])
        break
    else:
        points.append([int(now),int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])])
        nnn = nnn + 1
        img[int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])] = [0, 100, 0]
        now = result1.loc[int(now)]['parent']
        if (now == -1):
            break
        tx = round(result1.loc[int(now)]['x'])
        ty = round(result1.loc[int(now)]['y'])
        tz = round(result1.loc[int(now)]['z'])+10

resend=[]
lenn = len(points)
for i in range(lenn - 1):
    gz = points[i + 1][1] - points[i][1]
    gx = points[i + 1][2] - points[i][2]
    gy = points[i + 1][3] - points[i][3]
    fz = fy = fx = 1
    if (gz < 0):
        fz = -1
        gz = fz * gz
    if (gy < 0):
        fy = -1
        gy = fy * gy
    if (gx < 0):
        fx = -1
        gx = fx * gx

    gmax = max(gz, gy, gx)
    if(i<lenn-2):
        resend.append([points[i][0],points[i][2]+res_x[0],points[i][3]+res_y[0],points[i][1]+res_z[0]])
        for j in range(gmax):
            ez = round(points[i][1] + fz * 1.0 * gz / gmax * j)
            ex = round(points[i][2] + fx * 1.0 * gx / gmax * j)
            ey = round(points[i][3] + fy * 1.0 * gy / gmax * j)
            if not ((img[ez,ex,ey][0]==0)&(img[ez,ex,ey][1]==100)&(img[ez,ex,ey][2]==0)):
                img[ez, ex, ey] = [0, 100, 0]
                resend.append([endend+1,ex+res_x[0],ey+res_y[0],ez+res_z[0]])
                endend+=1

    else:
        resend.append([points[i][0], points[i][2] + res_x[0], points[i][3] + res_y[0], points[i][1] + res_z[0]])
        for j in range(gmax):
            ez = round(points[i][1] + fz * 1.0 * gz / gmax * j)
            ex = round(points[i][2] + fx * 1.0 * gx / gmax * j)
            ey = round(points[i][3] + fy * 1.0 * gy / gmax * j)
            if (ez>49)|(ez<0)|(ex>49)|(ex<0)|(ey>49)|(ey<0):
                break
            if not ((img[ez, ex, ey][0] == 0) & (img[ez, ex, ey][1] == 100) & (img[ez, ex, ey][2] == 0)):
                img[ez, ex, ey] = [0, 100, 0]
                resend.append([endend+1, ex + res_x[0], ey + res_y[0], ez + res_z[0]])
                endend+=1

name7 = str(final_num) + '-' + str(iii + 1) + "-2.tif"
tifffile.imwrite(name7, img)
print("三维图像标记连线完毕")

resend.append([points[lenn-1][0], points[lenn-1][2] + res_x[0], points[lenn-1][3] + res_y[0], points[lenn-1][1] + res_z[0]])

for mmm in range(len(resend)-1):
        result2.loc[resend[mmm][0]-1] = [resend[mmm][0], 0, resend[mmm][1] * 0.3, resend[mmm][2] * 0.3, resend[mmm][3] * 2, 1,
                                         resend[mmm+1][0], 0, 0, 0, 0, 0]
result2.to_csv('swc/03_new.txt',sep=' ',index=0,header=0)
print("文件改动完毕")

img = tifffile.imread(name7)
img_res_xy = np.zeros((50, 50, 3))
img_res_zy = np.zeros((50, 50, 3))
img_res_zx = np.zeros((50, 50, 3))
for x in range(50):
    for y in range(50):
        for z in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_xy[x][y][0] = 0
                img_res_xy[x][y][1] = 100
                img_res_xy[x][y][2] = 0
                break
            elif (img[z][x][y][0] > img_res_xy[x][y][0]):
                img_res_xy[x][y][0] = img[z][x][y][0]
                img_res_xy[x][y][1] = img[z][x][y][1]
                img_res_xy[x][y][2] = img[z][x][y][2]
for z in range(50):
    for y in range(50):
        for x in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_zy[z][y][0] = 0
                img_res_zy[z][y][1] = 100
                img_res_zy[z][y][2] = 0
                break
            elif (img[z][x][y][0] > img_res_zy[z][y][0]):
                img_res_zy[z][y][0] = img[z][x][y][0]
                img_res_zy[z][y][1] = img[z][x][y][1]
                img_res_zy[z][y][2] = img[z][x][y][2]
for z in range(50):
    for x in range(50):
        for y in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_zx[z][x][0] = 0
                img_res_zx[z][x][1] = 100
                img_res_zx[z][x][2] = 0
                break
            elif (img[z][x][y][0] > img_res_zx[z][x][0]):
                img_res_zx[z][x][0] = img[z][x][y][0]
                img_res_zx[z][x][1] = img[z][x][y][1]
                img_res_zx[z][x][2] = img[z][x][y][2]

name8_xy = str(final_num) + '-' + str(iii + 1) + "-xy.tif"
img_xy = Image.fromarray(np.uint8(img_res_xy))
img_xy.save(name8_xy)
name8_zy = str(final_num) + '-' + str(iii + 1) + "-zy.tif"
img_zy = Image.fromarray(np.uint8(img_res_zy))
img_zy.save(name8_zy)
name8_zx = str(final_num) + '-' + str(iii + 1) + "-zx.tif"
img_zx = Image.fromarray(np.uint8(img_res_zx))
img_zx.save(name8_zx)
print("二维图像映射完毕")

print("第" + str(iii + 1) + "个终端点图像生成完毕")

print("文件处理完毕")
final_num=final_num+1

tiff=tifffile.imread(name7)
tiff2=tifffile.imread(name6)

tiff3=tiff2.flatten()
tiff3=np.sort(tiff3)
yuzhi_H=tiff3[int(0.99*tiff3.shape[0])]
# print(yuzhi_H)
resend=np.array(resend)
res_near=[]
iii_=resend.shape[0]-1
for i_near in range(iii_):
    tmp=[]
    for a in range(-1,2):
        for b in range(-1,2):
            for c in range(-1,2):
                if abs(a)+abs(b)+abs(c)<3:
                    x=resend[i_near][1]+a-res_x[0]
                    y=resend[i_near][2]+b-res_y[0]
                    z=resend[i_near][3]+c-res_z[0]
                    if (x>=0)&(x<50)&(y>=0)&(y<50)&(z>=0)&(z<50):
                        tmp.append(x*2500+y*50+z)
    print(tmp)
    res_near.append(tmp)

qianjing=[]
biao=0
flag=0
while(biao<iii_):
    rec = []
    res_near = np.array(res_near)
    rec = np.array(rec)
    for i in range(biao,15+biao):
        if(i>iii_-1):
            break
        rec = np.union1d(rec, res_near[i])
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
    yuzhi=min(yuzhi_B,yuzhi_H)
    print(yuzhi)
    if (tiff2[resend[biao][3]-res_z[0]][resend[biao][1]-res_x[0]][resend[biao][2]-res_y[0]]>yuzhi)&(flag==0):
        qianjing.append(resend[biao][0])
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][0]=100
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][1]=0
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][2]=0
        flag=1
    elif (tiff2[resend[biao][3]-res_z[0]][resend[biao][1]-res_x[0]][resend[biao][2]-res_y[0]]<=yuzhi):
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][0] = tiff2[resend[biao][3]-res_z[0]][resend[biao][1]-res_x[0]][resend[biao][2]-res_y[0]]
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][1] = tiff2[resend[biao][3]-res_z[0]][resend[biao][1]-res_x[0]][resend[biao][2]-res_y[0]]
        tiff[resend[biao][3] - res_z[0]][resend[biao][1] - res_x[0]][resend[biao][2] - res_y[0]][2] = tiff2[resend[biao][3]-res_z[0]][resend[biao][1]-res_x[0]][resend[biao][2]-res_y[0]]
        flag=0
    # print(tiff2[res_new[biao][0]][res_new[biao][1]][res_new[biao][2]][0])
    # print("---")
    biao+=1

qianjing=np.array(qianjing)
np.save("qianjing",qianjing)
print("前景点保存完毕")

img=tiff
name8 = str(final_num) + '-' + str(iii + 1) + "-3.tif"
tifffile.imwrite(name8, img)
img_res_xy = np.zeros((50, 50, 3))
img_res_zy = np.zeros((50, 50, 3))
img_res_zx = np.zeros((50, 50, 3))
for x in range(50):
    for y in range(50):
        for z in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_xy[x][y][0] = 0
                img_res_xy[x][y][1] = 100
                img_res_xy[x][y][2] = 0
                break
            elif (img[z][x][y][0] == 100) & (img[z][x][y][1] == 0) & (img[z][x][y][2] == 0):
                img_res_xy[x][y][0] = 100
                img_res_xy[x][y][1] = 0
                img_res_xy[x][y][2] = 0
                break
            elif (img[z][x][y][0] > img_res_xy[x][y][0]):
                img_res_xy[x][y][0] = img[z][x][y][0]
                img_res_xy[x][y][1] = img[z][x][y][1]
                img_res_xy[x][y][2] = img[z][x][y][2]
for z in range(50):
    for y in range(50):
        for x in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_zy[z][y][0] = 0
                img_res_zy[z][y][1] = 100
                img_res_zy[z][y][2] = 0
                break
            elif (img[z][x][y][0] == 100) & (img[z][x][y][1] == 0) & (img[z][x][y][2] == 0):
                img_res_zy[z][y][0] = 100
                img_res_zy[z][y][1] = 0
                img_res_zy[z][y][2] = 0
                break
            elif (img[z][x][y][0] > img_res_zy[z][y][0]):
                img_res_zy[z][y][0] = img[z][x][y][0]
                img_res_zy[z][y][1] = img[z][x][y][1]
                img_res_zy[z][y][2] = img[z][x][y][2]
for z in range(50):
    for x in range(50):
        for y in range(50):
            if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 100) & (img[z][x][y][2] == 0):
                img_res_zx[z][x][0] = 0
                img_res_zx[z][x][1] = 100
                img_res_zx[z][x][2] = 0
                break
            elif (img[z][x][y][0] == 100) & (img[z][x][y][1] == 0) & (img[z][x][y][2] == 0):
                img_res_zx[z][x][0] = 100
                img_res_zx[z][x][1] = 0
                img_res_zx[z][x][2] = 0
                break
            elif (img[z][x][y][0] > img_res_zx[z][x][0]):
                img_res_zx[z][x][0] = img[z][x][y][0]
                img_res_zx[z][x][1] = img[z][x][y][1]
                img_res_zx[z][x][2] = img[z][x][y][2]

name8_xy = str(final_num) + '-' + str(iii + 1) + "-3-xy.tif"
img_xy = Image.fromarray(np.uint8(img_res_xy))
img_xy.save(name8_xy)
name8_zy = str(final_num) + '-' + str(iii + 1) + "-3-zy.tif"
img_zy = Image.fromarray(np.uint8(img_res_zy))
img_zy.save(name8_zy)
name8_zx = str(final_num) + '-' + str(iii + 1) + "-3-zx.tif"
img_zx = Image.fromarray(np.uint8(img_res_zx))
img_zx.save(name8_zx)
print("二维图像映射完毕")


