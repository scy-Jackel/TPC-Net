import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import cv2
from PIL import Image,ImageDraw
from libtiff import TIFF
import tifffile
import os
from multiprocessing import Pool

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
with os.scandir(path) as it:
    final_num=4
    for iiiii in it:
        print("开始处理文件"+str(iiiii.name))

        result1=pd.read_table('swc/'+str(iiiii.name),names=names,index_col='n',sep=' ')
        print(result1)
        result2=pd.read_table('swc/'+str(iiiii.name),names=names,sep=' ')
        # print(result2)

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
        for iii in range(0,len(res)):
            print("开始生成第"+str(iii+1)+"个终端点图像")
            res_z=[]
            res_x=[]
            res_y=[]

            #获取以终端点为中心的50*50*50像素坐标值
            count = np.arange(-24, 26)
            for i in count:
                res_z.append(tip_z[iii] + i)
                res_x.append(tip_x[iii] + i)
                res_y.append(tip_y[iii] + i)

            if (res_z[0]<10)|(res_z[-1]>10261):
                print("第"+str(iii+1)+"个端点超出范围，跳过")
                continue
            if (res_x[0]<0)|(res_x[-1]>21167):
                print("第"+str(iii+1)+"个端点超出范围，跳过")
                continue
            if (res_y[0]<0)|(res_y[-1]>36399):
                print("第"+str(iii+1)+"个端点超出范围，跳过")
                continue

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
            res_max_xy = np.max(res_thr, axis=0)
            res_max_zy = np.max(res_thr, axis=1)
            res_max_zx = np.max(res_thr, axis=2)

            if(np.max(res_thr)>256.0):
                res_max_xy = res_max_xy / 4096.0 * 256.0
                res_max_zy = res_max_zy / 4096.0 * 256.0
                res_max_zx = res_max_zx / 4096.0 * 256.0
                res_thr = res_thr / 4096.0 * 256.0
                res_rgb = res_rgb / 4096.0 * 256.0

            #保存图像
            im1_xy = Image.fromarray(res_max_xy)
            name1_xy = "data1_xy/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            im1_xy.save(name1_xy)
            im1_zy = Image.fromarray(res_max_zy)
            name1_zy = "data1_zy/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            im1_zy.save(name1_zy)
            im1_zx = Image.fromarray(res_max_zx)
            name1_zx = "data1_zx/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            im1_zx.save(name1_zx)
            print("原始灰度图像生成完毕")

            name5="data5/"+str(final_num)+'-'+str(iii + 1)+".tif"
            tifffile.imsave(name5, res_thr)
            print("三维灰度图像生成完毕")

            # #将灰度图像转为RGB图像
            # im2 = Image.open(name1)
            # out = im2.convert("RGB")
            # img = np.array(out)
            # im2 = Image.fromarray(img)
            # name2="data2/"+str(final_num)+'-'+str(iii + 1)+".tif"
            # im2.save(name2)
            # print("RGB图像生成完毕")

            name6="data6/"+str(final_num)+'-'+str(iii + 1)+".tif"
            tifffile.imsave(name6, res_rgb)
            print("三维RGB图像生成完毕")

            # im3=Image.open(name2)
            #
            iz = 24
            ix = 24
            iy = 24
            #
            # points = []
            # nnn = 0
            # points.append([int(iy), int(ix)])
            # nnn = nnn + 1
            #
            # img3 = np.array(im3)
            # img3[int(ix), int(iy)] = [0, 255, 0]
            # now = result1.loc[int(res[iii])]['parent']
            # tx = round(result1.loc[int(now)]['x'])
            # ty = round(result1.loc[int(now)]['y'])
            # tz = round(result1.loc[int(now)]['z'])
            #
            # #标记截取图像范围内的swc文件中标记点
            # while (1):
            #
            #     if ((tx > res_x[-1]) | (tx < res_x[0]) | (ty > res_y[-1]) | (ty < res_y[0]) | (tz > res_z[-1]) | (
            #             tz < res_z[0])):
            #         break
            #     else:
            #         points.append([int(ty - res_y[0]), int(tx - res_x[0])])
            #         nnn = nnn + 1
            #         img3[int(tx - res_x[0]), int(ty - res_y[0])] = [0, 255, 0]
            #         now = result1.loc[int(now)]['parent']
            #         if now == -1:
            #             break
            #         tx = round(result1.loc[int(now)]['x'])
            #         ty = round(result1.loc[int(now)]['y'])
            #         tz = round(result1.loc[int(now)]['z'])
            #
            # im3 = Image.fromarray(np.uint8(img3))
            # name3 = "data3/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            # im3.save(name3)
            # print("swc点标记完毕")
            #
            # im4 = Image.open(name3)
            # draw = ImageDraw.Draw(im4)
            # #标记点连线
            # for i in range(nnn - 1):
            #     draw.line([points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]], fill='green', width=1)
            # name4="data4/"+str(final_num)+'-'+str(iii + 1)+".tif"
            # im4.save(name4)
            # print("连线完毕")

            points = []
            nnn = 0
            points.append([int(iz), int(ix), int(iy)])
            nnn = nnn + 1

            img = tifffile.imread(name6)
            img[int(iz), int(ix), int(iy)] = [0, 255, 0]
            now = result1.loc[int(res[iii])]['parent']
            tx = round(result1.loc[int(now)]['x'])
            ty = round(result1.loc[int(now)]['y'])
            tz = round(result1.loc[int(now)]['z'])
            while (1):
                if ((tx > res_x[-1]) | (tx < res_x[0]) | (ty > res_y[-1]) | (ty < res_y[0]) | (tz > res_z[-1]) | (
                        tz < res_z[0])):
                    points.append([int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])])
                    break;
                else:
                    points.append([int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])])
                    nnn = nnn + 1
                    img[int(tz - res_z[0]), int(tx - res_x[0]), int(ty - res_y[0])] = [0, 255, 0]
                    now = result1.loc[int(now)]['parent']
                    if (now == -1):
                        break
                    tx = round(result1.loc[int(now)]['x'])
                    ty = round(result1.loc[int(now)]['y'])
                    tz = round(result1.loc[int(now)]['z'])

            lenn = len(points)
            for i in range(lenn - 1):
                gz = points[i + 1][0] - points[i][0]
                gx = points[i + 1][1] - points[i][1]
                gy = points[i + 1][2] - points[i][2]
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
                    for j in range(gmax):
                        ez = round(points[i][0] + fz * 1.0 * gz / gmax * j)
                        ex = round(points[i][1] + fx * 1.0 * gx / gmax * j)
                        ey = round(points[i][2] + fy * 1.0 * gy / gmax * j)
                        img[ez, ex, ey] = [0, 255, 0]
                else:
                    for j in range(gmax):
                        ez = round(points[i][0] + fz * 1.0 * gz / gmax * j)
                        ex = round(points[i][1] + fx * 1.0 * gx / gmax * j)
                        ey = round(points[i][2] + fy * 1.0 * gy / gmax * j)
                        if (ez>49)|(ez<0)|(ex>49)|(ex<0)|(ey>49)|(ey<0):
                            break
                        img[ez, ex, ey] = [0, 255, 0]

            name7 = "data7/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            tifffile.imsave(name7, img)
            print("三维图像标记连线完毕")

            img = tifffile.imread(name7)
            img_res_xy = np.zeros((50, 50, 3))
            img_res_zy = np.zeros((50, 50, 3))
            img_res_zx = np.zeros((50, 50, 3))
            for x in range(50):
                for y in range(50):
                    for z in range(50):
                        if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 255) & (img[z][x][y][2] == 0):
                            img_res_xy[x][y][0] = 0
                            img_res_xy[x][y][1] = 255
                            img_res_xy[x][y][2] = 0
                            break
                        elif (img[z][x][y][0] > img_res_xy[x][y][0]):
                            img_res_xy[x][y][0] = img[z][x][y][0]
                            img_res_xy[x][y][1] = img[z][x][y][1]
                            img_res_xy[x][y][2] = img[z][x][y][2]
            for z in range(50):
                for y in range(50):
                    for x in range(50):
                        if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 255) & (img[z][x][y][2] == 0):
                            img_res_zy[z][y][0] = 0
                            img_res_zy[z][y][1] = 255
                            img_res_zy[z][y][2] = 0
                            break
                        elif (img[z][x][y][0] > img_res_zy[z][y][0]):
                            img_res_zy[z][y][0] = img[z][x][y][0]
                            img_res_zy[z][y][1] = img[z][x][y][1]
                            img_res_zy[z][y][2] = img[z][x][y][2]
            for z in range(50):
                for x in range(50):
                    for y in range(50):
                        if (img[z][x][y][0] == 0) & (img[z][x][y][1] == 255) & (img[z][x][y][2] == 0):
                            img_res_zx[z][x][0] = 0
                            img_res_zx[z][x][1] = 255
                            img_res_zx[z][x][2] = 0
                            break
                        elif (img[z][x][y][0] > img_res_zx[z][x][0]):
                            img_res_zx[z][x][0] = img[z][x][y][0]
                            img_res_zx[z][x][1] = img[z][x][y][1]
                            img_res_zx[z][x][2] = img[z][x][y][2]

            name8_xy = "data8_xy/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            img_xy = Image.fromarray(np.uint8(img_res_xy))
            img_xy.save(name8_xy)
            name8_zy = "data8_zy/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            img_zy = Image.fromarray(np.uint8(img_res_zy))
            img_zy.save(name8_zy)
            name8_zx = "data8_zx/" + str(final_num) + '-' + str(iii + 1) + ".tif"
            img_zx = Image.fromarray(np.uint8(img_res_zx))
            img_zx.save(name8_zx)
            print("二维图像映射完毕")

            print("第" + str(iii + 1) + "个终端点图像生成完毕")

        print(str(iiiii.name)+"文件处理完毕")
        final_num=final_num+1







