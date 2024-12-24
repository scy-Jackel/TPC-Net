import os
import shutil
import os
import numpy as np
# path1="data_2D/1-1-xy.tif"
# path2="data_2D/1-ok"
# shutil.move(path1, path2)
path1=r"data8_2D-o-new"
path2=r"data8_2D-o"
panduan=np.load("ttt.npy")
with os.scandir(path2) as it:
    for iii in it:
        name=iii.name[0:-12]+".copy.png"
        if(name in panduan):
            shutil.move(iii.path, path1)
            print(iii.name)
        # name1 = "data8_2D-o/" + iii.name[0:-9] + "-xy.copy.tif"
        # name2 = "data8_2D-o/" + iii.name[0:-9] + "-zx.copy.tif"
        # name3 = "data8_2D-o/" + iii.name[0:-9] + "-zy.copy.tif"
        # name="data7-o/"+iii.name;
        # shutil.move(name1,"data_2D/2-over")
        # shutil.move(name2, "data_2D/2-over")
        # shutil.move(name3, "data_2D/2-over")
        # shutil.move(name,"data_o/2-over")
        # print(iii.name)



# path=r"data_o/2-over"
# with os.scandir(path) as it:
#     for iii in it:
#         name1 = "data_2D/" + iii.name[0:-4] + "-xy.tif"
#         name2 = "data_2D/" + iii.name[0:-4] + "-zx.tif"
#         name3 = "data_2D/" + iii.name[0:-4] + "-zy.tif"
#         shutil.move(name1,"data_2D/2-over")
#         shutil.move(name2, "data_2D/2-over")
#         shutil.move(name3, "data_2D/2-over")
#         print(iii.name)
#
#
# path=r"data_o/3-less"
# with os.scandir(path) as it:
#     for iii in it:
#         name1 = "data_2D/" + iii.name[0:-4] + "-xy.tif"
#         name2 = "data_2D/" + iii.name[0:-4] + "-zx.tif"
#         name3 = "data_2D/" + iii.name[0:-4] + "-zy.tif"
#         shutil.move(name1,"data_2D/3-less")
#         shutil.move(name2, "data_2D/3-less")
#         shutil.move(name3, "data_2D/3-less")
#         print(iii.name)