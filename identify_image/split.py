import os
import random
import shutil

# 训练集
train_pct = 0.8
# 验证集S
valid_pct = 0.1
# 测试集
test_pct = 0.1

random.seed(1)
# 'data\\RMB_data'
dataset_dir = "3D_img"
# 'data\\rmb_split'
# split_dir = os.path.join("data", "rmb_split")
# # 'data\\rmb_split\\train'
# train_dir = os.path.join(split_dir, "train")
# # 'data\\rmb_split\\valid'
# valid_dir = os.path.join(split_dir, "valid")
# # 'data\\rmb_split\\test'
# test_dir = os.path.join(split_dir, "test")

for root, dirs, files in os.walk(dataset_dir):
    for sub_dir in dirs:
        # 文件列表
        imgs = os.listdir(os.path.join(root, sub_dir))
        # 取出 jpg 结尾的文件
        random.shuffle(imgs)
        # 计算图片数量
        img_count = len(imgs)
        # 计算训练集索引的结束位置
        train_point = int(img_count * train_pct)
        # 计算验证集索引的结束位置
        valid_point = int(img_count * (train_pct + valid_pct))
        # 把数据划分到训练集、验证集、测试集的文件夹
        for i in range(img_count):
            if i < train_point:
                out_dir = os.path.join("train", sub_dir)
            elif i < valid_point:
                out_dir = os.path.join("valid", sub_dir)
            else:
                out_dir = os.path.join("test", sub_dir)
            # 构造目标文件名
            target_path = os.path.join(out_dir, imgs[i])
            # 构造源文件名
            src_path = os.path.join(dataset_dir, sub_dir, imgs[i])
            # 复制
            shutil.move(src_path, target_path)

        print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                             img_count-valid_point))
