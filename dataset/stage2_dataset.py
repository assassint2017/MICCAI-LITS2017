"""

随机取样方式下的数据集
"""

import os
import random

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset

on_server = True


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation'), self.ct_list))

        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # 在slice平面内随机选取256*256的patch
        point_x = random.randint(0, 256)
        point_y = random.randint(0, 256)

        ct_array = ct_array[:, point_x:point_x + 256, point_y:point_y + 256]
        seg_array = seg_array[:, point_x:point_x + 256, point_y:point_y + 256]

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


# 第二阶段的训练数据
ct_dir = '/home/zcy/Desktop/train/random/ct/' \
    if on_server is False else './train/random/ct/'
seg_dir = '/home/zcy/Desktop/train/random/seg/' \
    if on_server is False else './train/random/seg/'

train_random_ds = Dataset(ct_dir, seg_dir)


# # 测试代码
# from torch.utils.data import DataLoader
# train_dl = DataLoader(train_random_ds, 6, True)
# for index, (ct, seg) in enumerate(train_dl):
#
#     print(index, ct.size(), seg.size())
#     print('----------------')
