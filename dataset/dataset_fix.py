"""

固定取样方式下的数据集
"""

import os

import torch
import SimpleITK as sitk
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

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


ct_dir = '/home/zcy/Desktop/train/fix/ct/' \
    if on_server is False else './train/fix/ct/'
seg_dir = '/home/zcy/Desktop/train/fix/seg/' \
    if on_server is False else './train/fix/seg/'

train_fix_ds = Dataset(ct_dir, seg_dir)


# # 测试代码
# from torch.utils.data import DataLoader
#
# train_dl = DataLoader(train_fix_ds, 12, True, num_workers=2, pin_memory=True)
# for index, (ct, seg) in enumerate(train_dl):
#     # print(type(ct), type(seg))
#     print(index, ct.size(), seg.size())
#     print('----------------')

