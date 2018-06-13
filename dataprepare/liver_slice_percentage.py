"""

查看肝脏区域slice占据整体slice的比例
"""
import os

import SimpleITK as sitk


seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/train/seg/'

total_slice = .0
total_liver_slice = .0

for index, seg_file in enumerate(os.listdir(seg_path), start=1):

    seg = sitk.ReadImage(os.path.join(seg_path, seg_file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice or 2 in slice:
            liver_slice += 1

    total_slice += seg_array.shape[0]
    total_liver_slice += liver_slice

    print('index:{}, precent:{:.4f}'.format(index, liver_slice / seg_array.shape[0] * 100))

print(total_liver_slice / total_slice)

# 训练集包含肝脏的slice整体占比: 32.59%
# 测试集包含肝脏的slice整体占比: 33.23%
