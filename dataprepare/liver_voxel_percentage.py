"""

查看肝脏区域像素点个数占据只包含肝脏区域的slice的百分比
"""
import os

import SimpleITK as sitk


seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'

total_point = .0
total_liver_point = .0

for index, seg_file in enumerate(os.listdir(seg_path), start=1):

    seg = sitk.ReadImage(os.path.join(seg_path, seg_file))
    seg_array = sitk.GetArrayFromImage(seg)

    liver_slice = 0

    for slice in seg_array:
        if 1 in slice or 2 in slice:
            liver_slice += 1

    liver_point = (seg_array > 0).astype(int).sum()

    print('index:{}, precent:{:.4f}'.format(index, liver_point / (liver_slice * 512 * 512) * 100))

    total_point += (liver_slice * 512 * 512)
    total_liver_point += liver_point

print(total_liver_point / total_point)

# 训练集 6.95%
# 测试集 7.27%
