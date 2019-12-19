"""

选取合适的截断阈值
"""

import os

from tqdm import tqdm
import SimpleITK as sitk

import sys
sys.path.append(os.path.split(sys.path[0])[0])

import parameter as para


num_point = 0.0
num_inlier = 0.0

for file in tqdm(os.listdir(para.train_ct_path)):

    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    liver_roi = ct_array[seg_array > 0]
    inliers = ((liver_roi < para.upper) * (liver_roi > para.lower)).astype(int).sum()

    print('{:.4}%'.format(inliers / liver_roi.shape[0] * 100))
    print('------------')

    num_point += liver_roi.shape[0]
    num_inlier += inliers

print(num_inlier / num_point)

# -200 到 200 的阈值对于肝脏：训练集99.49%， 测试集99..0%
# -200 到 200 的阈值对于肿瘤：训练集99.95%， 测试集99.45%
