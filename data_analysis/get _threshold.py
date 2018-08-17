"""

选取合适的截断阈值
"""
import os

import SimpleITK as sitk

upper = 200
lower = -200

ct_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/CT/'
seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'

# segmentation-0.ct
# volume-0.ct

num_point = 0.0
num_inlier = 0.0

for ct_file in os.listdir(ct_path):

    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg_file = ct_file.replace('volume', 'segmentation')
    seg = sitk.ReadImage(os.path.join(seg_path, seg_file), sitk.sitkInt16)
    seg_array = sitk.GetArrayFromImage(seg)

    liver_mask = seg_array > 0
    liver_roi = ct_array[liver_mask]

    inliers = ((liver_roi < upper) * (liver_roi > lower)).astype(int).sum()

    print('{:.4}%'.format(inliers / liver_roi.shape[0] * 100))
    print('------------')

    num_point += liver_roi.shape[0]
    num_inlier += inliers

print(num_inlier / num_point)

# -200 到 200 的阈值对于肝脏：训练集99.42%， 测试集99.53%
# -200 到 200 的阈值对于肿瘤：训练集99.90%， 测试集99.98%

