"""

获取随机取样方式下的训练数据
"""

import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage


ct_dir = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/train/CT/'
seg_dir = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/train/seg/'

new_ct_dir = '/home/zcy/Desktop/train/ct/'
new_seg_dir = '/home/zcy/Desktop/train/seg/'

if os.path.exists('/home/zcy/Desktop/train/'):
    shutil.rmtree('/home/zcy/Desktop/train/')

os.mkdir('/home/zcy/Desktop/train/')
os.mkdir(new_ct_dir)
os.mkdir(new_seg_dir)

upper = 200
lower = -200
size = 48
down_scale = 0.5
expand_slice = 20
slice_thickness = 2

start = time()
for index, file in enumerate(os.listdir(ct_dir)):

    # 将CT和金标准入读内存
    ct = sitk.ReadImage(os.path.join(ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(seg_dir, file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 将金标准中肝脏和肝肿瘤的标签融合为一个
    seg_array[seg_array > 0] = 1

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int类型
    # 这里不应该对金标准进行缩小，而且对金标准的插值应该使用最近邻算法
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / slice_thickness, 1, 1), order=0)

    # 找到肝脏区域开始和结束的slice，并各向外扩张slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张slice
    if start_slice - expand_slice < 0:
        start_slice = 0
    else:
        start_slice -= expand_slice

    if end_slice + expand_slice >= seg_array.shape[0]:
        end_slice = seg_array.shape[0] - 1
    else:
        end_slice += expand_slice

    # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
    if end_slice - start_slice + 1 < size:
        print('!!!!!!!!!!!!!!!!')
        print(file, 'have too little slice', ct_array.shape[0])
        print('!!!!!!!!!!!!!!!!')
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    # 最终将数据保存为nii
    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / down_scale), ct.GetSpacing()[1] * int(1 / down_scale), slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], slice_thickness))

    sitk.WriteImage(new_ct, os.path.join(new_ct_dir, file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation')))

    print(index, file, ct_array.shape[0], 'already use {:.3f} min'.format((time() - start) / 60))

