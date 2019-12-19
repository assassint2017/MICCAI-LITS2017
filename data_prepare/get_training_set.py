"""

获取可用于训练网络的训练数据集
需要四十分钟左右,产生的训练数据大小3G左右
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

import parameter as para


if os.path.exists(para.training_set_path):
    shutil.rmtree(para.training_set_path)

new_ct_path = os.path.join(para.training_set_path, 'ct')
new_seg_dir = os.path.join(para.training_set_path, 'seg')

os.mkdir(para.training_set_path)
os.mkdir(new_ct_path)
os.mkdir(new_seg_dir)

start = time()
for file in tqdm(os.listdir(para.train_ct_path)):

    # 将CT和金标准入读内存
    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 将金标准中肝脏和肝肿瘤的标签融合为一个
    seg_array[seg_array > 0] = 1

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, 1, 1), order=0)

    # 找到肝脏区域开始和结束的slice，并各向外扩张slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # 两个方向上各扩张slice
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

    # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
    if end_slice - start_slice + 1 < para.size:
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
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], para.slice_thickness))

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
