"""

将测试集中的金标准转化为只包含肝脏区域的mask，方便进行查看
"""
import os

import SimpleITK as sitk

seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'
liver_seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/liverseg/'

for index, file in enumerate(os.listdir(seg_path)):

    seg = sitk.ReadImage(os.path.join(seg_path, file), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    seg_array[seg_array > 0] = 1

    liver_seg = sitk.GetImageFromArray(seg_array)

    liver_seg.SetDirection(seg.GetDirection())
    liver_seg.SetOrigin(seg.GetOrigin())
    liver_seg.SetSpacing(seg.GetSpacing())

    sitk.WriteImage(liver_seg, os.path.join(liver_seg_path, file.replace('segmentation', 'liver')))

    print(index)




