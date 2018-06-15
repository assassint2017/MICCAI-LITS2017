"""

将测试集中的金标准转化为只包含肝脏和者肿瘤区域的mask，方便进行查看
"""
import os

import SimpleITK as sitk

seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'
liver_seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/liverseg/'
tumor_seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/tumorseg/'

for index, file in enumerate(os.listdir(seg_path)):

    seg = sitk.ReadImage(os.path.join(seg_path, file), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 转换肝脏
    liver_array = seg_array.copy()
    liver_array[seg_array > 0] = 1

    liver_seg = sitk.GetImageFromArray(liver_array)

    liver_seg.SetDirection(seg.GetDirection())
    liver_seg.SetOrigin(seg.GetOrigin())
    liver_seg.SetSpacing(seg.GetSpacing())

    sitk.WriteImage(liver_seg, os.path.join(liver_seg_path, file.replace('segmentation', 'liver')))

    # 转换肿瘤
    tumor_array = seg_array.copy()
    tumor_array[seg_array == 1] = 0
    tumor_array[seg_array == 2] = 1

    tumor_seg = sitk.GetImageFromArray(tumor_array)

    tumor_seg.SetDirection(seg.GetDirection())
    tumor_seg.SetOrigin(seg.GetOrigin())
    tumor_seg.SetSpacing(seg.GetSpacing())

    sitk.WriteImage(tumor_seg, os.path.join(tumor_seg_path, file.replace('segmentation', 'tumor')))

    print(index)
