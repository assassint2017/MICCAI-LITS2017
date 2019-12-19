"""

三维全连接条件随机场后处理优化
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import collections

import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

from utilities.calculate_metrics import Metirc

import parameter as para


file_name = []  # 文件名称

# 定义评价指标
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

# 为了计算dice_global定义的两个变量
dice_intersection = 0.0  
dice_union = 0.0


for file_index, file in enumerate(os.listdir(para.test_ct_path)):

    print('file index:', file_index, file, '--------------------------------------')
    
    file_name.append(file)

    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    pred = sitk.ReadImage(os.path.join(para.pred_path, file.replace('volume', 'pred')), sitk.sitkUInt8)
    pred_array = sitk.GetArrayFromImage(pred)

    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 灰度截断
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # 切割出预测结果部分，减少crf处理难度
    z = np.any(pred_array, axis=(1, 2))
    start_z, end_z = np.where(z)[0][[0, -1]]

    y = np.any(pred_array, axis=(0, 1))
    start_y, end_y = np.where(y)[0][[0, -1]]

    x = np.any(pred_array, axis=(0, 2))
    start_x, end_x = np.where(x)[0][[0, -1]]

    # 扩张
    start_z = max(0, start_z - para.z_expand)
    start_x = max(0, start_x - para.x_expand)
    start_y = max(0, start_y - para.y_expand)

    end_z = min(ct_array.shape[0], end_z + para.z_expand)
    end_x = min(ct_array.shape[1], end_x + para.x_expand)
    end_y = min(ct_array.shape[2], end_y + para.y_expand)

    new_ct_array = ct_array[start_z: end_z, start_x: end_x, start_y: end_y]
    new_pred_array = pred_array[start_z: end_z, start_x: end_x, start_y: end_y]

    print('old shape', ct_array.shape)
    print('new shape', new_ct_array.shape)
    print('shrink to:', np.prod(new_ct_array.shape) / np.prod(ct_array.shape), '%')

    # 定义条件随机场
    n_labels = 2
    d = dcrf.DenseCRF(np.prod(new_ct_array.shape), n_labels)

    # 获取一元势
    unary = np.zeros_like(new_pred_array, dtype=np.float32)
    unary[new_pred_array == 0] = 0.1
    unary[new_pred_array == 1] = 0.9

    U = np.stack((1 - unary, unary), axis=0)
    d.setUnaryEnergy(unary_from_softmax(U))

    # 获取二元势
    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(para.s1, para.s1, para.s1), shape=new_ct_array.shape)
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(para.s2, para.s2, para.s2), schan=(para.s3,), img=new_ct_array)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # 进行推理
    Q, tmp1, tmp2 = d.startInference()
    for i in tqdm(range(para.max_iter)):
        # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)

    # 获取预测标签结果
    MAP = np.argmax(np.array(Q), axis=0).reshape(new_pred_array.shape)

    liver_seg = np.zeros_like(seg_array, dtype=np.uint8)
    liver_seg[start_z: end_z, start_x: end_x, start_y: end_y] = MAP.astype(np.uint8)

    # 计算分割评价指标
    liver_metric = Metirc(seg_array, liver_seg, ct.GetSpacing())

    liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
    liver_score['jacard'].append(liver_metric.get_jaccard_index())
    liver_score['voe'].append(liver_metric.get_VOE())
    liver_score['fnr'].append(liver_metric.get_FNR())
    liver_score['fpr'].append(liver_metric.get_FPR())
    liver_score['assd'].append(liver_metric.get_ASSD())
    liver_score['rmsd'].append(liver_metric.get_RMSD())
    liver_score['msd'].append(liver_metric.get_MSD())

    dice_intersection += liver_metric.get_dice_coefficient()[1]
    dice_union += liver_metric.get_dice_coefficient()[2]

    # 将CRF后处理的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(liver_seg)
    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.crf_path, file.replace('volume', 'crf')))

    print('dice:', liver_score['dice'][-1])
    print('--------------------------------------------------------------')


# 将评价指标写入到exel中
liver_data = pd.DataFrame(liver_score, index=file_name)

liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
liver_statistics.loc['mean'] = liver_data.mean()
liver_statistics.loc['std'] = liver_data.std()
liver_statistics.loc['min'] = liver_data.min()
liver_statistics.loc['max'] = liver_data.max()

writer = pd.ExcelWriter('./result-post-processing.xlsx')
liver_data.to_excel(writer, 'liver')
liver_statistics.to_excel(writer, 'liver_statistics')
writer.save()

# 打印dice global
print('dice global:', dice_intersection / dice_union)
