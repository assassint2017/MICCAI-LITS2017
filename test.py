"""

分辨率256*256下的肝脏分割测试脚本
"""

import os
from time import time

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.measure as measure
import skimage.morphology as sm

from net.VNet_kernel3_dropoutv2 import VNet


test_ct_dir = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/CT/'
test_seg_dir = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'

liver_pred_dir = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/liverpred/'

module_dir = './test/net90-0.023.pth'

upper = 200
lower = -200
down_scale = 0.5
size = 48
slice_thickness = 2
threshold = 0.5
dice_list = []
time_list = []

# 定义网络并加载参数
net = torch.nn.DataParallel(VNet(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


for file in os.listdir(test_ct_dir):

    start = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(test_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = 200
    ct_array[ct_array < lower] = -200

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)

    # 在轴向上进行切块取样
    flag = False
    start_slice = 0
    end_slice = start_slice + size - 1
    ct_array_list = []

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

        start_slice = end_slice + 1
        end_slice = start_slice + size - 1

    # 当无法整除的时候反向取最后一个block
    if end_slice is not ct_array.shape[0] - 1:
        flag = True
        count = ct_array.shape[0] - start_slice
        ct_array_list.append(ct_array[-size:, :, :])

    outputs_list = []
    for ct_array in ct_array_list:

        ct_tensor = torch.FloatTensor(ct_array).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0)
        ct_tensor = ct_tensor.unsqueeze(dim=0)

        outputs = net(ct_tensor)
        outputs = outputs.squeeze()

        # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
        outputs_list.append(outputs.cpu().detach().numpy())
        del outputs

    # 执行完之后开始拼接结果
    pred_seg = np.concatenate(outputs_list[0:-1], axis=0)
    if flag is False:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=0)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][-count:]], axis=0)

    pred_seg = (pred_seg > threshold).astype(np.int16)

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage(os.path.join(test_seg_dir, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0).unsqueeze(dim=0)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().numpy()
    pred_seg = np.round(pred_seg).astype(np.uint8)

    # # 先进行腐蚀
    # pred_seg = sm.binary_erosion(pred_seg, sm.ball(5))

    # 取三维最大连通域，移除小区域
    pred_seg = measure.label(pred_seg, 4)
    props = measure.regionprops(pred_seg)

    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    pred_seg[pred_seg != max_index] = 0
    pred_seg[pred_seg == max_index] = 1

    pred_seg = pred_seg.astype(np.uint8)

    # # 进行膨胀恢复之前的大小
    # pred_seg = sm.binary_dilation(pred_seg, sm.ball(5))
    # pred_seg = pred_seg.astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    dice = (2 * pred_seg * seg_array).sum() / (pred_seg.sum() + seg_array.sum())
    dice_list.append(dice)

    print('file: {}, dice: {:.3f}'.format(file, dice))

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(liver_pred_dir, file.replace('volume', 'pred')))
    del pred_seg

    time_list.append(time() - start)

    print('this case use {:.3f} s'.format(time_list[-1]))
    print('-----------------------')


# 最后输出整个测试集的平均dice系数和平均处理时间
print('dice per case: {:.3f}'.format(sum(dice_list) / len(dice_list)))
print('time per case: {:.3f}'.format(sum(time_list) / len(time_list)))
