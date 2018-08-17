"""

肝脏分割在自己的测试集下的脚本
"""

import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import xlsxwriter as xw
import SimpleITK as sitk
import scipy.ndimage as ndimage
import skimage.morphology as sm
import skimage.measure as measure


from net.VNet_dense import DialResUNet

on_server = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

val_ct_dir = '/mnt/zcy/val_data/CT/' if on_server is True else \
    '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/CT/'

val_seg_dir = '/mnt/zcy/val_data/GT/' if on_server is True else \
    '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg'

liver_pred_dir = '/mnt/zcy/val_data/liver_pred/' if on_server is True else \
    '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/liver_seg'

module_dir = './module/net1060-0.032-0.015.pth' if on_server is True else './net1060-0.032-0.015.pth'

upper = 200
lower = -200
down_scale = 0.5
size = 48
slice_thickness = 2
threshold = 0.7

dice_list = []
time_list = []


# 创建一个表格对象，并添加一个sheet，后期配合window的excel来出图
workbook = xw.Workbook('./result.xlsx')
worksheet = workbook.add_worksheet('result')

# 设置单元格格式
bold = workbook.add_format()
bold.set_bold()

center = workbook.add_format()
center.set_align('center')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(1, len(os.listdir(val_ct_dir)), width=15)
worksheet.set_column(0, 0, width=30, cell_format=center_bold)
worksheet.set_row(0, 20, center_bold)

# 写入文件名称
worksheet.write(0, 0, 'file name')
for index, file_name in enumerate(os.listdir(val_ct_dir), start=1):
    worksheet.write(0, index, file_name)

# 写入各项评价指标名称
worksheet.write(1, 0, 'liver:dice')
worksheet.write(2, 0, 'speed')
worksheet.write(3, 0, 'shape')


# 定义网络并加载参数
net = torch.nn.DataParallel(DialResUNet(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


for file_index, file in enumerate(os.listdir(val_ct_dir)):

    start = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape
    worksheet.write(3, file_index + 1, str(origin_shape))

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

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
    with torch.no_grad():
        for ct_array in ct_array_list:

            ct_tensor = torch.FloatTensor(ct_array).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            outputs = net(ct_tensor)
            #outputs = F.softmax(outputs, dim=1)
            #outputs = outputs[:, 1, :, :, :]
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

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0).unsqueeze(dim=0)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = (pred_seg > threshold).astype(np.int16)

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
    worksheet.write(1, file_index + 1, dice)

    print('file: {}, dice: {:.3f}'.format(file, dice))

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(liver_pred_dir, file.replace('volume', 'pred')))
    del pred_seg

    casetime = time() - start
    time_list.append(casetime)

    worksheet.write(2, file_index + 1, casetime)

    print('this case use {:.3f} s'.format(casetime))
    print('-----------------------')


# 输出整个测试集的平均dice系数和平均处理时间
print('dice per case: {}'.format(sum(dice_list) / len(dice_list)))
print('time per case: {}'.format(sum(time_list) / len(time_list)))

# 最后安全关闭表格
workbook.close()

