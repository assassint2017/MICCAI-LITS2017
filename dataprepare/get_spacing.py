"""

查看数据轴向spacing分布
"""
import os
import SimpleITK as sitk

upper = 200
lower = -200

ct_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/CT/'
seg_path = '/home/zcy/Desktop/dataset/MICCAI-LITS-2017/test/seg/'
slice_thickness = 3


spacing_list = []

for ct_file in os.listdir(ct_path):

    ct = sitk.ReadImage(os.path.join(ct_path, ct_file), sitk.sitkInt16)
    temp = ct.GetSpacing()[-1]
    print(temp)
    spacing_list.append(temp)

print('mean:', sum(spacing_list) / len(spacing_list))

num = 0
for item in spacing_list:
    if item > 2.0:
        num += 1

print('num > 2:', num)

num = 0
for item in spacing_list:
    if item > 3.0:
        num += 1

print('num > 3:', num)

# 训练集中的平均spacing是1.47mm， 超过2mm有20例数据，超过3mm有10例数据
# 测试集中的平均spacing是1.73mm， 超过2mm有5例数据，超过3mm有2例数据
