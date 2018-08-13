"""

Focal loss
同样用来处理分割过程中的前景背景像素非平衡的问题

使用时要将网络最后一层的softmax去掉，另外金标准的类型应该是LongTensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gama=2):
        """

        :param alpha: 论文中的正负样本的权重项，0.25是论文中的默认参数
        :param gama: 论文中的次方项，2是论文中的默认参数

        在默认参数的情况下效果也还可以，但是不如DiceLoss，所以要涉及到大量调参
        """
        super().__init__()

        self.gama = gama

        weight = torch.FloatTensor((1-alpha, alpha)).cuda()
        self.loss_func = nn.CrossEntropyLoss(weight=weight, reduce=False)

    def forward(self, pred, target):
        """
        :param pred: (B, 2, 48, 512, 512)
        :param target: (B, 48, 512, 512)
        """

        # 计算正样本的数量，这里所谓正样本就是属于器官的体素的个数
        num_target = target.sum()
        num_target = num_target.type(torch.cuda.FloatTensor)

        # 计算标准的交叉熵损失
        loss = self.loss_func(pred, target)

        pred = F.softmax(pred, dim=1)
        target = target.type(torch.cuda.FloatTensor)

        pos = pred[:, 1, :, :, :]

        # 对已经可以良好分类的数据的损失值进行衰减
        exponential_term = (1 - (pos * target + (1 - pos) * (1 - target))) ** self.gama
        loss *= exponential_term

        # 如果这一批数据中没有正样本，(虽然这样的概率非常小，但是还是要避免一下)
        if num_target == 0:

            # 则使用全部样本的数量进行归一化，和正常的CE损失一样
            return loss.mean()

        else:

            # 否侧用正样本的数量对损失值进行归一化
            return loss.sum() / num_target

