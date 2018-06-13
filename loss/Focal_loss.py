"""

Focal loss
同样用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gama=2):
        super().__init__()

        self.alpha = alpha
        self.gama = gama
        self.loss_func = nn.BCELoss(reduce=False)

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        # 最后取平均loss时除的点个数，这里只选取正样本
        normalize_num = torch.numel(target[target > 0])

        # 首先计算标准的交叉熵损失
        loss = self.loss_func(pred, target)

        exponential_term = ((1 - pred) ** self.gama) * target + (pred ** self.gama) * (1 - target)
        weight_term = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss *= (exponential_term * weight_term)

        return loss.sum() / normalize_num
