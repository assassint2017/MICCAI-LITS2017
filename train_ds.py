"""

深度监督下的训练脚本
"""

from time import time
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss
from net.DialResUNet import net
from dataset.dataset_random import train_ds


# 定义超参数
on_server = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
cudnn.benchmark = True
Epoch = 3000
leaing_rate_base = 1e-4
alpha = 0.33
batch_size = 1 if on_server is False else 3
num_workers = 1 if on_server is False else 3
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义数据加载
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [1500])

# 训练网络
start = time()
for epoch in range(Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 is 0:
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    if epoch % 10 is 0 and epoch is not 0:

        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))

    if epoch % 15 is 0 and epoch is not 0:

        alpha *= 0.8

# 深度监督的系数变化
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000


