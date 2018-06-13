from time import time
import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loss.Focal_loss import FocalLoss
from net.VNet_kernel3 import net
from dataset.stage1_dataset import train_fix_ds


# 定义超参数
on_server = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '2,3'
cudnn.benchmark = True
Epoch = 100
leaing_rate_base = 1e-4
module_dir = './net19-0.015.pth'

batch_size = 1 if on_server is False else 6
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net).cuda()

# 定义数据加载
train_dl = DataLoader(train_fix_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = FocalLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 60])

# 训练网络
start = time()
for epoch in range(Epoch):

    lr_decay.step()

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        # 如果一个正样本都没有就直接结束
        if torch.numel(seg[seg > 0]) == 0:
            continue

        ct = Variable(ct)
        seg = Variable(seg)

        outputs = net(ct)
        loss = loss_func(outputs, seg)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 is 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.data[0], (time() - start) / 60))

    # 每十个个epoch保存一次模型参数
    if epoch % 10 is 0:
        torch.save(net.state_dict(), './module/net{}-{:.3f}.pth'.format(epoch, loss.data[0]))
