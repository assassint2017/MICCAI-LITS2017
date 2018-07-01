from time import time
import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Dice_loss import DiceLoss
from VNet_dial import net
from dataset import train_stage1_ds


# 定义超参数
on_server = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '0,1,2'
cudnn.benchmark = True
Epoch = 100
leaing_rate_base = 1e-4
alpha = 0.33

batch_size = 1 if on_server is False else 3
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义数据加载
train_dl = DataLoader(train_stage1_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [50])

# 训练网络
start = time()
for epoch in range(Epoch):

    lr_decay.step()

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        ct = Variable(ct)
        seg = Variable(seg)

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        if epoch % 10 is 0 and epoch is not 0:
            alpha *= 0.8

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 is 0:
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], (time() - start) / 60))

    torch.save(net.state_dict(), './module/net{}-{:.3f}.pth'.format(epoch, loss.data[0]))
