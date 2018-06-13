# MICCAI-LITS2017
for the detail about the MICCAI-LITS 2017 challenge, you can check this link:
https://competitions.codalab.org/competitions/17094

i use VNet in this task. for the detail about the network architecture, you can check this link:
https://arxiv.org/abs/1606.04797

beacuse the original Vnet have so many parameters, so it may suffer from overfitting, thus,  i change the kernel size of each 3D convlayer to 3*3*3, and add dropout layer at some end of residual block. here i will show you some segmentation result i get:(blue repressent ground truth, red repressent the predict mask)

![reslut](https://github.com/assassint2017/MICCAI-LITS2017/blob/master/img/liver_seg.png)

## next work
my net still have some overfit problem. so next work foucs on using more powerful and efficient network and data augmentation like rotating or elastic deformation to get better result.

### TODO:
- [x] liver segmentation
- [ ] better network
- [ ] data augmentation
- [ ] tumor segmentation
