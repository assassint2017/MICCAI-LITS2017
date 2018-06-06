# MICCAI-LITS2017
for the detail about the MICCAI-LITS2017 challenge, you can check this link:
https://competitions.codalab.org/competitions/17094

i use VNet in this task. for the detail about the network architecture, you can check this link:
https://arxiv.org/abs/1606.04797

beacuse the original Vnet have so many parameter, so it may suffer from overfitting, so i change the kernel size of each 3D conv layer to 3*3*3, here are the result i get:

## TODO:
- [x] liver segmentation
- [ ] tumor segmentation
