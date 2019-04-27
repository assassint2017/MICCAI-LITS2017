# MICCAI-LITS2017
For the detail about the MICCAI-LITS 2017 challenge, you can check this link:
https://competitions.codalab.org/competitions/17094

I use VNet in this task. for the detail about the network architecture, you can check this link:  
https://arxiv.org/abs/1606.04797

## The change i make
Because the original Vnet have so many parameters, so it may suffer from overfitting, thus, I change the kernel size of each 3D convlayer to 3x3x3, and add dropout layer at some end of residual block. **and then I remove the last stage of the VNet since it is so coarse to help recover the segmentation detial, doing so, significantly reduced the receptive field, therefore, in order to compensate for the loss of receptive fields, I add some hybrid dilated convolution.** here I will show you some segmentation result I get: (blue represent ground truth, red represent the predict mask)

![reslut](https://github.com/assassint2017/MICCAI-LITS2017/blob/master/img/liver_seg.png)

## Implementation Details
I split the origin training set to 111 and 20 as my own training and test set. I use adam optimizer, set the initial learning rate to 1e-4 and decay 10 times at 50 epoch. The whole training process run on three GTX 1080Ti with batch size epual to three.

## Result 
I use dice per case as metrics, and find differenet inputs resolution affect the final result a lot, through a lot of experiment, the final input to the net is 256x256x48，with axial spacing norm to 2mm, and I get 0.957 Dice per case for liver segmentation at my test set.

|input resolution|slice spacing|expand slice|stride|Dice per case|
|:--:|:--:|:--:|:--:|:--:|
|128x128x32|3mm|5|5|0.895|
|128x128x32|3mm|15|3|0.914|
|256x256x32|3mm|15|3|0.932|
|256x256x48|2mm|20|3|0.957|

**since I remove the last stage of VNet, add dilated convolution and deep supervision, the performance has increased dramatically from 0.957 to 0.964**


## Next work
My net still have some overfit problem, so next work will foucs on using more powerful and efficient network and data augmentation like rotating or elastic deformation to get better result.

### TODO:
- [x] liver segmentation
- [x] better network
- [ ] data augmentation
- [ ] tumor segmentation
