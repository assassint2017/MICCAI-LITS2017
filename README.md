# liver segmengtation using deep learning
we use 3DResUnet to segment liver in CT images and use DenseCRF as post processing. I write as much comment as possible and hope you will find this reop useful!

## dataset
LiTS is a contain 131 CT images 
we train our model with and test on 3D dataset
and for the more detail, you can check this link:
https://competitions.codalab.org/competitions/17094

## Experiment
The whole traning process run on three GTX 1080Ti with batch size epual to three, Figure 1 show some of the segmentation resluts of our 3DResUNet eval on 3D dataset.

<div align=center><img src="https://github.com/assassint2017/MICCAI-LITS2017/blob/master/img/segmentation-result.png"alt="segmentation reslut"/></div>
<center><Figure 1></center>

Figure 2 show the loss curve draw by visdom.
<div align=center><img src="https://github.com/assassint2017/MICCAI-LITS2017/blob/master/img/loss_curve.png"alt="loss curve"/></div>
<center><Figure 2></center>
  
## Usage
i write all the parameter in parameter.py, so first set parameter in parameter.py and then run ./data_pareper/get_training_set.py to get training set then you can run ./train_ds.py to train the the network. after the model is trained, run val.py to test the model on test set, if you want to run ./Densecrf/3D-CRF.py or ./Densecrf/2D-CRF.py to 

## Main references:
1. Milletari F, Navab N, Ahmadi S A. V-net: Fully convolutional neural networks for volumetric medical image segmentation[C]//2016 Fourth International Conference on 3D Vision (3DV). IEEE, 2016: 565-571.
2. Wong K C L, Moradi M, Tang H, et al. 3d segmentation with exponential logarithmic loss for highly unbalanced object sizes[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018: 612-619.
3. Yuan Y, Chao M, Lo Y C. Automatic skin lesion segmentation using deep fully convolutional networks with jaccard distance[J]. IEEE transactions on medical imaging, 2017, 36(9): 1876-1886.
4. Salehi S S M, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks[C]//International Workshop on Machine Learning in Medical Imaging. Springer, Cham, 2017: 379-387.
5. Brosch T, Yoo Y, Tang L Y W, et al. Deep convolutional encoder networks for multiple sclerosis lesion segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2015: 3-11.
6. Xu W, Liu H, Wang X, et al. Liver Segmentation in CT based on ResUNet with 3D Probabilistic and Geometric Post Process[C]//2019 IEEE 4th International Conference on Signal and Image Processing (ICSIP). IEEE, 2019: 685-689.  
7. Krähenbühl P, Koltun V. Efficient inference in fully connected crfs with gaussian edge potentials[C]//Advances in neural information processing systems. 2011: 109-117.
