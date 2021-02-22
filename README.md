# NSD
Code and analysis of the NSD large scale fMRI dataset. We consider multiple model variant where the feature extractor varies, the connection model varies, and the training method vary to a method appropriate to the connection model. Some model, voxel-wise, were applied to whole brain activity whereas other, trained jointly, required a narrower scope.

## The fwRF encoding model
insert image of the equation representing the model here

### torched_alexnet_fwrf.ipynb
This notebook details the training process of the model based on a restricted alexnet feature extractor.

### torched_alexnet_fwrf_reload.ipynb
This notebook demonstrate reloading the saved parameters and test reproduction of the prediction validation accuracy. We also demonstrate how the pixel-gradient can be obtained.


## torched_gabor_fwrf.ipynb
This notebook details the training process of the model based on a gabor wavelet feature extractor for complex cells.

## torched_joined_gnet_encoding_multisubjecty.ipynb
This notebook details the joint training of the GNet feature extractor on V1-V4.
