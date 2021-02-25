# NSD
Code and analysis of the NSD large scale fMRI dataset. We consider multiple model variant where the feature extractor varies, the connection model varies, and the training method vary to a method appropriate to the connection model. Some model, voxel-wise, were applied to whole brain activity whereas other, trained jointly, required a narrower scope due to memory restrictions.

All models can be mapped into the following diagram:

![Model diagram](model_diagram_paper.png | width=200)

## The fwRF encoding model
The fwRF utilize a fixed gaussian pooling field (panel B above, left) with 3 parameters trained via line-search on a fixed set of candidate. The feature tuning utilize a rich, multilayer, feature sets and it is trained with ridge regression. This method provides very effective regularization and, since it is voxelwise and takes advatage of massive parallelization of the model candidates, it is most suitable for large numbers of voxels (i.e. whole brain model regression).

### torched_alexnet_fwrf.ipynb
This notebook details the training process of the model based on a restricted alexnet feature extractor.

### torched_alexnet_fwrf_reload.ipynb
This notebook demonstrate reloading the saved parameters and test reproduction of the prediction validation accuracy. We also demonstrate how the pixel-gradient can be obtained for a block of voxels.

### torched_alexnet_fwrf_encoding_scaling.ipynb
An extension of the previous notebook to load a analyse multiple subject at once.

## The Gabor-based model
The gabor model utilize a set of wavelet filter pairs at various, log-spaced, spatial frequencies. It is effectively a single layer network under our general framework above and it is trained with the fwRF connection model.

### torched_gabor_fwrf.ipynb
This notebook details the training process of the model based on a gabor wavelet feature extractor for complex cells.

### torched_gabor_fwrf_encoding_scaling.ipynb
An extension of the previous notebook to load a analyse multiple subject at once.

## The GNet-based (data-driven) encoding model

### torched_joined_gnet_encoding_multisubject.ipynb
This notebook details the joint training of the GNet feature extractor on brain ROI V1-V4.
