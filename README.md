# NSD
Prototypes of task-driven and data-driven encoding models for the NSD large scale fMRI dataset. We consider multiple model variant where the feature extractor varies (Alexnet, Gabors and GNet), the connection model varies ([fwRF](https://github.com/styvesg/fwrf) and flexible), and the training method vary to a method appropriate to the case. Some model, voxel-wise, were applied to whole brain activity whereas other, trained jointly, required a narrower scope due to memory restrictions.

All models can be mapped into the following diagram:

<p align="center">
  <img src="model_diagram_paper.png" width="600">
</p>


 
## The Alexnet-based fwRF encoding model
The fwRF utilize a fixed gaussian pooling field (panel B above, right) with 3 parameters trained via grid-search on a fixed set of candidates. The feature tuning utilizes a rich, multilayer, feature sets and it is trained with ridge regression. This method provides very effective regularization and, since it is voxelwise and takes advatage of massive parallelization of the model candidates, it is most suitable for large numbers of voxels (i.e. whole brain model regression).

### torched_alexnet_fwrf.ipynb
This notebook details the training process of the model based on a restricted (only a predetermined number of features with high variance to the training set are selected) alexnet feature extractor.

### torched_alexnet_fwrf_reload.ipynb
This notebook demonstrate reloading the saved parameters and test reproduction of the prediction validation accuracy. We also demonstrate how the pixel-gradient can be obtained for a block of voxels.

### torched_alexnet_fwrf_encoding_scaling.ipynb
An extension of the previous notebook to load a analyse multiple subject at once.

## The Gabor-based fwRF model
The gabor model utilize a set of wavelet filter pairs at various, log-spaced, spatial frequencies. It is effectively a single layer network under our general framework above and it is trained with the fwRF connection model.

### torched_gabor_fwrf.ipynb
This notebook details the training process of the model based on a gabor wavelet feature extractor for complex cells.

### torched_gabor_fwrf_encoding_scaling.ipynb
An extension of the previous notebook to load and analyse multiple subject at once.

## The GNet-based (data-driven) encoding model
In this case, the feature extractor (referred to as GNet) has to be trained along with the connection model (referred to as flexible in panel B above, left) via stochastic gradient descent. While this incurs a cost in terms of regularization, it is also more expressive for voxels with high SNR, which is beneficial for the early visual cortex. However, this method requires joint training of multiple voxel target and is unsuitable for whole brain modelling (other than piecewise). This is why the notebook below focuses on ROI V1 to V4 on 4 subjects. Alternately, we can model V1 alone for 8 subjects, or the nsdgeneral mask for 2 subjects.


### torched_joined_gnet_encoding_multisubject.ipynb
This notebook details the joint training of the GNet feature extractor on brain ROI V1-V4.
