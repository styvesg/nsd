# NSD
Code and analysis of the NSD large scale fMRI dataset.

## The fwRF encoding model

The model is
$$ r(t) = b + W * [f(\int_\mathrm{space}\phi(x,y,t) * g(x,y) dxdy) - m] / \sigma $$
where
$g(x,y)$ is a gaussian pooling field shared by all feature maps

$\phi(x,y,t)$ are the feature maps corresponding to stimuli $t$

$W, b$ are the feature weights and bias of the linearized model for each voxels

$f(\cdot)$ is an optional nonlinearity

$m,\sigma$ are normalization coefficient to facilitate regularization

## torched_alexnet_fwrf.ipynb
This notebook details the training process of the model. The models are saved in the following h5py files, one for each subject.

- S1: dnn_fwrf_Mar-27-2020_0301_params.h5py
- S2: dnn_fwrf_Apr-01-2020_0250_params.h5py
- S3:
- S4:
- S5: dnn_fwrf_Mar-29-2020_2349_params.h5py
- S6: dnn_fwrf_Mar-31-2020_0220_params.h5py
- S7:
- S8:

## torched_alexnet_fwrf_reload.ipynb
This notebook demonstrate reloading the saved parameters and test reproduction of the prediction validation accuracy. We also demonstrate how the pixel-gradient can be obtained.
