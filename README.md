# NSD
Code and analysis of the NSD large scale fMRI dataset.

## The fwRF encoding model
insert image of the equation representing the model here

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
