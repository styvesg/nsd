# NSD
Code and analysis of the NSD large scale fMRI dataset.

## The fwRF encoding model
insert image of the equation representing the model here

## torched_alexnet_fwrf.ipynb
This notebook details the training process of the model based on a restricted alexnet feature extractor. The models are saved in the following h5py files, one for each subject.

- S1: S01/dnn_fwrf_Mar-27-2020_0301/model_params.h5py
- S2: S02/dnn_fwrf_Apr-01-2020_0250/model_params.h5py
- S3: S03/dnn_fwrf_Apr-25-2020_1550/model_params.h5py
- S4: S04/dnn_fwrf_May-08-2020_2148/model_params.h5py
- S5: S05/dnn_fwrf_Mar-29-2020_2349/model_params.h5py
- S6: S06/dnn_fwrf_Mar-31-2020_0220/model_params.h5py
- S7: S07/dnn_fwrf_May-17-2020_1856/model_params.h5py
- S8: S08/dnn_fwrf_May-18-2020_2148/model_params.h5py

Currently, these files are on "NAS/styvesg/nsd_results/p3_full_brain_analysis/". Access will be granted on demand.

## torched_alexnet_fwrf_reload.ipynb
This notebook demonstrate reloading the saved parameters and test reproduction of the prediction validation accuracy. We also demonstrate how the pixel-gradient can be obtained.

## torched_gabor_fwrf.ipynb
This notebook details the training process of the model based on a gabor wavelet feature extractor for complex cells. The models are saved in the following h5py files, one for each subject.

- S1: S01/gabor_fwrf_May-24-2020_1943/model_params.h5py
...
