import sys
import os
import numpy as np
import h5py
from scipy.io import loadmat


def load_beta_file(filename, voxel_mask=None, zscore=True):
    from src.file_utility import load_mask_from_nii
    if ".mat" in filename:
        beta_data_set = h5py.File(filename, 'r')
        values = np.copy(beta_data_set['betas'])
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1), order='F').astype(np.float32) / 300.
        else:
            beta = values.reshape((len(values), -1), order='F')[:,voxel_mask.flatten()].astype(np.float32) / 300.
        beta_data_set.close()
    elif ".nii" in filename:
        values = load_mask_from_nii(filename).transpose((3,0,1,2))
        print (values.dtype, np.min(values), np.max(values), values.shape)
        if voxel_mask is None:
            beta = values.reshape((len(values), -1)).astype(np.float32) / 300.
        else:
            beta = values.reshape((len(values), -1))[:,voxel_mask.flatten()].astype(np.float32) / 300.       
        
    elif ".h5" in filename:
        print (".h5 not yet implemented")
        return None
    else:
        print ("Unknown file format")
        return None
    ###
    if zscore: 
        mb = np.mean(beta, axis=0, keepdims=True)
        sb = np.std(beta, axis=0, keepdims=True)
        beta = np.nan_to_num((beta - mb) / (sb + 1e-6))
        print ("<beta> = %.3f, <sigma> = %.3f" % (np.mean(mb), np.mean(sb)))
    return beta
     


def load_betas(folder_name, zscore=False, voxel_mask=None, up_to=0):
    '''load beta value in the structure of the NSD experiemnt'''
    from src.file_utility import list_files
    matfiles, betas = [], []
    k = 0
    for filename in list_files(folder_name):
        if ".mat" in filename:
            k += 1
            if up_to>0 and k>up_to:
                break
            print (filename) 
            matfiles += [filename,]  
            betas += [ load_beta_file(filename, voxel_mask=voxel_mask, zscore=zscore), ]       
    return np.concatenate(betas, axis=0), matfiles
