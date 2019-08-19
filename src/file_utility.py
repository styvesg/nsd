import sys
import os
import struct
import time
import numpy as np
import scipy.io as sio
from scipy import ndimage as nd
from scipy import misc
from glob import glob
import h5py
import pickle
import math
import matplotlib.pyplot as plt
import PIL.Image as pim
import nibabel as nib



def save_stuff(save_to_this_file, data_objects_dict):
    failed = []
    with h5py.File(save_to_this_file+'.h5py', 'w') as hf:
        for k,v in data_objects_dict.iteritems():
            try:
                hf.create_dataset(k,data=v)
                print 'saved %s in h5py file' %(k)
            except:
                failed.append(k)
                print 'failed to save %s as h5py. will try pickle' %(k)   
    for k in failed:
        with open(save_to_this_file+'_'+'%s.pkl' %(k), 'w') as pkl:
            try:
                pickle.dump(data_objects_dict[k],pkl)
                print 'saved %s as pkl' %(k)
            except:
                print 'failed to save %s in any format. lost.' %(k) 
                

### NIFTY STUFF ###
def load_mask_from_nii(mask_nii_file):
    return nib.load(mask_nii_file).get_data()
    
def view_data(vol_shape, idx_mask, data_vol, order='C', save_to=None):
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    view_vol[idx_mask.astype('int').flatten()] = data_vol
    view_vol = view_vol.reshape(vol_shape, order=order)
    if save_to:
        nib.save(nib.Nifti1Image(view_vol, affine=np.eye(4)), save_to)
    return view_vol
###


def rgb2gray(im):
    return 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]

def center_crop(im):
    wax = np.argmax(im.size)
    border = (max(im.size) - min(im.size)) /2 
    if(wax==0):
        return im.crop(box=(border, 0, im.size[0]-border, im.size[1]))
    return im.crop(box=(0, border, im.size[0], im.size[1]-border))

### DIRECTORY MANIPULATION
def list_files(dir_path):
    fileNames = []
    for f in os.listdir(dir_path):
        if os.path.isfile(dir_path+f):
            fileNames += [dir_path+f,]
    return sorted(fileNames)

def list_dir(dir_path):
    dirNames = []
    for f in os.listdir(dir_path):
        if os.path.isdir(dir_path+f):
            dirNames += [f,]
    return sorted(dirNames)    


