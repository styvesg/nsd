
# coding: utf-8

# ### Load, detrending, concatenating, and design matrix preparation for fMRI data
# 
# 
 

# #### Establish environment
# import packages we will need

import pandas as pd
import numpy as np
import nibabel as nib
import hrf_estimation as he

from scipy.stats.mstats import zscore
from os import path
from PIL import Image
from scipy import ndimage as ndi




##load on row of a database
def load_data(df_row):
    return nib.load(df_row['working_vol']) 




# #####Z-score / detrend / concatenate multiple voxels across multiple runs

def load_runs(db, runs, detrend = True, window_length = 0.5, poly_order=3, z_score=True):
    
    '''
    load_runs(db, runs, detrend = True, window_length = 0.5, poly_order=3, z_score=True)
    db   ~ pandas dataframe with standard column names
    runs ~ list on integer rund ids.
    detrend ~ True. apply detrending or not.
    window_length ~ as fraction of run length. ignore if no detrending
    poly_order ~ order of detrending filter. ignore if no detrending
    z_score ~ z-score each voxel / run independently, then concatenate.
    returns
        concat_img -- a concatenated nibabel spatial image. vols specified in "runs" stacked along temporal dimension.
    '''
    wvols = list(db.iloc[runs].working_vol)
    nvols = list(db.iloc[runs].nvols)
    starts = np.cumsum([0]+nvols[:-1])
    stops  = np.cumsum(nvols)
    concat_img = nib.funcs.concat_images(wvols, axis=3)
    if detrend:
        def get_wl(nv):
            wl = int(window_length*nv)
            if wl % 2 == 0:
                if wl+1 < nv: 
                    wl += 1
                elif wl-1 >= poly_order:
                    wl -= 1
            
            return wl
        for ii,_ in enumerate(wvols):
            wl = get_wl(nvols[ii])
            print 'detrending run %d with window_length: %d' %(ii,wl)
            time_course = concat_img.get_data()[:,:,:,starts[ii]:stops[ii]]
            time_course[:,:,:,:] = time_course - he.savitzky_golay.savgol_filter(time_course, wl, poly_order, axis=-1)
    if z_score:
        for ii,_ in enumerate(wvols):
            print 'zscoring %d thru %d' %(starts[ii],stops[ii])
            time_course = concat_img.get_data()[:,:,:,starts[ii]:stops[ii]]
            time_course[:,:,:,:] = zscore(time_course,axis=3)
            
    return concat_img


# Here is some code for interpreting "frame_files". These are listed in the database, and they are just a sequence of the stimuli presented during an experiment.
# Often the frames will be presented at a faster rate than the data were acquired. In this case we will need to downsample the stimuli so that we can associated one unique stimulus with
# each acquired data sample (volume). The following functions should be sufficient 

# #####simple function for loading a frame_file given a row of the database
def load_frame_file(df_row):    
    frame_file = path.join(df_row['frameFilePath'],df_row['frame_file'])
    with open(frame_file, 'r') as content_file:
        frame_list = content_file.read()
    frame_list = frame_list.strip().split('\n')
    return frame_list


# #####computes number of frames per volume. truncates frame_file if it doesn't divide neatly into number of samples (nvols)
def get_frames_per_volume(nvols, nframes):
    if nvols > nframes:
        Exception('more vols than frames. not sure what to do.')
    rem = np.remainder(nframes,nvols)
    if rem:
        print 'warning: frames not divisible by nvols. truncating frame file by %d frames' %(rem)
        nframes = nframes-rem
    frames_per_volume = nframes/nvols
    print 'current run had %d frames per sample'  %(frames_per_volume)
    return frames_per_volume
    


# #####downsamples a frame file, chunk by chunk. could also be used to downsample a sequence of feature values. fairly general
def chunkwise_downsampling(frame_list,chunksize,rule='first',ignore=None):
    '''
    chunkwise_downsampling(frame_list,chunksize,rule='first',ignore=None)
    
    frame_list ~ a temporally ordered list of elements that matches the presentation sequence during the experiment
    
    chunksize ~ number of the frames (=stimuli) presented while acquiring a single data sample
    
    rule ~ either 'first', 'last', 'max', 'min', 'median', or 'min'. this determines how each chunk is downsampled.
    note that if the elements of frame_list as strings (e.g., pointers to display images) only 'first' and 'last' make sense
    
    ignore ~ a list of frames to ignore. can be anything. we will check each chunk for items on this list and remove them.
    '''
    downsampled_frame_list = []
    if rule is 'first':
        reduce_chunk = lambda x: x[0]
    elif rule is 'max':
        reduce_chunk = np.max
    elif rule is 'last':
        reduce_chunk = lambda x: x[-1]
    elif rule is 'median':
        reduce_chunk = np.median
    elif rule is 'mean':
        reduce_chunk = np.mean
    elif rule is 'min':
        reduce_chunk = np.min
    ##and that's all I can think of
    for chunk in range(0,len(frame_list), chunksize):
        one_tr = frame_list[chunk:chunk+chunksize]
        try:
            for ig in ignore:
                if ig in one_tr:
                    one_tr.remove(ig)
        except:
            if ignore:
                one_tr.remove(ignore)
        downsampled_frame_list.append(reduce_chunk(one_tr))
    return downsampled_frame_list
        


# #####creates a binary conditions matrix from a frame file and a know sampling period (=TR)

def construct_conditions_matrix(downsampled_frame_list,sampling_period, hrf_length, not_a_condition=[None],basis='fir'):
    '''
    construct_conditions_matrix(downsampled_frame_list,sampling_period)
    wraps "create_design_matrix" from the "hrf_estimation" module
    
    downsampled_frame_list ~ a list of frames (=stimuli) that has the same length as the number of experimental samples
    (e.g., fmri volumes). 
    
    sampling_period ~ time per sample (= TR)
    
    hrf_length ~ number of samples per hrf
    
    not_a_condition ~ identify elements of the frame_list that should be treated as inter-stimulus intervals. must be a list or a tuple.
    
    basis = 'fir', 'hrf', or '3hrf'. these choices are part of the "hrf_estimation" module.
    
    returns:
    
    condition_map ~ a dictionary mapping frame names (i.e., the elements of the frame_list) to condition numbers
    
    condition_sequence ~ temporally ordered sequence of conditions. if not_a_condition = None, will be same length as frame_list
    
    condition_onsets ~ time (in units of sampling_period) at which each condition in condition_sequence occurred
    
    design_matrix ~ a time x conditions binary matrix. number of rows = length of frame_list

    
    '''
    from hrf_estimation import create_design_matrix
    nscans = len(downsampled_frame_list)
    hrf_length = hrf_length*sampling_period
    condition_map = dict() ## a map from image_ids to a condition number
    condition_number = 0   ##this will determine the design matrix column for each image
    condition_sequence = [] ##the sequence in which images were shown
    condition_onsets = []  ##the times (in seconds) when images were shown
    for tr_counter, ff in enumerate(downsampled_frame_list):
        if ff in not_a_condition:
            continue 
        if ff not in condition_map.keys():
            condition_map[ff] = condition_number
            condition_number += 1
        condition_sequence.append(condition_map[ff])
        condition_onsets.append(tr_counter*sampling_period)
    conditions_matrix,_ = create_design_matrix(condition_sequence,condition_onsets,TR = float(sampling_period),
                                                  n_scans=nscans,
                                                  basis = basis,
                                                  hrf_length=hrf_length)
    return condition_map, condition_sequence, condition_onsets,conditions_matrix

        




# #####median downsampling of the filter outputs
# In many cases, the frames specified by the frame file with not be perfectly synced to the onset of the sample acquisitions.
# In this case some frames will be split between two datasamples. For this reason, we use a "nearest_neighbor_downsampling" technique.

def nearest_neighbor_downsampling(timeseries_to_be_downsampled, length_of_downsampled_timeseries):
    '''
    timeseries_to_be_sampled is indexed at (nearly) uniform intervals so that has "length_of_downsampled_timeseris".
    
    rounding is into the past, because we assume this is a sensory experiment.
    
    use this if a movie has been presented and the frames don't exactly line up with the time-stamps of the acquired volumes
    '''
    fullness_of_time = len(timeseries_to_be_downsampled)
    idx = np.floor(np.linspace(0,fullness_of_time,num=length_of_downsampled_timeseries,endpoint=False)).astype(int)
    return timeseries_to_be_downsampled[idx]


# #####generate feature matrix (like "makeLags" in matlab)
def construct_feature_matrix(downsampled_filter_outputs,hrf_length = 10):
    '''
    downsampled_filter_outputs ~ time X features matrix of feature timeseries
    hrf_length ~ length (number of samples) of temporal kernel
    returns time X (features*hrf_length) matrix, where the additional columns are time-shifted versions of the features.
    '''
    feature_matrix = np.zeros((downsampled_filter_outputs.shape[0],downsampled_filter_outputs.shape[1]*hrf_length))
    cnt = 0
    for ii,column in enumerate(downsampled_filter_outputs.T):  ##transpose to loop over columns
        feature_matrix[:,cnt] = column
        cnt += 1
        for jj in range(1,hrf_length):
            feature_matrix[:,cnt] = np.pad(column,((jj,0)),mode='constant')[:-jj]
            cnt += 1
    return feature_matrix
        
