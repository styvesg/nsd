### DEFINITIONS ###
# time is the real time in second. Experiment starts is re-defined as time=0, which is the first sample in the recording.
# a frame is a presentation of any one of the stimulus or the ISI. It has a specified beginning and an end.
# a sample is a quasi-instataneous recording of the brain activity.
# an image index correspond to the entry in the Stimuli array, which include the ISI at index=0.

import numpy as np

def frame_to_start_time(frame, timing):
    return timing[frame]
    
def frame_to_end_time(frame, timing):
    return timing[frame+1] 

def frame_to_interval(frame, timing):
    return timing[frame+1] - timing[frame]

def frame_to_image_index(frame):
    if frame%2==0: # even, now correspond to image
        return int(frame//2 + 1)
    else: # odd, correspond to isi, indexed at 0 in array
        return int(0)



def sample_to_time(sample_index, sample_freq, time_start=0, sample_start=0):
    return np.float32(sample_index - sample_start) / sample_freq + time_start

def time_to_sample(time, sample_freq, time_start=0, sample_start=0):
    return int(np.floor((time - time_start) * sample_freq)) + sample_start



def time_to_frame(time, timing):
    '''return -1 if time requested is before the start of the experiment'''
    return int(np.sum((timing - time)<0) - 1)

def time_to_image_index(time, timing):
    return frame_to_image_index(time_to_frame(time, timing))
