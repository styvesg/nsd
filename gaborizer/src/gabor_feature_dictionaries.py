import numpy as np
import pandas as pd
from theano import tensor as tnsr
from theano import function, scan
from time import time
from features import make_complex_gabor, make_gabor
from PIL import Image
from skimage.transform import resize


def make_gabor_table(orientations,deg_per_stimulus,cycles_per_deg,
                     freq_spacing='log',
                     pix_per_cycle=2,cycles_per_radius=1,diams_per_filter=2,complex_cell=True):
    
    
    '''
    Generates a table of parameters used to apply a gabor transform.
    This approach assumes that the gabors will be of fixed size (in pixels), but the image
    the gabors are applied to are downsampled to effectively determining the spatial frequency of the
    gabor.
    
    Returns everything needed to construct a stack of gabor filters.
    
    gbr_table,pix_per_filter, cyc_per_filter, envelope_radius =
                                  make_gabor_table(orientations,deg_per_stimulus,cycles_per_deg,
                                                   freq_spacing='log',
                                                   pix_per_cycle=2,
                                                   cycles_per_radius=1,
                                                   diams_per_filter=2,)
                                                   
    orientations     ~ number of linearly spaced orientations in [0,pi)
    deg_per_stimulus ~ given the stimulus size and viewing distance
    cycles_per_deg   ~ specify range of spatial frequencies as (lowest, highest, number) is cyc/deg.
    freq_spacing     ~ log or linear. spacing of spatial frequencies. 
    pix_per_cycle    ~ how many pixels will be used to depict one cycle. default = 2, i.e., the Nyquist limit.
                       Nyquist = 2 is fine for vert. or horz. orientations, but too jaggy (probably) for obliques.
                       but if too high, usually will require stimuli with larger than native resolution.
    cycles_per_radius~ determines radius of gaussian envelop.
                       we specify how many cycles per radius (= one stdev of gaussian envelope)
                       default = 1 = one cycle of the sinewave per std. of the gaussian envelope.
    diams_per_filter ~ determines the size of the filter. default = 2 = 4std. of the gaussian envelope.
    complex_cell     ~ default = True. if False, we include distinguish between filters with 0 and pi/2 phase
    
    returns
    gbr_table      ~ a pandas table with details of each gabor filter
    pix_per_filter ~ number of pixels per filter.  a constant.
    cyc_per_filter ~ number of cycles per filter.  a constant.
    envelope_radius~ number of pixels needed for one std. of the gaussian envelope. a constant.
            
    Note: depending on the deg_per_stimulus of your experiment, you will be limited to a certain range of 
    spatial frequencies. If too low, the filter will be larger than the downsampled image, which is kind of
    stupid. If too high, the image will have to be upsampled to obtain the required number of pixels per cycle,
    which is also stupid. The "full" range will have a lowest frequency where the image is downsampled to the 
    size of the filter, and a highest frequecy where the image is not downsampled at all. The larger the number
    of pixels per cycle, the smaller this range will be.
    
    '''
    oris = np.linspace(0, np.pi, num=orientations, endpoint=False).reshape(orientations)
    
    if freq_spacing == 'log':
        cycles_per_deg = np.logspace(np.log10(cycles_per_deg[0]),np.log10(cycles_per_deg[1]),num=cycles_per_deg[2])
    elif freq_spacing == 'linear':
        cycles_per_deg = np.linspace(cycles_per_deg[0],cycles_per_deg[1],num=cycles_per_deg[2])
    
    ##------Inferred from your choices
    ##radius of gaussian envelope of gabor filters in deg.
    envelope_radius = cycles_per_radius * (1./cycles_per_deg)
    
    ##radius of gaussian envelope of gabor filters in pixels
    envelope_radius_pix = pix_per_cycle * cycles_per_radius

    ##given the radius per filter, this is how many degrees the picture of the filter should be
    deg_per_filter = 2*envelope_radius * diams_per_filter


    #given pix/cyc, here's the # of pixels per stimulus
    pixels_per_stimulus = pix_per_cycle * cycles_per_deg * deg_per_stimulus

    ##given deg/filter and min pix/cyc, this is how big the filter should be (in pixels)
    pix_per_filter = deg_per_filter * pix_per_cycle * cycles_per_deg ##should be constant
    
    ##cycles per filter
    cycles_per_filter = 2*cycles_per_radius * diams_per_filter ##should be constant
    
    


    metrics = {'cycles per deg.': cycles_per_deg,      ##len = Df
               'pix per stimulus' : pixels_per_stimulus,
               'radius of Gauss. envelope (deg)': envelope_radius,
               'filter size (deg.)': deg_per_filter,
               'pix_per_filter': np.round(pix_per_filter).astype('int'),
              'cycles_per_filter': cycles_per_filter}
    
    freq_table = pd.DataFrame(metrics)
    if not complex_cell:
        freq_table['phase'] = 0
        other_freq_table = freq_table.copy()
        other_freq_table['phase'] = np.pi/2.
        freq_table = pd.concat([freq_table,other_freq_table],axis=0,ignore_index=True)
        
    freq_table['orientation'] = oris[0]
    tmp_freq_table = freq_table.copy()
    for o in oris[1:]:
        tmp_freq_table['orientation'] = o
        freq_table = pd.concat([freq_table,tmp_freq_table],axis=0,ignore_index=True)
        
    
    return (freq_table,
            pix_per_filter[0],     ##<<only need 1st one because they are all the same.
            cycles_per_filter,  
            envelope_radius_pix,) 

                                                   
                                                   
                                                   
def make_gabor_stack(gbr_table, pix_per_filter, cycles_per_filter, envelope_radius_pix, complex_cell=True,color_channels=1):
    
    ##initialize
    filter_stack = np.zeros((gbr_table.shape[0],color_channels,int(pix_per_filter), int(pix_per_filter)))
    if complex_cell:
        filter_stack = filter_stack+1j
    
    ##args to gaborme
    center = (0,0)
    freq = cycles_per_filter
    radius = np.float32(envelope_radius_pix)
    n_pix = pix_per_filter.astype('int')
    for ii,tx in enumerate(gbr_table.index):
        ori = gbr_table.loc[tx,'orientation'] 
        for c in range(color_channels):
            if complex_cell:
                filter_stack[ii,c,:,:] = make_complex_gabor(freq,ori,center,radius,n_pix)
            else:
                ph = gbr_table.loc[ii,'phase']
                filter_stack[ii,c,:,:] = make_gabor(freq,ori,ph,center,radius,n_pix)
    return filter_stack


##here we use theano to construct a function that applies gabor filters to images.
##it can be either simple or complex cell-like.
def make_apply_gabor_function(filter_stack_shape,complex_cell=True):
    stim_tnsr = tnsr.tensor4('stim_tnsr')  ##T x n_color_channels x stim_size x stim_size
    real_filter_stack_tnsr = tnsr.tensor4('real_feature_map_tnsr') ##D x n_color_channels x stim_size x stim_size. complex
    imag_filter_stack_tnsr = tnsr.tensor4('imag_feature_map_tnsr') ##D x n_color_channels x stim_size x stim_size. complex
    real_feature_map_tnsr = tnsr.nnet.conv2d(stim_tnsr,
                                     real_filter_stack_tnsr,                                     
                                     filter_shape = filter_stack_shape,
                                     border_mode = 'full')  ##produces T x D x stim_size x stim_size maps
    imag_feature_map_tnsr = tnsr.nnet.conv2d(stim_tnsr,
                                     imag_filter_stack_tnsr,
                                     filter_shape = filter_stack_shape,
                                     border_mode = 'full')  ##produces T x D x stim_size x stim_size maps

    
    if complex_cell:
        ##for filtering with complex gabors, we need an operation for squaring/summing real/imag parts
        abs_value = tnsr.sqrt(tnsr.sqr(real_feature_map_tnsr) + tnsr.sqr(imag_feature_map_tnsr))
        ##functionize feature mapping
        make_feature_maps = function(inputs = [stim_tnsr,real_filter_stack_tnsr,imag_filter_stack_tnsr],
                                     outputs = abs_value)
    else:
        make_feature_maps = function(inputs = [stim_tnsr,real_filter_stack_tnsr],
                                     outputs = real_feature_map_tnsr)

    return make_feature_maps

def create_gabor_feature_map(image_stack,filter_stack,freq_table,complex_cell=True,interp_order=3):
    '''
    image_stack ~ T x n_colors x s_pix x s_pix
    filter_stack ~ D x n_colrs x f_pix x f_pix
    
    '''
    
    ##initialize feature dictionary
    T = image_stack.shape[0]
    n_color_channels = image_stack.shape[1]
    feature_indices = freq_table.index
    feature_dict = {}    
    
    ##this will be a theano function
    apply_filter = make_apply_gabor_function(filter_stack.shape, complex_cell=complex_cell)

    ##allocate memory first
    print 'allocating memory for feature maps'
    for ii in feature_indices:
        n_pix = np.round(freq_table.loc[ii,'pix per stimulus']).astype('int')
        feature_dict[ii] = np.zeros((T,n_color_channels,n_pix,n_pix)).astype('float32')
    
    print 'constructing feature maps'
    for ii,fidx in enumerate(feature_indices):
        this_filter = filter_stack[ii,np.newaxis,:,:,:]
        start = time()
        n_pix = feature_dict[ii].shape[2]  ##resolution of the feature map
        stimuli = np.zeros((T,n_color_channels,n_pix,n_pix)).astype('float32')
        for t in range(T):
            for c in range(n_color_channels):
                stimuli[t,c,:,:] = np.array(Image.fromarray(image_stack[t,c,:,:]).resize((n_pix,n_pix)),dtype='float32') #resize(image_stack[t,c,:,:], (n_pix,n_pix),order=interp_order)
        if complex_cell:
            tmp_feature_map = apply_filter(stimuli,
                                            np.real(this_filter).astype('float32'),
                                            np.imag(this_filter).astype('float32'))
        else:
            tmp_feature_map = apply_filter(stimuli,this_filter.astype('float32'))
        
        ##crop because convolution
        new_size = tmp_feature_map.shape[2]
        crop_start = np.round((new_size-n_pix)/2.).astype('int')
        crop_stop = crop_start+n_pix
        feature_dict[ii][:,:,:,:] = np.copy(tmp_feature_map[:, :, crop_start:crop_stop, crop_start:crop_stop])
        print 'feature %s took %f s.' %(ii,time()-start)
        
    return feature_dict


class gabor_feature_maps(object):
    def __init__(self,orientations,deg_per_stimulus,cycles_per_deg,
                     freq_spacing='log',
                     pix_per_cycle=2,cycles_per_radius=1,diams_per_filter=2,complex_cell=True,color_channels=1):
        self.number_of_orientations = orientations
        self.deg_per_stimulus = deg_per_stimulus
        self.lowest_freq = cycles_per_deg[0],
        self.highest_freq = cycles_per_deg[1],
        self.num_sp_freq = cycles_per_deg[2],
        self.freq_spacing = freq_spacing,
        self.pix_per_cycle = pix_per_cycle
        self.cycles_per_radius = cycles_per_radius,
        self.diams_per_filter = diams_per_filter
        self.complex_cell = complex_cell
        self.color_channels=color_channels
        (self.gbr_table,
         self.pix_per_filter,
         self.cycles_per_filter,
         self.envelope_radius_pix) = make_gabor_table(orientations,
                                                     deg_per_stimulus,
                                                     cycles_per_deg,
                                                     freq_spacing,
                                                     pix_per_cycle,
                                                     cycles_per_radius,
                                                     diams_per_filter,
                                                     complex_cell=self.complex_cell)
	 
	
        self.filter_stack = make_gabor_stack(self.gbr_table,
                                             self.pix_per_filter,
                                             self.cycles_per_filter,
                                             self.envelope_radius_pix,
                                             color_channels=color_channels,
                                             complex_cell = self.complex_cell)
    
    
    ###if 
    def create_feature_maps(self,image_stack,interp_order=3):
	'''
	image_stack ~ T x n_colors x s_pix x s_pix
	filter_stack ~ D x n_colrs x f_pix x f_pix
	
	'''
        return create_gabor_feature_map(image_stack,
                                   self.filter_stack,
                                   self.gbr_table,
                                   complex_cell=self.complex_cell,
                                   interp_order=interp_order)
    
    
    def sensitivity(self,feat_dict,parameter):
        '''
        given a feature dictionary produced by current instance, returns mean response per
        parameter. can be used to get spatial freq. or orientation or phase sensitivity of population.
        returns param_values, mean_response for plotting like
        plot(param_values, mean_response)
        '''
        
        param_group = self.gbr_table.groupby(parameter)
        mean_resp = np.zeros(len(param_group))
        param_values = np.zeros(len(param_group))
        ii = 0
        for name,grp in param_group:
            idx = grp.index
            mean_resp[ii] = np.mean(map(lambda x: np.mean(feat_dict[x]), idx))
            param_values[ii] = name
            ii += 1
        return param_values, mean_resp
