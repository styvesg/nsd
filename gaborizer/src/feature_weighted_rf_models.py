import numpy as np
import pandas as pd
from theano import tensor as tnsr
from theano import function
from hrf_fitting.src.features import make_gaussian, compute_grid_corners, compute_grid_spacing, construct_placement_grid
from itertools import product
from warnings import warn
from time import time


##===================Theano expressions and functions===================

##-----model space-----

#theano
rf_stack_tnsr = tnsr.tensor3('rf_stack_tnsr') ##G x stim_size x stim_size
feature_map_tnsr = tnsr.tensor4('feature_map_tnsr') ##T x D x stim_size x stim_size

apply_rf_to_feature_maps = function(inputs = [rf_stack_tnsr,feature_map_tnsr],
                                    outputs = tnsr.tensordot(rf_stack_tnsr,
							     feature_map_tnsr,
							     axes=[[1,2], [2,3]]))

#example python use case
#model_space = apply_rf_to_feature_maps(rf_stack, feature_maps)

##-----prediction menu----- (uses batched_tensordot. not sure why this is necessary, but memory error if normal tensordot is used.)
model_space_tnsr = tnsr.tensor3('X')      ##model-space tensor: G x T x D
feature_weight_tnsr = tnsr.tensor3('NU')  ##feature weight tensor: G x D x V
prediction_menu_tnsr = tnsr.batched_tensordot(model_space_tnsr,
                                              feature_weight_tnsr,
                                              axes=[[2],[1]]) ##prediction tensor: G x T x V
bigmult = function([model_space_tnsr,feature_weight_tnsr], prediction_menu_tnsr)

##example python use case
##prediction_menu = bigmult(model_space,feature_weights)  ##G x T x V


###-----error menu-----
voxel_data_tnsr = tnsr.matrix('voxel_data_tnsr')  ##voxel data tensor: T x V
diff = voxel_data_tnsr-prediction_menu_tnsr  ##difference tensor: (T x V) - (G x T x V) = (G x T x V)
sq_diff = (diff*diff).sum(axis=1) ##sum-sqaured-diffs tensor: G x V
sq_diff_func = function(inputs=[voxel_data_tnsr,prediction_menu_tnsr],
                        outputs = sq_diff)  

##example python use case
##error_menu = sq_diff_func(voxel_data,prediction_menu)

###-----gradient menu-----
SQD_sum = sq_diff.sum()  ##<<this is critical
grad_SQD_wrt_NU = tnsr.grad(SQD_sum,feature_weight_tnsr) ##<<the summing trick above makes this easy. 
compute_grad = function(inputs = [voxel_data_tnsr,model_space_tnsr,feature_weight_tnsr],
                        outputs=grad_SQD_wrt_NU)




##===================Python classes and functions===================

##make a table describing the sizes and locations of the receptive fields
def make_rf_table(deg_per_stim,deg_per_radius,spacing,pix_per_stim = None):
    '''
    here is the machinery for setting up grid of rfs
    includes machinery for downsampling to different pixel resolutions
    
    make_rf_table(deg_per_stim,deg_per_radius,spacing,pix_per_stim = None)
    
    deg_per_stim   ~ scalar, determined by experiment
    deg_per_radius ~ (min_rad, max_rad, num_rad) specify the range rf sizes
    spacing        ~ scalar, spacing between rfs in deg
    pix_per_stim   ~ integer, default = None. If defined, add columns to rf_table with rf dimensions in pixels.
    returns
        rf_table   ~ pandas dataframe, each row an rf with columns 'deg_per_radius', 'x_deg','y_deg'
                     all units in deg. relative to origin of feature map = (0,0)
                     If pix_per_stim given, add columns 'pix_per_radius' and 'x_pix', 'y_pix' 
    '''
    n_sizes = deg_per_radius[2]
    rf_radii_deg = np.linspace(deg_per_radius[0],deg_per_radius[1],num=n_sizes,endpoint=True)
    
    corners = compute_grid_corners(deg_per_stim, 0, boundary_condition=0) ##<<puts center of stim at (0,0)
    x_deg,y_deg = construct_placement_grid(corners,spacing)
    
    
    number_of_rfs = x_deg.ravel().size*rf_radii_deg.size
    rf_array = np.zeros((number_of_rfs,3))
    all_rfs = product(rf_radii_deg,np.concatenate((x_deg.ravel()[:,np.newaxis], y_deg.ravel()[:,np.newaxis],),axis=1))
    
    for ii,rf in enumerate(all_rfs):
        rf_array[ii,:] = np.array([rf[0],rf[1][0],rf[1][1]])
    
    rf_table = pd.DataFrame(data=rf_array, columns=['deg_per_radius', 'x_deg', 'y_deg'])
    
    if pix_per_stim:
        scale_factor = lambda row: row*pix_per_stim * (1./deg_per_stim) 
        rf_table['pix_per_radius'] = rf_table['deg_per_radius'].apply(scale_factor)
        rf_table['x_pix'] = rf_table['x_deg'].apply(scale_factor)
        rf_table['y_pix'] = rf_table['y_deg'].apply(scale_factor)
    
    return rf_table


##grids of receptive fields
class receptive_fields():
    def __init__(self,deg_per_stim, deg_per_radius, spacing):
      
        '''
        receptive_fields(deg_per_stim, deg_per_radius, spacing)
        construct a receptive_field object
	deg_per_stim   ~ scalar, determined by experiment
	deg_per_radius ~ (min_rad, max_rad, num_rad) specify the range rf sizes
	spacing        ~ scalar, spacing between rfs in deg
	
	calls make_rf_table, stores a table of receptive field attributes
	

        
        '''
        self.deg_per_stim = deg_per_stim
        self.deg_per_radius = deg_per_radius
        self.spacing = spacing
        self.rf_table = make_rf_table(deg_per_stim,deg_per_radius,spacing)
        self.G = self.rf_table.shape[0]
    

    def make_rf_stack(self, pix_per_stim,min_pix_per_radius=None):
        '''
        make_rf_stack(pix_per_stim,min_pix_per_radius=None)
        construct stack of rfs at specified resolution
   
        pix_per_stim ~ scalar, determined by resolution of feature map
        min_pix_per_stim ~ scalar. if rf radius has few than this pixels at desired resolution, return 0's for that rf (i.e., a picture of nothing)
        
        returns G x S x S tensor of pictures of gaussian rf blobs
        '''
        ##these are cheap to make, so just rebuild it with added pixel columns. that way it won't accidentally get saved
        rf_table_pix = make_rf_table(self.deg_per_stim,self.deg_per_radius,self.spacing,pix_per_stim=pix_per_stim)
        rf_sizes = rf_table_pix['deg_per_radius'].unique()
        
        too_small = np.array(map(lambda x: min_pix_per_radius > x, rf_table_pix['pix_per_radius'].unique())).astype('bool')
        
        if np.any(too_small):
#             warn("some rf sizes are too small for resolution %d" %(pix_per_stim))
            print "at pixel resolution %d the following rfs will default to 0: %s" %(pix_per_stim,(rf_sizes[too_small],))
                
        rf_grid = np.zeros((self.G, pix_per_stim, pix_per_stim))
        for cnt,rf in enumerate(rf_table_pix.iterrows()):
            center = (rf[1]['x_pix'],rf[1]['y_pix'])
            rad = rf[1]['pix_per_radius']
            if not (rad < min_pix_per_radius): ##will fail if min_pix = None or if test fails
                rf_grid[cnt,:,:] = make_gaussian(center,rad,pix_per_stim) ##if rf too small, default to 0.
                
        return rf_grid
 
 
##the essential model_space class
class model_space():
    '''
    on init, commits to a feature_dictionary and a receptive_fields instance
    records feature depth and resolutions and number of rf models but doesn't commit to 
    a particular set of stimuli
    
    knows how to generate and apply rf_stack to feature maps in the dictionary. 
    enforces the "min_pix_per_radius" constraint.
    
    shits out a 3D model_space_tensor.
    
    complains if dimensions/names of feature_dict doesn't match what it has already recorded.
    
    after training/model selection, these objects used to interpret models and generate predictions.
    
    
    '''
    
    def __init__(self, feature_dict, rf_instance, min_pix_per_radius=1, activation_function=None):
        '''
        
        model_space(feature_dict, rf_instance, min_pix_per_radius=1)
        
        feature_dictionary ~ dictionary of T x Di x Si x Si feature map tensors.
                             T = integer, # of time-points (or trials, or sitmuli), constant for all features
                             Di = feature depth, may vary across keys in dict.
                             Si is feature map resolution in pixels. it may vary across keys.
               rf_instance ~ instance of receptive_fields class
          min_pix_per_stim ~ scalar, default = 1. don't consider rf's with fewer pixels than this.
                             rf's will have to be downsampled to be applied to some feature maps. if rf
                             has fewer than this number of pixels, returns a feature map of all 0's
                 
	parses the feature dictionary and stores feature names, resoltions, and indices into concatentated model_space_tensor
	stores information about receptive fields
	
        '''
        self.min_pix_per_radius = min_pix_per_radius
        self.receptive_fields = rf_instance
        
        if activation_function:
	  self.activation_function = activation_function

        
        ##get feature depths, indices, resolutions
        self.feature_depth = {}
        self.feature_indices = {}
        self.feature_resolutions = {}
        idx = 0
        for f_key in feature_dict.keys():
            self.feature_depth[f_key] = feature_dict[f_key].shape[1]
            self.feature_indices[f_key] = np.arange(idx,idx + self.feature_depth[f_key],step=1)
            idx += self.feature_depth[f_key]
            self.feature_resolutions[f_key] = feature_dict[f_key].shape[2]
        
        ##total feature depth
        self.D = np.sum(self.feature_depth.values())
        
 
    
    def normalize_model_space_tensor(self, mst,save=False):
        '''
        normalize_model_space_tensor(self, mst,save=False):
        z-score each feature of each model in a model_space_tensor across time.
        
        if normalization_constants is already defined as an attribute, apply to mst.
        
        otherwise, if save=True, calculate mean and stdev. from mst provided then apply to tensor
        and store the calculated mean and std
        
        otherwise, complain and die
        
        '''
      
        if hasattr(self, 'normalization_constant'):
            mn = self.normalization_constant[0]
            stdev = self.normalization_constant[1]
            if save:
                warn('not saving because constants are already defined')
        elif save: 
            mn = np.expand_dims(np.mean(mst,axis=1),axis=1)
            stdev = np.expand_dims(np.std(mst,axis=1),axis=1)
            self.normalization_constant = []
            self.normalization_constant.append(mn)
            self.normalization_constant.append(stdev)
            print 'normalization constants have been saved'
        else:
            raise Exception('if you want to compute the mean and stdev from the current data, you have to commit to saving it as an attribute')
        
        ##z-score 
        mst -= mn
        mst /= stdev
        
        ##convert nans to 0's because there are feature/rf pairs where the feature map is too low-res for the rf to be meaningful
        try:
	  mst = np.nan_to_num(mst)
	  print 'converted nans to nums'
	except:
	  print 'huh?'
	  1/0
        
        print 'model_space_tensor has been z-scored'
        return mst
    
        
    
    def construct_model_space_tensor(self,feature_dict,normalize=True):
        '''
        construct_model_tensor(feature_dict)
        
        checks feature_dict for appropriate keys/resolutions
        
        allocates memory for model_space_tensor
        
        loop over keys in feature dictionary
        feature maps for each key have potentially unique resolution, so call make_rf_grid for each        
        call theano function "apply_rf_to_feature_maps" for each map in dictionary
        concatentates across features to form a model_space_tensor
        
        will normalize model space (z-score each rf/feature row across time) by default. note: you have to explicity
        commit to normalization by running the normalize method on whatever you consider your training data to be "save=True".
        until you've done that you won't be able to apply normalization. so typically you will running
        
        training_mst = model_space.construct_model_space_tensor(training_feature_dict, normalize = False)
        training_mst = model_space.normalize_model_space_tensor(training_mst, save=True)
        
        mean/stdev. now stored and all subsequent calls to "construct_model_space_tensor" will be normalized by default.
        
       returns
        model_space_tensor ~ G x T x D tensor.
                             D = sum(Di), total feature depth across all keys in the feature dictionary
                             G = size of rf grid, or, the number of rf models we consider.
                             each (D,T) plane give time-series for the D features after filtering by one of the G rf's.
        '''
        
        ##check feature_dict for proper names/resolutions
        key_list = self.feature_depth.keys()
        for f_key in feature_dict.keys():
            if f_key in key_list:
                key_list.remove(f_key)
            else:
                raise ValueError("this feature dictionary doesn't match your model")
        
        
        ##determine T = number of time points / trials / stimuli . if T is not same for all keys freak out
        all_Ts = map(lambda k: feature_dict[k].shape[0],feature_dict.keys())
        if np.any(map(lambda x: all_Ts[0] != x, all_Ts)):
            raise ValueError('temporal dimensiosn of feature map are not equal: %s' %(all_Ts,))
        else:
            self.T = all_Ts[0]
        
        ##allocate memory for model space
        mst = np.zeros((self.receptive_fields.G, self.T, self.D),dtype='float32')
        
        ##loop over keys in feature dictionary
        for feats in feature_dict.keys():
	    print '-----------feature: %s' %(feats)
            rf_stack = self.receptive_fields.make_rf_stack(self.feature_resolutions[feats],min_pix_per_radius=self.min_pix_per_radius).astype('float32')
            mst[:,:,self.feature_indices[feats]] = apply_rf_to_feature_maps(rf_stack,feature_dict[feats])
            
        ##apply activation function if it exist
	if hasattr(self, 'activation_function'):
	  mst = self.activation_function(mst)
            
        ##
        if normalize:    
            mst = self.normalize_model_space_tensor(mst,save=False)  ##save = false so will throw error unless
                                                                     ##you've already stored normalization_constants
                
        return mst
 
 
##function for generating predictions
def prediction_menu(model_space_tensor, feature_weights, rf_indices=None):   
    '''
    prediction_menu(model_space_tensor, feature_weights, rf_indices=None)

    model_space_tensor ~ G x T x D   
       feature_weights ~ G x D x V, or 1 x D x V. If the latter, rf_indices = list of length = V.
 
     if rf_indices=None, returns G x T x V prediction menu tensor.
     otherwise,              returns T x V prediction menu tensor.
    '''
    G = model_space_tensor.shape[0] 
    V = feature_weights.shape[2]
    if G != feature_weights.shape[0]:
        feature_weights = np.tile(feature_weights,[G,1,1])
    
    
    pmt = bigmult(model_space_tensor, feature_weights)
    
    ##if rf_indices defined, select along G dimension and then diagonalize.
    if rf_indices != None:
        pmt = np.diagonal(pmt[rf_indices],axis1=0,axis2=2)
    

    return pmt
    

##--training function
def train_fwrf_model(model_space_tensor, voxel_data,
		     initial_feature_weights = 'zeros',
                     early_stop_fraction = 0.2,
                     max_iters=100,
                     mini_batch_size = 0.1,
                     learning_rate=10**(-5),
                     voxel_binsize=100,
                     rf_grid_binsize = 200,
                     report_every = 10):

    '''
    train_fwrf_model(model_space_tensor, voxel_data,
		     initial_feature_weights = 'zeros',
                     early_stop_fraction = 0.2,
                     max_iters=100,
                     mini_batch_size = 0.1,
                     learning_rate=10**(-5),
                     voxel_binsize=100,
                     rf_grid_binsize = 200,
                     report_every = 10)
    
    for each of V voxels, find the optimal feature weights associated with each of G receptive field models, and return the receptive field model and associated feature weights with
    the lowest error on a held-out set of "early stopping" (a.k.a. validation) data.
    
    inputs
    
	 model_space_tensor ~ G x T x D tensor, G = rf models, D = features, T = time/trials
		 voxel_data ~ T x V array of neural data, V ~ voxels
    initial_feature_weights ~ default = 'zeros', in which case set all to 0. otherwise, a G x D x V array of initial guesses at feature weights (if you have the memory to spare)
	early_stop_fraction ~ amount of your training data reserved for early stopping. this subset called the "validation data".
			      error reports refer to error on validation data. gradients are estimated on the remaining fraction, i.e., the "training data"
		  max_iters ~ the only way to get this thing to stop. just go for as long as you have to wait.
	      learning_rate ~ size of gradient step. mess with it until you get it right.
	      voxel_binsize ~ how many voxels to train at one time
	    rf_grid_binsize ~ how many rf models to evaluate at one time
	       report_every ~ print a summary of error every this-many iterations
   
   outputs
   
   final_validation_loss ~ V. for each voxel and each rf, the minimum validation loss.
   final_feature_weights ~ D x V. for each voxel, the feature weights associated with the best rf model for that voxel
		final_rf ~ V. index of best rf model for each voxel
      best_error_history ~ max_iters x V. the error trajectory for the best rf model for each voxel. the min of this trajectory = final_validation_loss
    
    '''

    ##basic dimenisions
    G,T,D = model_space_tensor.shape ##G = size of rf grid, T = number of trials (timepoints), D = number of feature weights
    _,V = voxel_data.shape ##V = number of voxels

    ##chunk up the voxels
    trnIdx = np.arange(0,T)
    early_stop_num = np.round(len(trnIdx)*early_stop_fraction).astype('int')
    voxel_bin_num = max(2,np.ceil(V/voxel_binsize))  
    voxel_bins = np.linspace(0,V,num=voxel_bin_num,endpoint=True,dtype='int')

    ##chunk up the rf grids
    gdx = np.arange(0,G)
    rf_bin_num = max(2,np.ceil(G/rf_grid_binsize))
    rf_bins = np.linspace(0,G,num=rf_bin_num,endpoint=True,dtype='int')

    ##clock the whole function execution.
    big_start = time()
            
    ##prepare indices for data split
    perm_dx = np.random.permutation(trnIdx)
    validation_idx = perm_dx[0:early_stop_num]
    training_idx = perm_dx[early_stop_num:]
    
    ##for storing the final model for each voxel
    final_rf = np.zeros(V).astype('int')
    final_feature_weights = np.zeros((D,V)).astype('float32')
    final_validation_loss = np.inf*np.ones(V)
    
    ####store error history of best rf model for each voxel
    best_error_history = np.zeros((max_iters,V))
    
    ##iterate over batches of voxels
    for v in range(len(voxel_bins)-1):
        
        ##indices for current batch of voxels
        v_slice = slice(voxel_bins[v], voxel_bins[v+1])
        this_vox_batch_size = voxel_bins[v+1] - voxel_bins[v]
        print '--------------voxels from %d to %d' %(voxel_bins[v],voxel_bins[v+1])
        
        ##get data for these voxels
        this_trn_voxel_data = voxel_data[training_idx, v_slice]
        this_val_voxel_data = voxel_data[validation_idx, v_slice]
        
        
        ##iterate over batches of rf models
        for g in range(len(rf_bins)-1):
            
            ##indices for current batch of rf models
            rf_slice = slice(rf_bins[g], rf_bins[g+1])
            this_rf_batch_size = rf_bins[g+1] - rf_bins[g]
            rf_idx = np.arange(rf_bins[g], rf_bins[g+1])
            print '--------candiate rf models %d to %d' %(rf_bins[g], rf_bins[g+1])
            
            ##slice model space for this batch of models / this batch of voxels
            this_trn_model_space = model_space_tensor[rf_slice,training_idx,:]
            this_val_model_space = model_space_tensor[rf_slice,validation_idx,:]

            ##initialize best and current loss containers for this batch of voxels/models
            best_validation_loss = np.inf*np.ones((this_rf_batch_size,this_vox_batch_size)) #rf_chunk x voxel_chunk
            this_validation_loss = np.zeros(best_validation_loss.shape)
            
            ##initialize best and current weight containers for this batch of voxels/models
            if initial_feature_weights == 'zeros':
	      best_feature_weights = np.zeros((this_rf_batch_size, D, this_vox_batch_size),dtype='float32')
	    else:
	      best_feature_weights = initial_feature_weights[rf_slice,:,:]
	      #best_feature_weights = best_feature_weights[:,:,v_idx.flatten()]
              best_feature_weights = best_feature_weights[:,:,v_slice]
            feature_weights = np.copy(best_feature_weights)
            
            
            ##initialize reports. so you can waste an entire afternoon watching your models train.
            iter_error = np.zeros((max_iters, this_vox_batch_size))
            bestie_change = np.zeros(max_iters)
            old_besties = np.zeros(this_vox_batch_size)
            
            ##initialize counters
            iters = 0
            start = time()
                        
            ##take gradient steps for a fixed number of iterations
            while (iters < max_iters):
                
                ##gradient: put a loop here over chunks of rf models to save on memory
                d_loss_wrt_params = compute_grad(this_trn_voxel_data,
                                                 this_trn_model_space,
                                                 feature_weights)
                
                
                ##update feature weights
                feature_weights -= learning_rate * d_loss_wrt_params
                
                ##predictions with updated feature weights
                pm = prediction_menu(this_val_model_space, feature_weights)

                ##updated loss
                this_validation_loss = sq_diff_func(this_val_voxel_data, pm)
                
                ##if new loss minimum, save as best
                improved = this_validation_loss < best_validation_loss  ##rf batch x voxel batch
                imp = np.sum(improved)
                for ii in range(this_rf_batch_size):
                    best_validation_loss[ii,improved[ii,:]] = np.copy(this_validation_loss[ii,improved[ii,:]])
                    best_feature_weights[ii,:,improved[ii,:]] = np.copy(feature_weights[ii,:,improved[ii,:]])
                
                ##reporting business
                iter_error[iters,:]  = np.min(this_validation_loss,axis=0)
                besties = np.argmin(this_validation_loss,axis=0)
                bestie_change[iters] = np.sum(besties - old_besties)
                old_besties = np.copy(besties)
                if iters % report_every == 0:
                    print '-------'
                    print 'errors: %f' %(np.nanmean(iter_error[iters,:]))
                    print 'change in best rf: %f' %(bestie_change[iters])
                    print 'norm of feature weights: %f' %(np.sqrt(np.sum(feature_weights*feature_weights)))
                    print 'improvements: %d' %(imp)
                    print time()-start
                    start = time()
                
                ##update iteration
                iters += 1
            
            ##if the best of this batch of models has achieved new loss minimum, save it.
            for ii,this_voxel in enumerate(np.arange(voxel_bins[v], voxel_bins[v+1])):
                best_of_batch_rf = np.argmin(best_validation_loss[:,ii]) 				         ##best rf for current rf_batch / current voxel
                if best_validation_loss[best_of_batch_rf,ii] < final_validation_loss[this_voxel]: 	         ##if best of this batch better than all previous batches
                    final_validation_loss[this_voxel] = np.copy(best_validation_loss[best_of_batch_rf,ii])	 ##reset loss for current voxel
                    final_feature_weights[:,this_voxel] = np.copy(best_feature_weights[best_of_batch_rf,:,ii])   ##reset corresponding feature weights for current voxel
                    final_rf[this_voxel] = rf_idx[best_of_batch_rf]						 ##reset corresponding rf model for current voxel
                    best_error_history[:,this_voxel] = np.copy(iter_error[:,ii])                    		 ##save the whole error history.

    print time()-big_start
    return final_validation_loss,final_feature_weights,final_rf,best_error_history
##----



def leave_k_out_training(val_idx, *args, **kwargs):
  '''
leave_k_out_training(val_idx,
		     model_space_tensor, voxel_data,
		     initial_feature_weights = 'zeros',
                     early_stop_fraction = 0.2,
                     max_iters=100,
                     mini_batch_size = 0.1,
                     learning_rate=10**(-5),
                     voxel_binsize=100,
                     rf_grid_binsize = 200,
                     report_every = 10)
    
    runs "train_fwrf_model" multiple times over multiple validation sets.
    
    for each validation set, trains a separate model using the gradient-descent-with-early-stopping procedure specified by "train_fwrf_model".
    
    
    inputs
	 val_idx            ~ a dictionary of validation sets. each key is an integer from 0 to number of val. sets. each value is an np.array of validation indices idx: 0 <= idx < T,
			      where T is the total number of available data samples
    
	 model_space_tensor ~ G x T x D tensor, G = rf models, D = features, T = time/trials
		 voxel_data ~ T x V array of neural data, V ~ voxels
    initial_feature_weights ~ default = 'zeros', in which case set all to 0. otherwise, a G x D x V array of initial guesses at feature weights (if you have the memory to spare)
	early_stop_fraction ~ amount of your training data reserved for early stopping. this subset called the "validation data".
			      error reports refer to error on validation data. gradients are estimated on the remaining fraction, i.e., the "training data"
		  max_iters ~ the only way to get this thing to stop. just go for as long as you have to wait.
	      learning_rate ~ size of gradient step. mess with it until you get it right.
	      voxel_binsize ~ how many voxels to train at one time
	    rf_grid_binsize ~ how many rf models to evaluate at one time
	       report_every ~ print a summary of error every this-many iterations
   
   outputs
   
   trn_idx               ~ dicionaries with integer keys indicating resampling iteration. Each value is an array of validation/training trials used for validation and sampling.
   params                ~ a dictionary with same integer keys as above. each dictionary 
   final_validation_loss ~ V. for each voxel and each rf, the minimum validation loss.
   final_feature_weights ~ D x V. for each voxel, the feature weights associated with the best rf model for that voxel
		final_rf ~ V. index of best rf model for each voxel
      best_error_history ~ max_iters x V. the error trajectory for the best rf model for each voxel. the min of this trajectory = final_validation_loss
  
  
  
  '''

  trn_idx = {}
  params = {}
  voxel_data = args[1]
  mst = args[0]
  T = voxel_data.shape[0] ##number of data samples
  print 'number of data samples: %d' %(T)
  
  n_resamples = len(val_idx.keys())
  for r_iter in range(n_resamples):
      print '======beginning training round %d' %(r_iter)
      params[r_iter] = {}
      trn_idx[r_iter] = np.setdiff1d(np.arange(T), val_idx[r_iter]).astype('int').ravel()
      trn_voxel_data = voxel_data[trn_idx[r_iter],:]
      training_mst = mst[:,trn_idx[r_iter],:]
      params[r_iter]['fvl'],params[r_iter]['ffw'],params[r_iter]['frf'],params[r_iter]['beh'] = train_fwrf_model(training_mst, trn_voxel_data, **kwargs)

  return trn_idx, params
      


def split_em_up(n_stim, val_frac, n_resamples):
  '''
  val_idx = split_em_up(n_stim, val_frac, n_resamples)
  
  splits up data samples into muliple, nonoverlapping validation sets
  n_stim ~ number of training samples available
  val_frac ~ fraction of training samples in each validation set.
  n_resamples ~ number of validation sets
  
  *Note*: prioritizes n_resamples over val_frac to make sure validation sets never overlap.
  For example, suppose n_stim = 500, and you ask for val_frac = .9 and n_resamples = n_stim. The number of samples
  in each validation set will be exactly one, even though you asked for ~460 samples. This is because we want
  to strongly enforce non-overlapment of validation sets.
  
  returns:
    val_idx: a dictionary with integer keys. each value is an np.array of indices between in the range 0 <= index < n_stim.
  
  '''
  Tval = int(np.floor(val_frac*n_stim))
  val_idx = {}
  cur = 0 
  cnt = 0
  val_idx[0] = []
  while (cur <= (n_stim - n_resamples)) and (cnt <= Tval):
    val_idx[0].append(cur)
    cur += n_resamples
    cnt += 1
  
  val_idx[0] = np.array(val_idx[0],dtype='int').ravel()
  print 'number of validation samples: %d' %(len(val_idx[0]))

  for nsamp in np.arange(1,n_resamples).astype('int'):
    val_idx[nsamp] = val_idx[0]+nsamp
    
  return val_idx





