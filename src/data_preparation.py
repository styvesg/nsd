import sys
import numpy as np
import h5py
import pickle
import theano
import theano.tensor as T


def dnn_feature_extractor(_incoming, param_file_name='', drop_ratio_conv=0., drop_ratio_fc=0.5):
    '''Need to provide a size-compatible input at runtime i.e.: ('x', 3, 227, 227)'''
    import lasagne
    import lasagne.layers as L
    import lasagne.nonlinearities as NL
    import lasagne.init as I
    from src.lasagne_utility import deconv, conv, batch_norm, batch_norm_n, fc_concat, \
    conv_concat, avg, flatten, sigmoid, tanh
    from src.lasagne_utility import print_lasagne_network
  
    _Xbar = theano.shared(np.array( [122.67891434, 116.66876762, 104.00698793] ).astype(np.float32))
    _Xbar = T.patternbroadcast(_Xbar.dimshuffle(('x', 0, 'x', 'x')), (True, False, True, True))

    l_in = L.InputLayer(shape=(None, 3, 227, 227), input_var=(_incoming.astype(theano.config.floatX) - _Xbar) / 255)
    # block 1
    conv1 = conv(l_in, num_filters=96, filter_size=7, stride=2, pad=0, W=I.Normal(.02), b=I.Constant(0), nonlinearity=NL.rectify) #111
    pool1 = L.MaxPool2DLayer(conv1, pool_size=3, stride=2)  # 55  
    lrn1  = L.LocalResponseNormalization2DLayer(pool1, alpha=0.0001/5, k=2, beta=0.75, n=5)
    # block 2
    conv2 = conv(lrn1, num_filters=256, filter_size=5, stride=2, pad=1, W=I.Normal(.02), b=I.Constant(1), nonlinearity=NL.rectify) #27
    pool2 = L.MaxPool2DLayer(conv2, pool_size=3, stride=2)    
    lrn2  = L.LocalResponseNormalization2DLayer(pool2, alpha=0.0001/5, k=2, beta=0.75, n=5)
    # block 3
    conv3 = batch_norm_n(conv(lrn2, num_filters=384, filter_size=3, stride=1, pad=1, W=I.Normal(.02), b=I.Constant(0), nonlinearity=NL.rectify)) #13
    # block 4    
    drop4 = L.DropoutLayer(conv3, p=drop_ratio_conv, rescale=True)
    conv4 = batch_norm_n(conv(drop4, num_filters=384, filter_size=3, stride=1, pad=1, W=I.Normal(.02), b=I.Constant(1), nonlinearity=NL.rectify)) #11
    # block 5      
    drop5 = L.DropoutLayer(conv4, p=drop_ratio_conv, rescale=True)
    conv5 = batch_norm_n(conv(drop5, num_filters=256, filter_size=3, stride=1, pad=1, W=I.Normal(.02), b=I.Constant(0), nonlinearity=NL.rectify)) #9    
    pool5 = L.MaxPool2DLayer(conv5, pool_size=3, stride=2)    # 4
    # block 6
    fc6 = batch_norm_n(L.DenseLayer(pool5, num_units=4096, W=I.Normal(.02), b=I.Constant(1), nonlinearity=NL.rectify))
    drop6 = L.DropoutLayer(fc6, p=drop_ratio_fc, rescale=True)
    # block 7
    fc7 = batch_norm_n(L.DenseLayer(drop6, num_units=4096, W=I.Normal(.02), b=I.Constant(0), nonlinearity=NL.rectify))
    drop7 = L.DropoutLayer(fc7, p=drop_ratio_fc, rescale=True)
    # output layer p(y|h(x))
    fc8 = L.DenseLayer(drop7, num_units=1000, W=I.Normal(.02), b=I.Constant(1), nonlinearity=None)
    ### load the parameters from file
    params = L.get_all_params(fc8, trainable=True)
    param_file = open(param_file_name, 'rb') 
    param_value = pickle.load(param_file)
    L.set_all_param_values(fc8, param_value)
    param_file.close()
    ###
    aux = [L.get_output(pool1, deterministic=True),
            L.get_output(conv2, deterministic=True), 
            L.get_output(conv3, deterministic=True), 
            L.get_output(conv4, deterministic=True), 
            L.get_output(conv5, deterministic=True), 
            L.get_output(L.DimshuffleLayer(fc6, (0,1,'x','x')), deterministic=True), 
            L.get_output(L.DimshuffleLayer(fc7, (0,1,'x','x')), deterministic=True), 
            L.get_output(L.DimshuffleLayer(fc8, (0,1,'x','x')), deterministic=True)]
    print_lasagne_network(fc8, skipnoparam=True)
    return aux, L.get_output(L.NonlinearityLayer(fc8, nonlinearity=NL.softmax), deterministic=True), params


def get_dnn_feature_maps(stim_data, fmaps_fn, batch_size):
    def iterate_range(start, length, batchsize):
        batch_count = int(length // batchsize )
        residual = int(length % batchsize)
        for i in range(batch_count):
            yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
        if(residual>0):
            yield range(start+batch_count*batchsize,start+length),residual  
 
    size = len(stim_data)  
    fmaps = fmaps_fn(stim_data[:batch_size])
    fmaps = [np.zeros(shape=(len(stim_data),)+fm.shape[1:], dtype=np.float32) for fm in fmaps] 
    for rr,rl in iterate_range(0, len(stim_data), batch_size):
        fb = fmaps_fn(stim_data[rr])
        for k in range(len(fb)):
            fmaps[k][rr] = fb[k][:]
    return fmaps





#def create_dnn_feature_maps(stim_data, fmaps_fn, batch_size, fmap_max=1024, trn_size=None):
#    def iterate_range(start, length, batchsize):
#        batch_count = int(length // batchsize )
#        residual = int(length % batchsize)
#        for i in range(batch_count):
#            yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
#        if(residual>0):
#            yield range(start+batch_count*batchsize,start+length),residual  
# 
#    size = trn_size if trn_size is not None else len(stim_data)
#    fmaps = fmaps_fn(stim_data[:batch_size])
#    fmaps = [np.zeros(shape=(len(stim_data),)+fm.shape[1:], dtype=np.float32) for fm in fmaps] 
#    for rr,rl in iterate_range(0, len(stim_data), batch_size):
#        fb = fmaps_fn(stim_data[rr])
#        for k in range(len(fb)):
#            fmaps[k][rr] = fb[k][:]
#    # fmaps is all the feature maps for all the image. We need to reduce it a bit 
#    fmask = [np.zeros(shape=(fm.shape[1]), dtype=bool) for fm in fmaps]
#    for k,fm in enumerate(fmaps):  
#        if fm.shape[1]>fmap_max:
#            #select the feature map with the most variance to the dataset
#            fmap_var = np.var(fm[:size], axis=(0,2,3))
#            most_var = fmap_var.argsort()[-fmap_max:] #the feature indices with the top-fmap_max variance
#            fmaps[k] = fm[:,np.sort(most_var),:,:]
#            fmask[k][most_var] = True
#        else:
#            fmask[k][:] = True
#        print "layer: %s, shape=%s" % (k, (fmaps[k].shape))      
#        sys.stdout.flush() 
#
#    # ORIGINAL PARTITIONING OF LAYERS
#    fmaps_sizes = [fm.shape for fm in fmaps]
#    fmaps_count = sum([fm[1] for fm in fmaps_sizes])   
#    partitions = [0,]
#    for r in fmaps_sizes:
#        partitions += [partitions[-1]+r[1],]
#    layer_rlist = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
#    # concatenate fmaps of identical dimension to speed up rf application
#    clmask, cfmask, cfmaps = [],[],[]
#    print ""
#    # I would need to make sure about the order and contiguousness of the fmaps to preserve the inital order.
#    # It isn't done right now but since the original feature maps are monotonically decreasing in resultion in
#    # the examples I treated, the previous issue doesn't arise.
#    for k,us in enumerate(np.unique([np.prod(fs[2:4]) for fs in fmaps_sizes])[::-1]): ## they appear sorted from small to large, so I reversed the order
#        mask = np.array([np.prod(fs[2:4])==us for fs in fmaps_sizes]) # mask over layers that have that spatial size
#        lmask = np.arange(len(fmaps_sizes))[mask] # list of index for layers that have that size
#        bfmask = np.concatenate([fmask[l] for l in lmask], axis=0)
#        clmask += [lmask,]
#        cfmask += [np.arange(len(bfmask))[bfmask],]
#        cfmaps += [np.concatenate([fmaps[l] for l in lmask], axis=1),]
#        print "fmaps: %s, shape=%s" % (k, (cfmaps[-1].shape)) 
#        sys.stdout.flush()
#    fmaps_sizes = [fm.shape for fm in cfmaps]
#    return cfmaps, layer_rlist, fmaps_sizes, fmaps_count, clmask, cfmask #, scaling


def create_dnn_feature_maps(stim_data, fmaps_fn, batch_size, fmap_max=1024, trn_size=None):
    def iterate_range(start, length, batchsize):
        batch_count = int(length // batchsize )
        residual = int(length % batchsize)
        for i in range(batch_count):
            yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
        if(residual>0):
            yield range(start+batch_count*batchsize,start+length),residual 
    size = trn_size if trn_size is not None else len(stim_data)
    fmaps = fmaps_fn(stim_data[:batch_size])
    run_avg = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps]
    run_sqr = [np.zeros(shape=(fm.shape[1]), dtype=np.float64) for fm in fmaps] 
    for rr,rl in iterate_range(0, size, batch_size):
        fb = fmaps_fn(stim_data[rr])
        for k,f in enumerate(fb):
	    if f.shape[1]>fmap_max: # only need the average if we're going to use them to reduce the number of feature maps
                run_avg[k] += np.sum(np.mean(f.astype(np.float64), axis=(2,3)), axis=0)
                run_sqr[k] += np.sum(np.mean(np.square(f.astype(np.float64)), axis=(2,3)), axis=0)
    for k in range(len(fb)):
        run_avg[k] /= size
        run_sqr[k] /= size
    ###
    fmask = [np.zeros(shape=(fm.shape[1]), dtype=bool) for fm in fmaps]
    fmap_var = [np.zeros(shape=(fm.shape[1]), dtype=np.float32) for fm in fmaps]
    for k,fm in enumerate(fmaps):  
        if fm.shape[1]>fmap_max:
            #select the feature map with the most variance to the dataset
            fmap_var[k] = (run_sqr[k] - np.square(run_avg[k])).astype(np.float32)
            most_var = fmap_var[k].argsort()[-fmap_max:] #the feature indices with the top-fmap_max variance
            fmaps[k] = fm[:,np.sort(most_var),:,:]
            fmask[k][most_var] = True
        else:
            fmask[k][:] = True
        print "layer: %s, shape=%s" % (k, (fmaps[k].shape))      
        sys.stdout.flush() 

    # ORIGINAL PARTITIONING OF LAYERS
    fmaps_sizes = [fm.shape for fm in fmaps]
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])   
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    layer_rlist = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    # concatenate fmaps of identical dimension to speed up rf application
    clmask, cfmask, cfmaps = [],[],[]
    print ""
    # I would need to make sure about the order and contiguousness of the fmaps to preserve the inital order.
    # It isn't done right now but since the original feature maps are monotonically decreasing in resultion in
    # the examples I treated, the previous issue doesn't arise.
    for k,us in enumerate(np.unique([np.prod(fs[2:4]) for fs in fmaps_sizes])[::-1]): ## they appear sorted from small to large, so I reversed the order
        mask = np.array([np.prod(fs[2:4])==us for fs in fmaps_sizes]) # mask over layers that have that spatial size
        lmask = np.arange(len(fmaps_sizes))[mask] # list of index for layers that have that size
        bfmask = np.concatenate([fmask[l] for l in lmask], axis=0)
        clmask += [lmask,]
        cfmask += [np.arange(len(bfmask))[bfmask],]
        cfmaps += [np.concatenate([fmaps[l] for l in lmask], axis=1),]
        print "fmaps: %s, shape=%s" % (k, (cfmaps[-1].shape)) 
        sys.stdout.flush()
    fmaps_sizes = [fm.shape for fm in cfmaps]
    return fmap_var, layer_rlist, fmaps_sizes, fmaps_count, clmask, cfmask #, scaling


def preprocess_gabor_feature_maps(feat_dict, act_func=None, dtype=np.float32):
    '''
    Apply optional nonlinearity to the feature maps itself and concatenate feature maps of the same dimensions.
    Returns the feature maps and a list of theano variables to represent them, and the shape of the fmaps.
    '''
    fmap_rez = []
    for k in feat_dict.keys():
        fmap_rez += [feat_dict[k].shape[2],]
    resolutions = np.unique(fmap_rez)
    # concatenate and sort as list
    fmaps_res_count = len(resolutions)
    fmaps_count = 0
    fmaps = []
    for r in range(fmaps_res_count):
        fmaps  += [[],]
    nonlinearity = act_func
    if nonlinearity is None:
        nonlinearity = lambda x: x
    for k in feat_dict.keys():
        # determine which resolution idx this map belongs to
        ridx = np.argmax(resolutions==feat_dict[k].shape[2])
        if len(fmaps[ridx])==0:
            fmaps[ridx] = nonlinearity(feat_dict[k].astype(dtype))
        else:
            fmaps[ridx] = np.concatenate((fmaps[ridx], nonlinearity(feat_dict[k].astype(dtype))), axis=1)       
        fmaps_count += 1
    fmaps_sizes = [] 
    for fm in fmaps:
        fmaps_sizes += [fm.shape]
    print fmaps_sizes
    print "total fmaps = %d" % fmaps_count 
    return fmaps, fmaps_sizes 


def create_gabor_feature_maps(stim_data, gabor_params, nonlinearity=lambda x: x):
    '''input should be a dictionary of control parameters
    output should be the model space tensors'''
    from gaborizer.src.gabor_feature_dictionaries import gabor_feature_maps
    print gabor_params
    n_ori = gabor_params['n_orientations']
    gfm = gabor_feature_maps(n_ori,\
        gabor_params['deg_per_stimulus'], (gabor_params['lowest_sp_freq'], gabor_params['highest_sp_freq'], gabor_params['num_sp_freq']),\
        pix_per_cycle=gabor_params['pix_per_cycle'], complex_cell=gabor_params['complex_cell'],\
        diams_per_filter = gabor_params['diams_per_filter'],\
        cycles_per_radius = gabor_params['cycles_per_radius'])
    #
    fmaps, fmaps_sizes = preprocess_gabor_feature_maps(gfm.create_feature_maps(stim_data), act_func=nonlinearity, dtype=np.float32)
    fmaps_res_count = len(fmaps_sizes)
    fmaps_count = sum([fm[1] for fm in fmaps_sizes])
    #
    ori  = np.array(gfm.gbr_table['orientation'])[0:fmaps_count:fmaps_res_count]
    freq = np.array(gfm.gbr_table['cycles per deg.'])[:fmaps_res_count]
    env  = np.array(gfm.gbr_table['radius of Gauss. envelope (deg)'])[:fmaps_res_count]
    # preprocess_gabor_feature_maps sorts the frequencies and angle such that all feature with the same freq are contiguous.
    partitions = [0,]
    for r in fmaps_sizes:
        partitions += [partitions[-1]+r[1],]
    freq_rlist  = [range(start,stop) for start,stop in zip(partitions[:-1], partitions[1:])] # the frequency ranges list
    ori_rlist = [range(0+i,partitions[-1],n_ori) for i in range(0,n_ori)] # the angle ranges list
    return fmaps, freq, env, ori, freq_rlist, ori_rlist, fmaps_sizes, fmaps_count


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
