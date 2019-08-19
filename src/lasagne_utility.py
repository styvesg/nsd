import sys
import struct
from time import time
import numpy as np
import scipy.io as sio
from scipy import ndimage as nd
from scipy import misc
import pickle
import math

import theano
import theano.tensor as T

import lasagne
import lasagne.layers as L
import lasagne.regularization as R
import lasagne.nonlinearities as NL
import lasagne.objectives as O
import lasagne.init as I

# lasagne utility
def get_layer(net, name):
    for i,l in enumerate(L.get_all_layers(net)):
        if l.name == name:
            return i,l
        
def print_lasagne_network(_net, skipnoparam=True):
    layers = L.get_all_layers(_net)
    for l in layers:
        out = l.output_shape
        par = l.get_params()
        if skipnoparam and len(par)==0 and l.name==None:
            continue
        print "Layer\t: %s\nName\t: %s\nType\t: %s" % (l, l.name, type(l))
        print "Shape\t: %s" % (out,)
        if len(par)>0:
            print "Params"
            for p in par:
                print "        |-- {:<10}: {:}".format(p.name, p.get_value().shape,)
        print "\n"
        
def validate_parameters(_net, param_list):
    param_name = L.get_all_params(_net)
    value_list = L.get_all_param_values(_net)
    assert len(value_list)==len(param_list)
    err = 0
    for i in range(len(value_list)):
        if value_list[i].shape==param_list[i].shape:
            print GREEN+ "Param %d: %s = %s" % (i, param_name[i], value_list[i].shape) + END
        else:
            print RED  + "Param %d: %s = %s != %s" % (i, param_name[i], value_list[i].shape, param_list[i].shape) + END
            err += 1
    if err==0:
        print GREEN+"Valid parameters"+END
        return 0
    else:
        print RED+"%d parameter mismatch" % err + END
        return err
        
        
# layer shortcuts        
def winit(_W, n, sigma=0.02):
    return (I.Normal(sigma) if _W==None else _W[n])

def conv(_in, *args, **kwargs):
    return L.Conv2DLayer(_in, *args, untie_biases=False, flip_filters=True, convolution=theano.tensor.nnet.conv2d, **kwargs)

def deconv(_in, *args, **kwargs):
    return L.Deconv2DLayer(_in, *args, untie_biases=False, flip_filters=False, **kwargs)  
  
def fc_concat(_in, _vec):
    return L.ConcatLayer([_in, _vec], axis=1)

def conv_concat(_in, _vec):
    n = _in.output_shape[2]
    _bcast = L.ExpressionLayer(_vec, lambda __X: __X.dimshuffle(0, 1, 'x', 'x') * T.ones((__X.shape[0], __X.shape[1], n, n)), output_shape='auto')      
    return L.ConcatLayer([_in, _bcast], axis=1)

def batch_norm(_in, *args, **kwargs):
    return L.batch_norm(_in, beta=I.Constant(0.), gamma=I.Constant(1.), *args, **kwargs)

def batch_norm_n(_in, *args, **kwargs):
    return L.batch_norm(_in, beta=None, gamma=None, *args, **kwargs)

def avg(_in, *args, **kwargs):
    return L.Pool2DLayer(_in, *args, ignore_border=True, mode='average_exc_pad', **kwargs)

def flatten(_in, **kwargs):
    return L.FlattenLayer(_in, **kwargs)

def sigmoid(_in, **kwargs):
    return L.NonlinearityLayer(_in, nonlinearity=NL.sigmoid)

def tanh(_in, **kwargs):
    return L.NonlinearityLayer(_in, nonlinearity=NL.tanh)

# extra loss
def pullaway_loss(_emb):
    ## input of this is a flattened layer after the encoding step of the autoencoder
    _norm = T.sqrt(T.sqr(_emb).sum(axis=1, keepdims=True))
    _nemb = _emb / _norm
    _sim = T.sqr(T.dot(_nemb, _nemb.T))
    _batch_size = _sim.shape[0].astype(theano.config.floatX)
    _pt_loss = (T.sum(_sim) - _batch_size) / (_batch_size * (_batch_size - 1.))
    return _pt_loss

# noise and condition concatenations. 
def create_slices_from(_source, ish, start=0, num_slices=1):
    ns = num_slices if len(ish)==2 else num_slices * ish[2] * ish[3]
    osh = (num_slices,) if len(ish)==2 else (num_slices, ish[2], ish[3])
    
    _slice = L.SliceLayer(_source, indices=slice(start, start+ns), axis=1)
    return L.ReshapeLayer(_slice, ([0],)+osh), ns

def create_conditon_slices_from(_cond, ish):
    if len(ish)==2:
        return _cond
    else:
        return L.ExpressionLayer(_cond, lambda __X: __X.dimshuffle(0, 1, 'x', 'x') \
                * T.ones((__X.shape[0], __X.shape[1],)+ish[-2:]), output_shape='auto')
    
def concat_tc(_top, _cond):
    _cond1 = create_conditon_slices_from(_cond, _top.output_shape)
    return L.ConcatLayer([_top, _cond1], axis=1)

def concat_tn(_top, _seed, start=0, num_slices=1):
    if _top==None:
        return L.SliceLayer(_seed, indices=slice(start, start+num_slices), axis=1), start+num_slices
    elif num_slices>0:
        _seed1, n = create_slices_from(_seed, _top.output_shape, start=start, num_slices=num_slices)
        return L.ConcatLayer([_top, _seed1], axis=1), start+n  
    else:
        return _top, start

def concat_tcn(_top, _cond, _seed, start=0, num_slices=1):
    _cond1 = create_conditon_slices_from(_cond, _top.output_shape)
    if num_slices>0:
        _seed1, n = create_slices_from(_seed, _top.output_shape, start=start, num_slices=num_slices)
        return L.ConcatLayer([_top, _cond1, _seed1], axis=1), start+n
    else:
        return L.ConcatLayer([_top, _cond1], axis=1), start
