import sys
import os
import struct
import time
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import math

import src.numpy_utility as pnu

import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torch.optim as optim


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
        
class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = np.float32(width)/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = np.float32(width)/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)
    
def model_space(model_specs):
    vm = np.asarray(model_specs[0])
    nt = np.prod([sms.length for sms in model_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(model_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')    
    return np.concatenate([xs.reshape((nt,1)).astype(dtype=np.float32), 
                           ys.reshape((nt,1)).astype(dtype=np.float32), 
                           ss.reshape((nt,1)).astype(dtype=np.float32)], axis=1)      



class Torch_fwRF_voxel_block(nn.Module):
    def __init__(self, _fmaps, models, params, mst_avg, mst_std, _nonlinearity=None, input_space=227, aperture=1.0):
        super(Torch_fwRF_voxel_block, self).__init__()
        
        self.aperture = aperture
        device = next(_fmaps.parameters()).device
        _x =torch.empty(1, 3, input_space, input_space, device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        self.fmaps_rez = []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
        
        self.pfs = []
        for k,n_pix in enumerate(self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            self.pfs += [nn.Parameter(torch.from_numpy(pf).to(device), requires_grad=False),]
            self.register_parameter('pf%d'%k, self.pfs[-1])
            
        self.params = [] 
        for k,p in enumerate(params):
            self.params += [nn.Parameter(torch.from_numpy(p).to(device), requires_grad=False),]
            self.register_parameter('params%d'%k, self.params[-1])
            
        self.mstm = None
        self.msts = None
        if mst_avg is not None:
            self.mstm = nn.Parameter(torch.from_numpy(mst_avg.T).to(device), requires_grad=False)
        if mst_std is not None:
            self.msts = nn.Parameter(torch.from_numpy(mst_std.T).to(device), requires_grad=False)
        self._nl = _nonlinearity
        
    def load_voxel_block(self, models, params, mst_avg, mst_std):
        for _pf,n_pix in zip(self.pfs, self.fmaps_rez):
            pf = pnu.make_gaussian_mass_stack(models[:,0], models[:,1], models[:,2], n_pix, size=aperture, dtype=np.float32)[2]
            set_value(_pf, pf)
        for _p,p in zip(self.params, params):
            set_value(_p, p)
        if mst_avg is not None:
            set_value(self.mstm, mst_avg.T)
        if mst_avg is not None:    
            set_value(self.msts, mst_std.T)
        
    def forward(self, _fmaps):
        _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [1,2]]) for _fm,_pf in zip(_fmaps, self.pfs)], dim=1) # [#samples, #features, #voxels] 
        if self._nl is not None:
            _mst = self._nl(_mst)
        if self.mstm is not None:              
            _mst -= self.mstm[None]
        if self.msts is not None:
            _mst /= self.msts[None]
        _mst = torch.transpose(torch.transpose(_mst, 0, 2), 1, 2) # [#voxels, #samples, features]
        _r = torch.squeeze(torch.bmm(_mst, torch.unsqueeze(self.params[0], 2))).t() # [#samples, #voxels]
        if len(self.params)>1:
            _r += torch.unsqueeze(self.params[1], 0)
        return _r



def learn_params_ridge_regression(data, voxels, _fmaps_fn, models, lambdas, aperture=1.0, _nonlinearity=None, zscore=False, sample_batch_size=100, voxel_batch_size=100, holdout_size=100, shuffle=True, add_bias=False):

    def _cofactor_fn(_x, lambdas):
        '''input matrix [#samples, #features], a list of lambda values'''
        _f = torch.stack([(torch.mm(torch.t(_x), _x) + torch.eye(_x.size()[1], device=device) * l).inverse() for l in lambdas], axis=0) # [#lambdas, #feature, #feature]       
        return torch.tensordot(_f, _x, dims=[[2],[1]]) # [#lambdas, #feature, #sample]
    
    def _loss_fn(_cofactor, _vtrn, _xout, _vout):
        '''input '''
        _beta = torch.tensordot(_cofactor, _vtrn, dims=[[2], [0]]) # [#lambdas, #feature, #voxel]
        _pred = torch.tensordot(_xout, _beta, dims=[[1],[1]]) # [#samples, #lambdas, #voxels]
        _loss = torch.sum(torch.pow(_vout[:,None,:] - _pred, 2), dim=0) # [#lambdas, #voxels]
        return _beta, _loss
    
    #############################################################################
    dtype = voxels.dtype.type
    device = next(_fmaps_fn.parameters()).device
    trn_size = len(voxels) - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/len(voxels)))
    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    print ('---------------------------------------')
    # shuffle
    nt = len(data)
    nm = len(models)
    nv = voxels.shape[1]
    order = np.arange(len(voxels), dtype=int)
    if shuffle:
        np.random.shuffle(order)
    data   = data[order]
    voxels = voxels[order]  
    trn_voxels = voxels[:trn_size]
    out_voxels = voxels[trn_size:]
    ### Calculate total feature count
    nf = 0
    _fmaps = _fmaps_fn(_to_torch(data[:sample_batch_size], device=device))

    fmaps_rez = []
    for k,_fm in enumerate(_fmaps):
        nf += _fm.size()[1]
        assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
        fmaps_rez += [_fm[k].size()[2],]
        print (_fm.size())    
    print ('---------------------------------------')
    #############################################################################        
    ### Create full model value buffers    
    best_models = np.full(shape=(nv,), fill_value=-1, dtype=np.int)   
    best_lambdas = np.full(shape=(nv,), fill_value=-1, dtype=np.int)
    best_losses = np.full(fill_value=np.inf, shape=(nv), dtype=dtype)
    best_w_params = np.zeros(shape=(nv, nf), dtype=dtype)
    if add_bias:
        best_w_params = np.concatenate([best_w_params, np.ones(shape=(len(best_w_params),1), dtype=dtype)], axis=1)
    mst_mean = None
    mst_std = None
    if zscore:
        mst_mean = np.zeros(shape=(nv, nf), dtype=dtype)
        mst_std  = np.zeros(shape=(nv, nf), dtype=dtype)
    
    start_time = time.time()
    vox_loop_time = 0
    print ('')
    with torch.no_grad():
        for m,(x,y,sigma) in enumerate(models):
            mst = np.zeros(shape=(nt, nf), dtype=dtype)
            _pfs = [_to_torch(pnu.make_gaussian_mass(x, y, sigma, n_pix, size=aperture, dtype=dtype)[2], device=device) for n_pix in fmaps_rez]
            for rt,rl in iterate_range(0, nt, sample_batch_size):
                _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [0,1]]) for _fm,_pf in zip(_fmaps_fn(_to_torch(data[rt], device=device)), _pfs)], dim=1) # [#samples, #features]
                if _nonlinearity is not None:
                    _mst = _nonlinearity(_mst)
                mst[rt] = get_value(_mst)

            if zscore:  
                mstm = np.mean(mst, axis=0, keepdims=True) #[:trn_size]
                msts = np.std(mst, axis=0, keepdims=True) + 1e-6          
                mst -= mstm
                mst /= msts    

            if add_bias:
                mst = np.concatenate([mst, np.ones(shape=(len(mst), 1), dtype=dtype)], axis=1)

            trn_mst = mst[:trn_size]
            out_mst = mst[trn_size:]
            trn_voxels = voxels[:trn_size]
            out_voxels = voxels[trn_size:]    

            _xtrn = _to_torch(trn_mst, device=device)
            _xout = _to_torch(out_mst, device=device)           
            _cof = _cofactor_fn(_xtrn, lambdas)
            ###    
            vox_start = time.time()
            for rv,lv in iterate_range(0, nv, voxel_batch_size):
                sys.stdout.write('\rmodel %4d of %-4d, voxels [%6d:%-6d] of %d' % (m, nm, rv[0], rv[-1], nv))

                _vtrn = _to_torch(trn_voxels[:,rv], device=device)
                _vout = _to_torch(out_voxels[:,rv], device=device)

                _betas, _loss = _loss_fn(_cof, _vtrn, _xout, _vout) #   [#lambda, #feature, #voxel, ], [#lambda, #voxel]
                _values, _select = torch.min(_loss, dim=0)

                betas = get_value(_betas)
                values, select = get_value(_values), get_value(_select)
                imp = values<best_losses[rv]
                if np.sum(imp)>0:
                    arv = np.array(rv)[imp]
                    li = select[imp]
                    best_lambdas[arv] = li
                    best_losses[arv] = values[imp]
                    best_models[arv] = m
                    if zscore:
                        mst_mean[arv] = mstm # broadcast over updated voxels
                        mst_std[arv]  = msts
                    best_w_params[arv,:] = pnu.select_along_axis(betas[:,:,imp], li, run_axis=2, choice_axis=0).T
            vox_loop_time += (time.time() - vox_start)
        
    #############################################################################   
    total_time = time.time() - start_time
    inv_time = total_time - vox_loop_time
    return_params = [best_w_params[:,:nf],]
    if add_bias:
        return_params += [best_w_params[:,-1],]
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('total throughput = %fs/voxel' % (total_time / nv))
    print ('voxel throughput = %fs/voxel' % (vox_loop_time / nv))
    print ('setup throughput = %fs/model' % (inv_time / nm))
    return best_losses, best_lambdas, models[best_models], return_params, mst_mean, mst_std


    
def get_predictions(data, _fmaps_fn, models, params, mst_avg, mst_std, _nonlinearity=None, aperture=1.0, sample_batch_size=100, voxel_batch_size=100):
    '''the forward fwRF model'''

    dtype = params[0].dtype.type
    device = next(_fmaps_fn.parameters()).device
    if mst_avg is not None:
        assert mst_avg.dtype==dtype, 'dtype of parameters don\'t match'
    if mst_avg is not None:
        assert mst_std.dtype==dtype, 'dtype of parameters don\'t match'

    nv = len(models)
    if torch.is_tensor(data) and nv<=voxel_batch_size:
        allow_grad = True ## all in one batch
        nt = data.size()[0]
        assert device==data.device, "Tensor input need to be on the same device as model"
        _fmaps = _fmaps_fn(data)
    else:   
        nt = len(data)
        _fmaps = _fmaps_fn(_to_torch(data[:sample_batch_size], device=device))

    pred = np.zeros(shape=(nt, nv), dtype=dtype)  
    fmaps_rez = []
    for k,_fm in enumerate(_fmaps):
        assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
        fmaps_rez += [_fm.size()[2],]
   
    allow_grad = nv<=voxel_batch_size ## all in one batch
    start_time = time.time()

    class default_grad(object):
        def __enter__(self): 
            print ('gradients on')
        def __exit__(*x): pass

    with torch.no_grad() if not allow_grad else default_grad():
        for rt,lt in iterate_range(0, nt, sample_batch_size):
            if not allow_grad:
            	_fmaps = _fmaps_fn(_to_torch(data[rt], device=device))

            batch = np.zeros(shape=(lt,nv), dtype=dtype)
            for rv,lv in iterate_range(0, nv, voxel_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], nt, rv[0], rv[-1], nv))
                # create the pf stack for these voxels
                if mst_avg is not None:
                    _mstm = torch.t(_to_torch(mst_avg[rv], device=device)).requires_grad_(False) # [#features, #voxels]
                if mst_std is not None:
                    _msts = torch.t(_to_torch(mst_std[rv], device=device)).requires_grad_(False) # [#features, #voxels]
                _pars = [_to_torch(p[rv], device=device).requires_grad_(False) for p in params] # [#voxels, #features]...

                _pfs = [_to_torch(pnu.make_gaussian_mass_stack(models[rv,0], models[rv,1], models[rv,2], n_pix, size=aperture, dtype=dtype)[2], device=device).requires_grad_(False) for n_pix in fmaps_rez] # [nv, nx, nx]
                _mst = torch.cat([torch.tensordot(_fm, _pf, dims=[[2,3], [1,2]]) for _fm,_pf in zip(_fmaps, _pfs)], dim=1) # [#samples, #features, #voxels] 
                if _nonlinearity is not None:
                    _mst = _nonlinearity(_mst)
                if mst_avg is not None:              
                    _mst -= _mstm[None]
                if mst_std is not None:
                    _mst /= _msts[None]

                _mst = torch.transpose(torch.transpose(_mst, 0, 2), 1, 2)
                _r = torch.squeeze(torch.bmm(_mst, torch.unsqueeze(_pars[0], 2))).t() # [#samples, #voxels]
                if len(_pars)>1:
                    _r += torch.unsqueeze(_pars[1], 0)
                if allow_grad:
                    return _r
                batch[:,rv] = get_value(_r)
            pred[rt,:] = batch
    total_time = time.time() - start_time
    print ('\n---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / nt))
    print ('voxel throughput = %fs/voxel' % (total_time / nv))
    return pred