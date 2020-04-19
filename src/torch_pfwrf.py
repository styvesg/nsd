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
from src.numpy_utility import iterate_range


def _to_torch(x, device=None):
    return torch.from_numpy(x).float().to(device)

def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
def set_value(_x, x):
    if list(x.shape)!=list(_x.size()):
        _x.resize_(x.shape)
    _x.data.copy_(torch.from_numpy(x))
    
def flat_init(shape):
    return np.full(shape=shape, fill_value=1./np.prod(shape[1:]), dtype=np.float32)
def zeros_init(shape):
    return np.zeros(shape=shape, dtype=np.float32)
def ones_init(shape):
    return np.ones(shape=shape, dtype=np.float32)
class normal_init(object):
    def __init__(self, scale=1.):
        self.scale = scale   
    def __call__(self, shape):
        return np.random.normal(0, self.scale, size=shape).astype(np.float32)
    
def downsampling(fm_s, rf_s):
    if fm_s==rf_s:
        return 0, None
    elif fm_s[0]<rf_s[0]:
        return -1, pnu.create_downsampling_array(old_dim=rf_s, new_dim=fm_s).T
    else:
        return 1,  pnu.create_downsampling_array(old_dim=fm_s, new_dim=rf_s).T   

    
class Torch_fwRF_voxel_block(nn.Module):
    def __init__(self, _fmaps_fn, _nonlinearity=None, \
                 input_shape=(1,3,227,227), max_fpf_resolution=21, voxel_batch_size=1000, shared_fpf=False):
        super(Torch_fwRF_voxel_block, self).__init__()
        ###
        nv = voxel_batch_size
        device = next(_fmaps_fn.parameters()).device
        _x =torch.empty((1,)+input_shape[1:], device=device).uniform_(0, 1)
        _fmaps = _fmaps_fn(_x)
        ###
        self.fmaps_shapes, self.fmaps_rez = [], []
        for k,_fm in enumerate(_fmaps):
            assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
            self.fmaps_rez += [_fm.size()[2],]
            self.fmaps_shapes += [_fm.size(),] 
        nf = np.sum([s[1] for s in self.fmaps_shapes])
        ###
        self.rfs, self.fpf_rez = [], [] 
        if shared_fpf:
            nr = min(max_fpf_resolution, np.max(self.fmaps_rez))
            self.fpf_rez += [nr,]
            self.rfs += [nn.Parameter(torch.tensor(np.ones(shape=(nv, nr, nr), dtype=np.float32), requires_grad=True).to(device)),]
            self.register_parameter('rf', self.rfs[-1])
        else:
            for k,fm_rez in enumerate(self.fmaps_rez):
                assert _fm.size()[2]==_fm.size()[3], 'All feature maps need to be square'
                nr = min(max_fpf_resolution, fm_rez)
                self.fpf_rez += [nr,]
                if nr==1:
                    self.rfs    += [nn.Parameter(torch.tensor(np.ones(shape=(nv, nr, nr), dtype=np.float32), requires_grad=False).to(device)),]
                    self.register_parameter('rf%d'%k, self.rfs[-1])
                else:
                    self.rfs    += [nn.Parameter(torch.tensor(np.ones(shape=(nv, nr, nr), dtype=np.float32), requires_grad=True).to(device)),]
                    self.register_parameter('rf%d'%k, self.rfs[-1])
        ###
        self.dl = []
        self.ul = []
        for k,fm_rez in enumerate(self.fmaps_rez):
            if shared_fpf:
                nr = self.fpf_rez[0]
            else:
                nr = self.fpf_rez[k]
            d, u = downsampling([fm_rez, fm_rez], [nr, nr])
            self.dl += [d,]
            if u is not None:
                self.ul += [torch.tensor(u.astype(np.float32), requires_grad=False).to(device),]
                if d<0: # downsample rf
                    print ('downsample fpf from', nr, 'to', fm_rez)
                elif d>0: # downsample fm
                    print ('downsample fmaps from', fm_rez, 'to', nr)
            else:
                self.ul += [None,]
                print ('native fpf resolution', fm_rez)                  
        ###
        self.nl = _nonlinearity
        self.sm = nn.Softmax(dim=1)
        self.w  = nn.Parameter(torch.tensor(np.random.normal(0,0.001, size=(nv, nf)).astype(dtype=np.float32), requires_grad=True).to(device))
        self.b  = nn.Parameter(torch.tensor(np.full(fill_value=0.0, shape=(nv,), dtype=np.float32), requires_grad=True).to(device))
      
    def get_param_inits(self):
        return [ones_init,]*len(self.rfs) + [normal_init(0.001), zeros_init]
    
    def reset_params(self):
        for _p,init in zip(self.parameters(), self.get_param_inits()):
            set_value(_p, init(_p.size()))
                      
    def load_voxel_block(self, *params):
        for _p,p in zip(self.parameters(), params):
            if len(p)<_p.size()[0]:
                pp = np.zeros(shape=_p.size(), dtype=p.dtype)
                pp[:len(p)] = p
                set_value(_p, pp)
            else:
                set_value(_p, p)
             
    def forward(self, _fmaps):
        phi = []
        if len(self.rfs)==1:
            for fm,d,u in zip(_fmaps, self.dl, self.ul):
                g = self.sm(torch.flatten(self.rfs[0], start_dim=1))
                f = torch.flatten(fm, start_dim=2)  
                if self.nl is not None:          
                    f = self.nl(f)
                if d<0: # downsample rf
                    g = torch.mm(g, u) # g : [nv, space]
                elif d>0: # downsample fm     
                    f = torch.tensordot(f, u, dims=[[2],[0]])  # f : [batch, features, space]
                phi += [torch.tensordot(g, f, dims=[[1],[2]]),] # apply pooling field and add to list. phi : [nv, batch, features]              
        else:
            for fm,rf,d,u in zip(_fmaps, self.rfs, self.dl, self.ul):
                g = self.sm(torch.flatten(rf, start_dim=1))
                f = torch.flatten(fm, start_dim=2)  
                if self.nl is not None:          
                    f = self.nl(f)            
                if d<0: # downsample rf
                    g = torch.mm(g, u)
                elif d>0: # downsample fm               
                    f = torch.tensordot(f, u, dims=[[2],[0]])
                phi += [torch.tensordot(g, f, dims=[[1],[2]]),] 
        ###
        Phi = torch.cat(phi, dim=2)
        _r = torch.squeeze(torch.bmm(Phi, torch.unsqueeze(self.w,2))).t() + torch.unsqueeze(self.b,0) # _r : [batch, nv]
        return _r
    
    
def learn_params_gradient(data, voxels, _fmaps_fn, _fwrf_fn, \
                                  sample_batch_size=100, holdout_size=100, num_epochs=100, \
                                  lr=1., l1=0., l2=0., shuffle=False):
    '''The fwrf_fn module needs to have some extra methods:
    1) get_param_inits()
    2) reset_params()
    '''
    ###
    def loss_fn(x, v):
        r = _fwrf_fn(_fmaps_fn(x))
        err = torch.sum((r - v)**2, dim=0)
        loss = torch.sum(err) 
        loss += float(l1) * torch.sum(torch.abs(_fwrf_fn.w))
        return err, loss

    dtype = voxels.dtype.type
    nt, nv = voxels.shape
    device = next(_fmaps_fn.parameters()).device
    trn_size = nt - holdout_size
    assert trn_size>0, 'Training size needs to be greater than zero'
    print ('trn_size = %d (%.1f%%)' % (trn_size, float(trn_size)*100/nt))
    print ('dtype = %s' % dtype)
    print ('device = %s' % device)
    print ('---------------------------------------')
    
    order = np.arange(nt, dtype=int)
    if shuffle:
        np.random.shuffle(order)
        data = data[order] 
        voxels = voxels[order]
    
    _params = [_p for _p in _fwrf_fn.parameters()]
    voxel_batch_size = _params[0].size()[0]
    import torch.optim as optim
    optimizer = optim.Adam(_params, lr=float(lr), betas=(0.9, 0.999), eps=1e-08)
    
    best_epochs = np.full(fill_value=-1, shape=(nv), dtype=int)
    best_losses = np.full(fill_value=np.inf, shape=(nv), dtype=dtype)
    param_inits = _fwrf_fn.get_param_inits()
    best_param_values = [init(shape=(nv,)+_p.size()[1:]) for _p,init in zip(_params, param_inits)]
    
    for rv, lv in iterate_range(0, nv, voxel_batch_size):
        voxel_batch = voxels[:,rv]
        if lv<voxel_batch_size:
            voxel_batch = np.concatenate([voxel_batch, np.zeros(shape=(nt, voxel_batch_size-lv), dtype=dtype)], axis=1)
        # reset param block
        _fwrf_fn.reset_params()
        for epoch in range(num_epochs):
            ### training loop
            for rt, lt in iterate_range(0, trn_size, sample_batch_size):
                optimizer.zero_grad()
                _,loss = loss_fn(_to_torch(data[rt], device), _to_torch(voxel_batch[rt], device))
                loss.backward()
                optimizer.step()     
            ### holdout validation loop
            val_err = np.zeros(shape=(voxel_batch_size), dtype=dtype)
            for rt, lt in iterate_range(trn_size, holdout_size, sample_batch_size):
                err,_ = loss_fn(_to_torch(data[rt], device), _to_torch(voxel_batch[rt], device))
                val_err += get_value(err)
            ### save params
            cur_losses = val_err / holdout_size
            cur_params = [get_value(_p) for _p in _params]

            improvement = (cur_losses[:lv]<=best_losses[rv])
            if np.sum(improvement)>0:
                rvimp = np.array(rv)[improvement]
                for bpar,cpar in zip(best_param_values, cur_params):
                    bpar[rvimp] = np.copy(cpar[:lv][improvement])
                best_losses[rvimp] = cur_losses[:lv][improvement] 
                best_epochs[rvimp] = epoch+1
            ## Then we print the results for this epoch:
            sys.stdout.write('\rvoxels [%6d:%-6d] of %d, epoch %3d of %3d, loss = %.6f (%d)' % (rv[0], rv[-1], nv, epoch+1, num_epochs, np.mean(cur_losses), np.sum(improvement)))     
    sys.stdout.flush()
    return best_losses, best_epochs, best_param_values


def get_predictions(data, _fmaps_fn, _fwrf_fn, params, sample_batch_size=100):
    dtype = data.dtype.type
    device = next(_fmaps_fn.parameters()).device
    _params = [_p for _p in _fwrf_fn.parameters()]
    voxel_batch_size = _params[0].size()[0]    
    nt, nv = len(data), len(params[0])
    print ('val_size = %d' % nt)
    pred = np.full(fill_value=0, shape=(nt, nv), dtype=dtype)
    start_time = time.time()
    with torch.no_grad():
        for rv, lv in iterate_range(0, nv, voxel_batch_size):
            _fwrf_fn.load_voxel_block(*[p[rv] for p in params])
            pred_block = np.full(fill_value=0, shape=(nt, voxel_batch_size), dtype=dtype)
            for rt, lt in iterate_range(0, nt, sample_batch_size):
                sys.stdout.write('\rsamples [%5d:%-5d] of %d, voxels [%6d:%-6d] of %d' % (rt[0], rt[-1], nt, rv[0], rv[-1], nv))
                pred_block[rt] = get_value(_fwrf_fn(_fmaps_fn(_to_torch(data[rt], device)))) 
            pred[:,rv] = pred_block[:,:lv]
    total_time = time.time() - start_time
    print ('---------------------------------------')
    print ('total time = %fs' % total_time)
    print ('sample throughput = %fs/sample' % (total_time / nt))
    print ('voxel throughput = %fs/voxel' % (total_time / nv))
    sys.stdout.flush()
    return pred
