"""Reduce tiny-operation launch overhead while preserving FP32 HC math."""
import numpy as np
import torch
import torch_npu

FAST_OPS_VERSION=5

_params = {}
_hc_weights = {}
_rotary_tables = {}
_sources = {}


def fused_rms_forward(self, x):
    # FP32 input and weight retain the original FP32 norm arithmetic.
    return torch_npu.npu_rms_norm(x.float(), self.weight, epsilon=self.eps)[0].to(x.dtype)


def hc_pre_cpu(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype, device = x.shape, x.dtype, x.device
    values = x.detach().to(device='cpu', dtype=torch.float32)
    self._hc_cpu_residual = values
    key = hc_fn.data_ptr()
    if key not in _hc_weights:
        _hc_weights[key] = hc_fn.detach().float().cpu().contiguous()
        _sources[('hc',key)]=hc_fn
    flat = values.flatten(2)
    rsqrt = torch.rsqrt(flat.square().mean(-1,keepdim=True)+self.norm_eps)
    mixes = torch.nn.functional.linear(flat,_hc_weights[key])*rsqrt
    pre,post,comb = sinkhorn_cpu(mixes,hc_scale,hc_base,self.hc_mult,self.hc_sinkhorn_iters,self.hc_eps)
    output = (pre.unsqueeze(-1)*values).sum(dim=2)
    return output.to(device=device,dtype=dtype),post,comb


def hc_post_cpu(self, x, residual, post, comb):
    values = x.detach().to(device='cpu',dtype=torch.float32)
    saved = self._hc_cpu_residual
    output = post.unsqueeze(-1)*values.unsqueeze(-2)+(comb.unsqueeze(-1)*saved.unsqueeze(-2)).sum(dim=2)
    return output.to(device=x.device,dtype=x.dtype)


def rope_fused_half(x, frequencies, inverse=False):
    key=(frequencies.untyped_storage().data_ptr(),frequencies.storage_offset(),
         tuple(frequencies.shape),str(x.device),x.ndim,inverse)
    if key not in _rotary_tables:
        _sources[('rope',key)]=frequencies
        table=frequencies.detach().float().cpu()
        cosine=torch.cat((table[...,0],table[...,0]),dim=-1)
        sine=torch.cat((table[...,1],table[...,1]),dim=-1)
        if inverse:sine=-sine
        cosine,sine=cosine[None,:,None,:],sine[None,:,None,:]
        _rotary_tables[key]=(cosine.to(x.device),sine.to(x.device))
    cosine,sine=_rotary_tables[key]
    paired=x.float().unflatten(-1,(-1,2)).transpose(-1,-2).contiguous().flatten(-2)
    if x.ndim==3:paired=paired.unsqueeze(-2)
    result=torch_npu.npu_rotary_mul(paired,cosine,sine,rotary_mode='half')
    if x.ndim==3:result=result.squeeze(-2)
    result=result.unflatten(-1,(2,-1)).transpose(-1,-2).contiguous().flatten(-2)
    x.copy_(result.to(x.dtype))
    return x


def sigmoid(x):
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, np.float32(1)/(np.float32(1)+z), z/(np.float32(1)+z))


def hc_pre_numpy(self,x,hc_fn,hc_scale,hc_base):
    values=x.detach().cpu().numpy().astype(np.float32)
    self._hc_numpy_residual=values
    key=hc_fn.data_ptr()
    if key not in _hc_weights:
        _hc_weights[key]=hc_fn.detach().float().cpu().contiguous()
        _sources[('hc',key)]=hc_fn
    flat=values.reshape(*values.shape[:2],-1)
    mixes=(flat @ _hc_weights[key].numpy().T)/np.sqrt(
        np.mean(flat*flat,axis=-1,keepdims=True)+np.float32(self.norm_eps))
    pre,post,comb=sinkhorn_numpy(mixes,hc_scale,hc_base,self.hc_mult,
                                 self.hc_sinkhorn_iters,self.hc_eps)
    output=(pre[...,None]*values).sum(axis=2)
    return torch.from_numpy(output.astype(np.float16)).to(x.device),post,comb


def hc_post_numpy(self,x,residual,post,comb):
    values=x.detach().cpu().numpy().astype(np.float32)
    output=post[...,None]*values[...,None,:]+np.einsum(
        '...ij,...id->...jd',comb,self._hc_numpy_residual,optimize=False)
    return torch.from_numpy(output.astype(np.float16)).to(x.device)


def sinkhorn_numpy(values,scale,base,hc_mult=4,sinkhorn_iters=20,eps=1e-6):
    key=(scale.data_ptr(),base.data_ptr())
    if key not in _params:
        _sources[('params',key)]=(scale,base)
        _params[key]=(scale.detach().float().cpu().numpy().copy(),
                      base.detach().float().cpu().numpy().copy())
    scale,base=_params[key]
    eps=np.float32(eps)
    pre=sigmoid(values[...,:hc_mult]*scale[0]+base[:hc_mult])+eps
    post=np.float32(2)*sigmoid(values[...,hc_mult:2*hc_mult]*scale[1]+base[hc_mult:2*hc_mult])
    comb=(values[...,2*hc_mult:]*scale[2]+base[2*hc_mult:]).reshape(*values.shape[:-1],hc_mult,hc_mult)
    comb=np.exp(comb-comb.max(axis=-1,keepdims=True))
    comb/=comb.sum(axis=-1,keepdims=True)
    comb+=eps
    comb/=comb.sum(axis=-2,keepdims=True)+eps
    for _ in range(sinkhorn_iters-1):
        comb/=comb.sum(axis=-1,keepdims=True)+eps
        comb/=comb.sum(axis=-2,keepdims=True)+eps
    return pre,post,comb


def sinkhorn_cpu(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    device = mixes.device
    key = (scale.data_ptr(), base.data_ptr())
    if key not in _params:
        _sources[('params',key)]=(scale,base)
        _params[key] = (scale.detach().float().cpu().numpy().copy(),
                        base.detach().float().cpu().numpy().copy())
    scale, base = _params[key]
    values = mixes.detach().float().cpu().numpy()
    eps = np.float32(eps)
    pre = sigmoid(values[..., :hc_mult]*scale[0]+base[:hc_mult])+eps
    post = np.float32(2)*sigmoid(values[..., hc_mult:2*hc_mult]*scale[1]+base[hc_mult:2*hc_mult])
    comb = (values[..., 2*hc_mult:]*scale[2]+base[2*hc_mult:]).reshape(*values.shape[:-1], hc_mult, hc_mult)
    comb = np.exp(comb-comb.max(axis=-1,keepdims=True))
    comb /= comb.sum(axis=-1,keepdims=True)
    comb += eps
    comb /= comb.sum(axis=-2,keepdims=True)+eps
    for _ in range(sinkhorn_iters-1):
        comb /= comb.sum(axis=-1,keepdims=True)+eps
        comb /= comb.sum(axis=-2,keepdims=True)+eps
    # One small DMA instead of three transfers.
    packed = np.concatenate((pre,post,comb.reshape(*values.shape[:-1],hc_mult*hc_mult)),axis=-1)
    result = torch.from_numpy(packed).to(device)
    return result[..., :hc_mult], result[..., hc_mult:2*hc_mult], result[..., 2*hc_mult:].unflatten(-1,(hc_mult,hc_mult))
