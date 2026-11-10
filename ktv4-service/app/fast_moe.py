"""Optional shared projection fusion and CPU reduction of resident experts."""
import torch
import torch.nn.functional as F
import torch.distributed as dist

FAST_MOE_VERSION=3


def compact_shared_weights(module):
    if getattr(module,'_storage_compacted',False):return
    if not hasattr(module,'_fused_w13'):
        module._fused_w13=torch.cat((module.w1.weight,module.w3.weight),dim=0).contiguous()
    rows=module.w1.weight.shape[0]
    # Keep the original projections as exact views, so fallback math still works.
    module.w1.weight.data=module._fused_w13[:rows]
    module.w3.weight.data=module._fused_w13[rows:]
    module._storage_compacted=True


def restore_shared_weights(module):
    if not getattr(module,'_storage_compacted',False):return
    module.w1.weight.data=module.w1.weight.detach().clone()
    module.w3.weight.data=module.w3.weight.detach().clone()
    del module._fused_w13
    module._storage_compacted=False


def shared_fused_forward(self, x, weights=None):
    if not hasattr(self,'_fused_w13'):
        self._fused_w13=torch.cat((self.w1.weight,self.w3.weight),dim=0).contiguous()
    gate,up=F.linear(x,self._fused_w13).float().chunk(2,-1)
    if self.swiglu_limit>0:
        gate=gate.clamp(max=self.swiglu_limit)
        up=up.clamp(-self.swiglu_limit,self.swiglu_limit)
    activated=F.silu(gate)*up
    if weights is not None:activated=weights*activated
    return self.w2(activated.to(x.dtype))


def cpu_reduce_moe_forward(self,x,input_ids):
    shape=x.shape
    flat=x.reshape(-1,self.dim)
    weights,ids=self.gate(flat,input_ids.flatten())
    host_x=flat.to(device='cpu',dtype=torch.float32)
    host_ids=ids.to(device='cpu',dtype=torch.int64)
    host_weights=weights.to(device='cpu',dtype=torch.float32)
    shared=self.shared_experts(flat)
    partial=self.experts(host_x,host_ids,host_weights)
    # FP32 reduction avoids an intermediate FP16 rounding before the sum.
    dist.all_reduce(partial,group=self._cpu_reduce_group)
    routed=partial.to(device=x.device,dtype=x.dtype)
    return (routed+shared).reshape(shape)


def shared_overlap_moe_forward(self,x,input_ids):
    shape=x.shape
    flat=x.reshape(-1,self.dim)
    weights,ids=self.gate(flat,input_ids.flatten())
    host_x=flat.to(device='cpu',dtype=torch.float32)
    host_ids=ids.to(device='cpu',dtype=torch.int64)
    host_weights=weights.to(device='cpu',dtype=torch.float32)
    # Enqueue the NPU shared expert before computing the resident CPU experts.
    shared=self.shared_experts(flat)
    partial=self.experts(host_x,host_ids,host_weights)
    routed=partial.to(device=x.device,dtype=x.dtype)
    dist.all_reduce(routed)
    return (routed+shared).reshape(shape)
