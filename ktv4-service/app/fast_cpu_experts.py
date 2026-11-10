"""Single-token routing without repeated boolean gathers and index_add."""
import torch
import torch.nn.functional as F

FAST_CPU_EXPERTS_VERSION=2


def fast_decode_forward(self, hidden_states, topk_ids, topk_weights):
    if hidden_states.shape[0] != 1:
        return type(self)._original_forward(self,hidden_states,topk_ids,topk_weights)
    if not self._ready:raise RuntimeError('Experts are not loaded')
    with torch._C.DisableTorchFunction():
        device,dtype=hidden_states.device,hidden_states.dtype
        x=hidden_states.to(device='cpu',dtype=torch.float32)
        ids=topk_ids.to(device='cpu',dtype=torch.int64)[0].tolist()
        weights=topk_weights.to(device='cpu',dtype=torch.float32)[0].tolist()
        output=torch.zeros(1,self.hidden_size,dtype=torch.float32,device='cpu')
        if not hasattr(self,'_zero_offsets'):
            self._zero_offsets=(torch.count_nonzero(self.o13).item()==0 and
                                torch.count_nonzero(self.o2).item()==0)
        for expert in sorted(set(ids)):
            if not self.expert_start<=expert<self.expert_end:continue
            index=expert-self.expert_start
            if self._zero_offsets and not getattr(self,'_use_a8',False):
                gate_up=torch._weight_int8pack_mm(x,self.w13[index],self.s13[index])
            else:
                gate_up=self._linear(x,self.w13[index],self.s13[index],self.o13[index])
            gate,up=gate_up.chunk(2,-1)
            if self.swiglu_limit is not None and self.swiglu_limit>0:
                gate=gate.clamp(max=self.swiglu_limit)
                up=up.clamp(-self.swiglu_limit,self.swiglu_limit)
            activated=F.silu(gate)*up
            if self._zero_offsets and not getattr(self,'_use_a8',False):
                result=torch._weight_int8pack_mm(activated,self.w2[index],self.s2[index])
            else:
                result=self._linear(activated,self.w2[index],self.s2[index],self.o2[index])
            for slot,value in enumerate(ids):
                if value==expert:output.add_(result*weights[slot])
        return output.to(device=device,dtype=dtype)
