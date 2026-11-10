"""Use the natural rolling-cache order for single-token attention."""
import torch
from legacy_reference import sparse_attention


def direct_cache_attention(q, kv, sink, indices, scale, fp16=False):
    if q.shape[1] != 1 or kv.shape[1] > indices.shape[-1]:
        return sparse_attention(q, kv, sink, indices, scale)
    # Before the window fills, valid slots are a prefix. Afterwards every
    # window slot is valid, and jointly permuting K/V leaves attention intact.
    selected = kv.unsqueeze(1)
    valid = indices[..., :kv.shape[1]] >= 0
    query, keys = (q, selected) if fp16 else (q.float(), selected.float())
    logits = torch.matmul(query, keys.transpose(-1, -2)).float() * scale
    logits = logits.masked_fill(~valid.unsqueeze(-2), float('-inf'))
    sinks = sink[None, None, :, None].expand(*q.shape[:3], 1)
    probs = torch.cat((logits, sinks), -1).softmax(-1)[..., :-1]
    return torch.matmul(probs.to(q.dtype), selected).contiguous()


def direct_cache_attention_fp16(q, kv, sink, indices, scale):
    return direct_cache_attention(q, kv, sink, indices, scale, fp16=True)
