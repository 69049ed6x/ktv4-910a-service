"""Grow NPU KV buffers as needed while keeping the checkpoint context limit."""
import torch


@torch.no_grad()
def ensure_capacity(model, required_tokens):
    current = model.kv_cache_capacity
    if required_tokens <= current:
        return
    capacity = min(model.max_seq_len, max(required_tokens, current*2))
    if required_tokens > capacity:
        raise ValueError('Requested position exceeds configured model context')
    for layer in model.layers:
        attention = layer.attn
        if not attention.compress_ratio:
            continue
        old = attention.kv_cache
        length = attention.window_size + capacity//attention.compress_ratio
        new = old.new_zeros((old.shape[0], length, old.shape[2]))
        new[:, :old.shape[1]].copy_(old)
        attention.kv_cache = new
        if attention.compressor.kv_cache is not None:
            attention.compressor.kv_cache = new[:, attention.window_size:]
        indexer = getattr(attention, 'indexer', None)
        if indexer is not None and getattr(indexer, 'use_learned', False):
            old_index = indexer.kv_cache
            new_index = old_index.new_zeros((old_index.shape[0], capacity//indexer.compress_ratio, old_index.shape[2]))
            new_index[:, :old_index.shape[1]].copy_(old_index)
            indexer.kv_cache = new_index
            if indexer.compressor.kv_cache is not None:
                indexer.compressor.kv_cache = new_index
    model.kv_cache_capacity = capacity

