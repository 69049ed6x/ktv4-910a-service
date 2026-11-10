"""Causal window/compressed indices with stable decode operator shapes."""
from functools import lru_cache

import torch

IMPLEMENTATION_VERSION = 4
_rope_cache = {}


def rope_on_device(frequencies, device):
    if frequencies.device == device:
        return frequencies
    key = (frequencies.untyped_storage().data_ptr(), frequencies.storage_offset(),
           tuple(frequencies.shape), tuple(frequencies.stride()), str(device))
    if key not in _rope_cache:
        _rope_cache[key] = frequencies.to(device).contiguous()
    return _rope_cache[key]


def window_cpu(window_size, batch, length, position):
    with torch.device('cpu'):
        if position >= window_size - 1:
            slot = position % window_size
            matrix = torch.cat((torch.arange(slot + 1, window_size),
                                torch.arange(slot + 1))).unsqueeze(0)
        elif position > 0:
            matrix = torch.full((1, window_size), -1, dtype=torch.int64)
            matrix[0, :position + 1] = torch.arange(position + 1)
        else:
            base = torch.arange(length).unsqueeze(1)
            matrix = (base - window_size + 1).clamp(0) + torch.arange(window_size)
            matrix = torch.where(matrix > base, -1, matrix)
        return matrix.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)


def compressed_cpu(ratio, batch, length, position, offset):
    with torch.device('cpu'):
        count = (position + length) // ratio
        capacity = max(64, ((count + 63) // 64) * 64) if position > 0 and length == 1 else max(64, count)
        slots = torch.arange(capacity).unsqueeze(0)
        counts = (torch.tensor([[position + 1]]) if position else
                  torch.arange(1, length + 1).unsqueeze(1)) // ratio
        matrix = torch.where(slots < counts, slots + offset, -1)
        return matrix.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)


@lru_cache(512)
def fixed_window_indices(window_size, batch, length, position):
    return window_cpu(window_size, batch, length, position).to('npu')


@lru_cache(1024)
def fixed_compressed_indices(ratio, batch, length, position, offset):
    return compressed_cpu(ratio, batch, length, position, offset).to('npu')
