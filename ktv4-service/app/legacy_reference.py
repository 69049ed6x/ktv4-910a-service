"""V4 reference math for dav-c100; no A2-only fused kernels are executed.

Uses the local official model structure and the same Eco-Tech checkpoint.
Routed experts stay INT8 on CPU, disjoint across four TP workers. The NPU
projections use FP16 dequantized copies of the same checkpoint weights.
The short-context index selection includes every compressed KV entry, which
is equivalent to top-k while the compressed length does not exceed top-k.
"""
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import types
import textwrap

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open
from legacy_indices import fixed_window_indices, fixed_compressed_indices, rope_on_device

from settings import ROOT, APP, CHECKPOINT


def sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult:2*hc_mult] * scale[1] + base[hc_mult:2*hc_mult]
    )
    comb = (mixes[..., 2*hc_mult:] * scale[2] + base[2*hc_mult:]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attention(q, kv, sink, indices, scale):
    b, s, h, d = q.shape
    # Stabilize decode operator shapes as the number of compressed KV grows.
    if indices.shape[-1] < 192:
        indices = F.pad(indices, (0, 192 - indices.shape[-1]), value=-1)
    valid = indices >= 0
    batch = torch.arange(b, device=kv.device)[:, None, None]
    selected = kv[batch, indices.clamp(min=0).long()]
    logits = torch.matmul(q.float(), selected.float().transpose(-1, -2)) * scale
    logits = logits.masked_fill(~valid.unsqueeze(-2), float('-inf'))
    sinks = sink[None, None, :, None].expand(b, s, h, 1)
    probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
    return torch.matmul(probs.to(q.dtype), selected).contiguous()


def apply_rope(x, frequencies, inverse=False):
    frequencies = rope_on_device(frequencies, x.device)
    shape = x.shape
    pairs = x.float().reshape(*shape[:-1], shape[-1] // 2, 2)
    if x.ndim == 3:
        frequencies = frequencies[None, :, :, :]
    else:
        frequencies = frequencies[None, :, None, :, :]
    cosine, sine = frequencies[..., 0], frequencies[..., 1]
    if inverse:
        sine = -sine
    real, imag = pairs[..., 0], pairs[..., 1]
    result = torch.stack((real*cosine-imag*sine, real*sine+imag*cosine), -1)
    x.copy_(result.reshape(shape).to(x.dtype))
    return x


def embedding_forward(self, x):
    outside = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
    local = torch.where(outside, torch.zeros_like(x), x - self.vocab_start_idx)
    result = F.embedding(local, self.weight)
    result = torch.where(outside.unsqueeze(-1), torch.zeros_like(result), result)
    dist.all_reduce(result)
    return result


def patch_attention_cache(reference):
    if not getattr(reference.Compressor, '_fixed_cache_write', False):
        source = textwrap.dedent(inspect.getsource(reference.Compressor.forward))
        def index(value):
            return f'torch.tensor([{value}], dtype=torch.int64, device="cpu").to(x.device)'
        replacements = {
            'self.ape[start_pos % ratio]':
                f'self.ape.index_select(0, {index("start_pos % ratio")}).squeeze(0)',
        }
        for name, value, slot in (
            ('kv_state', 'kv', 'ratio + start_pos % ratio'),
            ('score_state', 'score', 'ratio + start_pos % ratio'),
            ('kv_state', 'kv', 'start_pos % ratio'),
            ('score_state', 'score', 'start_pos % ratio'),
            ('kv_cache', 'kv', 'start_pos // ratio'),
        ):
            replacements[f'self.{name}[:bsz, {slot}] = {value}.squeeze(1)'] = (
                f'self.{name}[:bsz].index_copy_(1, {index(slot)}, {value})'
            )
        for old, new in replacements.items():
            if old not in source:
                raise RuntimeError(f'Official compressor pattern has changed: {old}')
            source = source.replace(old, new)
        namespace = {}
        exec(compile(source, '<fixed_compressor_cache>', 'exec'),
             reference.__dict__, namespace)
        reference.Compressor.forward = namespace['forward']
        reference.Compressor._fixed_cache_write = True
    if getattr(reference.Attention, '_fixed_cache_write', False):
        return
    source = textwrap.dedent(inspect.getsource(reference.Attention.forward))
    source = source.replace(
        'q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)',
        'q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps)).to(q.dtype)',
    )
    old = 'self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)'
    new = ('self.kv_cache[:bsz].index_copy_(1, '
           'torch.tensor([start_pos % win], dtype=torch.int64, device="cpu").to(x.device), kv)')
    if old not in source:
        raise RuntimeError('Official attention cache update pattern has changed')
    namespace = {}
    exec(compile(source.replace(old, new), '<fixed_attention_cache>', 'exec'),
         reference.__dict__, namespace)
    reference.Attention.forward = namespace['forward']
    reference.Attention._fixed_cache_write = True


def load_reference_module():
    # Replace CUDA/TileLang entry points with explicit PyTorch math.
    kernel = types.ModuleType('kernel')
    kernel.act_quant = lambda x, *args, **kwargs: x
    kernel.fp4_act_quant = lambda x, *args, **kwargs: x
    kernel.sparse_attn = sparse_attention
    kernel.hc_split_sinkhorn = sinkhorn
    def unused_quant_gemm(*args, **kwargs):
        raise RuntimeError('FP4/FP8 GEMM is not used by the W8A8 reference path')
    kernel.fp8_gemm = kernel.fp4_gemm = unused_quant_gemm
    sys.modules['kernel'] = kernel
    path = APP / 'official_reference_model.py'
    source = path.read_text().replace('torch.bfloat16', 'torch.float16')
    source = source.replace('args.max_seq_len // compress_ratio',
        "getattr(args, 'kv_cache_initial_capacity', args.max_seq_len) // compress_ratio")
    source = source.replace('args.max_seq_len // self.compress_ratio',
        "getattr(args, 'kv_cache_initial_capacity', args.max_seq_len) // self.compress_ratio")
    source = source.replace(
        'q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)',
        'q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps)).to(q.dtype)',
    )
    module = types.ModuleType('dav_c100_v4_reference')
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    original_freqs = module.precompute_freqs_cis
    def real_frequencies(*args):
        with torch.device('cpu'):
            value = torch.view_as_real(original_freqs(*args))
        # Keep positional metadata on CPU; copying a small selected range gives
        # RoPE zero-offset NPU storage without per-position AsStrided compilation.
        return value
    module.precompute_freqs_cis = real_frequencies
    module.apply_rotary_emb = apply_rope
    module.ParallelEmbedding.forward = embedding_forward
    module.get_window_topk_idxs = fixed_window_indices
    module.get_compress_topk_idxs = fixed_compressed_indices

    cpu_spec = importlib.util.spec_from_file_location(
        'partitioned_cpu_experts', APP / 'cpu_experts.py'
    )
    cpu = importlib.util.module_from_spec(cpu_spec)
    cpu_spec.loader.exec_module(cpu)

    class PartitionedMoE(nn.Module):
        def __init__(self, layer_id, args):
            super().__init__()
            self.dim = args.dim
            self.gate = module.Gate(layer_id, args)
            self.experts = cpu.CPUExpertMoE(
                args.n_routed_experts, args.dim, args.moe_inter_dim,
                dist.get_rank(), dist.get_world_size(), args.swiglu_limit, layer_id,
            )
            # Shared expert is replicated; reduce only routed partial sums.
            self.shared_experts = module.Expert(
                args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
            )
        def forward(self, x, input_ids):
            shape = x.shape
            flat = x.reshape(-1, self.dim)
            weights, ids = self.gate(flat, input_ids.flatten())
            routed = self.experts(flat, ids, weights)
            dist.all_reduce(routed)
            return (routed + self.shared_experts(flat)).reshape(shape)
    module.MoE = PartitionedMoE

    def rotate_activation(x):
        # Normalized Walsh-Hadamard transform without a CUDA extension.
        width = x.shape[-1]
        if width & (width - 1):
            raise ValueError('Hadamard dimension must be a power of two')
        value = x.float()
        step = 1
        while step < width:
            shape = value.shape
            blocks = value.reshape(*shape[:-1], -1, 2, step)
            left, right = blocks[..., 0, :], blocks[..., 1, :]
            value = torch.stack((left + right, left - right), dim=-2).reshape(shape)
            step *= 2
        return (value / math.sqrt(width)).to(x.dtype)
    module.rotate_activation = rotate_activation

    LearnedIndexer = module.Indexer
    class CompatibleIndexer(LearnedIndexer):
        """Keep checkpoint parameter names and maintain learned-indexer state."""
        def __init__(self, args, ratio):
            self.use_learned = args.max_seq_len // ratio > args.index_topk
            if self.use_learned:
                super().__init__(args, ratio)
            else:
                nn.Module.__init__(self)
                self.compress_ratio = ratio
                self.index_topk = args.index_topk
        def forward(self, x, qr, start_pos, offset):
            end_pos = start_pos + x.shape[1]
            if not self.use_learned:
                return module.get_compress_topk_idxs(
                    self.compress_ratio, x.shape[0], x.shape[1], start_pos, offset)
            if end_pos // self.compress_ratio <= self.index_topk:
                if self.compressor.kv_cache is None:
                    self.compressor.kv_cache = self.kv_cache
                    self.compressor.freqs_cis = self.freqs_cis
                self.compressor(x, start_pos)
                return module.get_compress_topk_idxs(
                    self.compress_ratio, x.shape[0], x.shape[1], start_pos, offset)
            return super().forward(x, qr, start_pos, offset)
    module.Indexer = CompatibleIndexer
    return module


def make_model(max_seq_len=None):
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    configured_max = int(config.get('max_position_embeddings', 1048576))
    if max_seq_len is None:
        max_seq_len = int(os.environ.get('DEEPSEEK_MAX_SEQ_LEN', configured_max))
    max_seq_len = int(max_seq_len)
    if max_seq_len < 1 or max_seq_len > configured_max:
        raise ValueError(
            f'max_seq_len={max_seq_len} outside checkpoint range [1,{configured_max}]'
        )
    ref = load_reference_module()
    args = ref.ModelArgs(
        max_batch_size=1, max_seq_len=max_seq_len, dtype='bf16', expert_dtype=None,
        n_layers=config['num_hidden_layers'], n_mtp_layers=0,
        dim=config['hidden_size'], vocab_size=config['vocab_size'],
        moe_inter_dim=config['moe_intermediate_size'],
        n_hash_layers=config['num_hash_layers'], n_heads=config['num_attention_heads'],
        n_routed_experts=config['n_routed_experts'], n_shared_experts=1,
        n_activated_experts=config['num_experts_per_tok'],
        route_scale=config['routed_scaling_factor'], swiglu_limit=config['swiglu_limit'],
        q_lora_rank=config['q_lora_rank'], head_dim=config['head_dim'],
        rope_head_dim=config['qk_rope_head_dim'], norm_eps=config['rms_norm_eps'],
        o_groups=config['o_groups'], o_lora_rank=config['o_lora_rank'],
        window_size=config['sliding_window'], compress_ratios=tuple(config['compress_ratios']),
        compress_rope_theta=config['compress_rope_theta'], rope_theta=config['rope_theta'],
        original_seq_len=config['rope_scaling']['original_max_position_embeddings'],
        rope_factor=config['rope_scaling']['factor'],
        index_n_heads=config['index_n_heads'], index_head_dim=config['index_head_dim'],
        index_topk=config['index_topk'], hc_mult=config['hc_mult'],
        hc_sinkhorn_iters=config['hc_sinkhorn_iters'], hc_eps=config['hc_eps'],
    )
    args.kv_cache_initial_capacity = min(max_seq_len,
        max(128, int(os.environ.get('DEEPSEEK_KV_INITIAL_TOKENS', '2048'))))
    torch.set_default_dtype(torch.float16)
    with torch.device('npu'):
        model = ref.Transformer(args).eval()
    model.kv_cache_capacity = args.kv_cache_initial_capacity
    model.requires_grad_(False)
    return model


@torch.no_grad()
def load_checkpoint(model):
    params = dict(model.named_parameters())
    seen = set()
    rank, world = dist.get_rank(), dist.get_world_size()
    expert_pattern = re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|weight_scale|weight_offset)$')
    files = sorted(CHECKPOINT.glob('*.safetensors'))
    for file_index, path in enumerate(files):
        with safe_open(path, framework='pt', device='cpu') as reader:
            keys = set(reader.keys())
            for name in sorted(keys):
                match = expert_pattern.match(name)
                if match:
                    layer, expert, matrix = map(int, match.group(1, 2, 3))
                    if layer < len(model.layers):
                        target = model.layers[layer].ffn.experts
                        if target.expert_start <= expert < target.expert_end:
                            target.load_tensor(expert, matrix, match.group(4), reader.get_tensor(name))
                    continue
                if name not in params:
                    continue
                tensor = reader.get_tensor(name)
                if tensor.dtype == torch.int8:
                    scale_key = name.replace('.weight', '.weight_scale')
                    offset_key = name.replace('.weight', '.weight_offset')
                    if scale_key not in keys or offset_key not in keys:
                        raise RuntimeError(f'Quantization metadata not in same shard: {name}')
                    tensor = tensor.float() * reader.get_tensor(scale_key) + reader.get_tensor(offset_key)
                # Each attention head/group projection and vocab matrix is TP-sharded.
                dim = None
                if name in ('embed.weight', 'head.weight'):
                    dim = 0
                elif name.endswith(('.attn.wq_b.weight', '.attn.wo_a.weight', '.attn.attn_sink', '.attn.indexer.wq_b.weight', '.attn.indexer.weights_proj.weight')):
                    dim = 0
                elif name.endswith('.attn.wo_b.weight'):
                    dim = 1
                if dim is not None:
                    tensor = tensor.chunk(world, dim=dim)[rank]
                dest = params[name]
                if tensor.shape != dest.shape:
                    raise RuntimeError(f'{name}: checkpoint {tensor.shape} != model {dest.shape}')
                # Clone mmap-backed storage before NPU DMA (driver mmap-pin issue).
                dest.copy_(tensor.to(dest.dtype).contiguous().clone())
                seen.add(name)
        print(f'WEIGHT_SHARD rank={rank} file={file_index+1}/{len(files)}', flush=True)
    missing = set(params) - seen
    if missing:
        raise RuntimeError(f'Unloaded NPU parameters: {sorted(missing)}')
    for layer in model.layers:
        layer.ffn.experts.finish_loading()
    torch.npu.synchronize()
    print(f'MODEL_LOADED rank={rank} npu_allocated_gib={torch.npu.memory_allocated()/2**30:.3f}', flush=True)
