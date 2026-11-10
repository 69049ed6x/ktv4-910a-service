"""Inference only: active optimizations, finite-logit validation and sampling."""
import sys
import time
import torch
import torch.distributed as dist
from settings import RUN, atomic_json
from legacy_reference import patch_attention_cache


def initialize(model):
    # Freeze the tested configuration once; no hot reloads or profiling hooks.
    import fast_small_ops
    import fast_moe
    from fast_cpu_experts import fast_decode_forward
    from fast_attention import direct_cache_attention
    reference = sys.modules['dav_c100_v4_reference']
    torch.set_num_threads(32)
    reference.apply_rotary_emb = fast_small_ops.rope_fused_half
    reference.sparse_attn = direct_cache_attention
    reference.hc_split_sinkhorn = fast_small_ops.sinkhorn_cpu
    reference.RMSNorm.forward = fast_small_ops.fused_rms_forward
    cpu_class = type(model.layers[0].ffn.experts)
    cpu_class._use_a8 = False
    cpu_class._original_forward = cpu_class.forward
    cpu_class.forward = fast_decode_forward
    for layer in model.layers:
        fast_moe.compact_shared_weights(layer.ffn.shared_experts)
    torch.npu.synchronize()
    torch.npu.empty_cache()
    reference.Expert.forward = fast_moe.shared_fused_forward
    patch_attention_cache(reference)
    for layer in model.layers:
        attention = layer.attn
        if attention.freqs_cis.device.type != 'cpu':
            attention.freqs_cis = attention.freqs_cis.cpu()
        if attention.compress_ratio:
            attention.compressor.freqs_cis = attention.freqs_cis


def reset_cache(model):
    for layer in model.layers:
        attention = layer.attn
        attention.kv_cache.zero_()
        if attention.compress_ratio:
            attention.compressor.kv_state.zero_()
            attention.compressor.score_state.fill_(float('-inf'))
        indexer = getattr(attention, 'indexer', None)
        if indexer is not None and getattr(indexer, 'use_learned', False):
            indexer.kv_cache.zero_()
            indexer.compressor.kv_state.zero_()
            indexer.compressor.score_state.fill_(float('-inf'))


def forward_step(model, command):
    started = time.perf_counter()
    position = command['position']
    if position == 0:
        reset_cache(model)
    if position + len(command['tokens']) > model.max_seq_len:
        raise ValueError('Forward exceeds configured model context')
    from dynamic_kv_cache import ensure_capacity
    ensure_capacity(model, position + len(command['tokens']))
    if position and len(command['tokens']) != 1:
        raise ValueError('Decode requires one token per step')
    with torch.inference_mode(), torch.device('npu'):
        current = torch.tensor([command['tokens']], dtype=torch.int64, device='npu')
        logits = model(current, position)
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Non-finite logits')
        token = torch.zeros(1, dtype=torch.int64, device='npu')
        if dist.get_rank() == 0:
            scores = logits[0].float().cpu()
            sampling = command.get('sampling', {})
            history = command.get('history', [])
            if history:
                ids = torch.tensor(history, dtype=torch.int64, device='cpu')
                counts = torch.bincount(ids, minlength=scores.numel())
                penalty = sampling.get('repetition_penalty', 1.0)
                if penalty != 1:
                    repeated = counts > 0
                    scores[repeated] = torch.where(
                        scores[repeated] > 0, scores[repeated] / penalty,
                        scores[repeated] * penalty,
                    )
            output_history = command.get('output_history', [])
            if output_history:
                counts = torch.bincount(
                    torch.tensor(output_history, dtype=torch.int64, device='cpu'),
                    minlength=scores.numel(),
                )
                scores -= sampling.get('presence_penalty', 0.0) * (counts > 0)
                scores -= sampling.get('frequency_penalty', 0.0) * counts
            for key, bias in (sampling.get('logit_bias') or {}).items():
                scores[int(key)] += bias
            allowed = sampling.get('allowed_token_ids')
            if allowed is not None:
                restricted = torch.full_like(scores, float('-inf'))
                allowed = torch.tensor(allowed, dtype=torch.int64, device='cpu')
                restricted[allowed] = scores[allowed]
                scores = restricted
            if sampling.get('mask_eos'):
                scores[1] = float('-inf')
            temperature = sampling.get('temperature', 0.0)
            if temperature == 0:
                sampled = scores.argmax().item()
            else:
                scores /= temperature
                top_k = sampling.get('top_k', -1)
                if top_k > 0:
                    threshold = scores.topk(min(top_k, scores.numel())).values[-1]
                    scores.masked_fill_(scores < threshold, float('-inf'))
                sorted_scores, sorted_ids = scores.sort(descending=True)
                probabilities = sorted_scores.softmax(-1)
                min_p = sampling.get('min_p', 0.0)
                if min_p > 0:
                    probabilities.masked_fill_(probabilities < probabilities[0] * min_p, 0)
                top_p = sampling.get('top_p', 1.0)
                if top_p < 1:
                    remove = probabilities.cumsum(-1) - probabilities > top_p
                    probabilities.masked_fill_(remove, 0)
                generator = None
                if sampling.get('seed') is not None:
                    generator = torch.Generator(device='cpu').manual_seed(
                        sampling['seed'] + sampling.get('generated_count', 0)
                    )
                sampled = sorted_ids[torch.multinomial(probabilities, 1, generator=generator)].item()
            token = torch.tensor([sampled], dtype=torch.int64, device='cpu').to('npu')
        dist.broadcast(token, src=0)
        sampled = token.item()
    if dist.get_rank() == 0:
        atomic_json(RUN / f'result_{command["id"]}.json', {
            'token_id': sampled, 'seconds': time.perf_counter() - started,
        })
