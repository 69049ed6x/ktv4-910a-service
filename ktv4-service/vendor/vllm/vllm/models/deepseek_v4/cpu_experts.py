# SPDX-License-Identifier: Apache-2.0
"""CPU-resident routed experts for DeepSeek-V4-Flash on Ascend 910B.

Architecture (see plan doc): router(gate) + shared_experts stay on NPU;
ALL routed experts live in host RAM as bf16 stacks and are computed on the
Kunpeng CPU (Fiddler/KTransformers/PagedMoE principle: experts computed on
CPU, never streamed). Routing semantics are bit-identical to the FusedMoE
path because we reuse vllm's fused_topk_bias.

Weight source: Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp safetensors
  layers.{L}.ffn.experts.{E}.w{1,2,3}.weight          int8
  layers.{L}.ffn.experts.{E}.w{1,2,3}.weight_scale    f32 [rows,1]
  layers.{L}.ffn.experts.{E}.w{1,2,3}.weight_offset   f32 [rows,1]
dequantized once at load: w_bf16 = int8 * scale + offset (per output row).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    _topk_softplus_sqrt_torch,
)


def _log(msg: str) -> None:
    print(f"[CPUExpertMoE] {msg}", flush=True)


class CPUExpertMoE(nn.Module):
    """Drop-in replacement for the experts part of DeepseekV4MoE.

    is_internal_router=True so DeepseekV4MoE._forward_fused_moe calls us as
    self.experts(hidden_states=..., router_logits=<placeholder>, input_ids=...).
    """

    def __init__(
        self,
        *,
        gate,
        shared_experts,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool,
        scoring_func: str,
        routed_scaling_factor: float,
        e_score_correction_bias,
        hash_indices_table,
        swiglu_limit,
        layer_index: int,
        prefix: str = "",
    ):
        super().__init__()
        self.gate = gate  # stays on NPU, owned by DeepseekV4MoE (shared ref)
        self.shared_experts = shared_experts  # stays on NPU
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.layer_index = layer_index
        self.prefix = prefix

        # routing aux data lives on the gate (moved to NPU with the model)
        self.e_score_correction_bias = e_score_correction_bias
        self.hash_indices_table = hash_indices_table

        self.is_internal_router = True

        # expert weight stacks, created lazily from checkpoint on first use
        # w13: [E, 2*I, H] bf16, w2: [E, H, I] bf16, all on CPU
        self.w13: torch.Tensor | None = None
        self.w2: torch.Tensor | None = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    # weight loading (bypasses vllm's DefaultModelLoader for experts)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def load_from_checkpoint(self, model_path: str) -> None:
        import glob
        import re

        from safetensors import safe_open

        E, H, I = self.num_experts, self.hidden_size, self.intermediate_size
        L = self.layer_index
        w13 = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
        w2 = torch.empty(E, H, I, dtype=torch.bfloat16)
        seen = 0

        pat = re.compile(
            rf"^layers\.{L}\.ffn\.experts\.(\d+)\.w([123])\.(weight|weight_scale|weight_offset)$"
        )
        cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

        for shard in sorted(glob.glob(os.path.join(model_path, "quant_model_weights-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    m = pat.match(name)
                    if not m:
                        continue
                    e, mat, field = int(m.group(1)), int(m.group(2)), m.group(3)
                    cache.setdefault((e, mat), {})[field] = f.get_tensor(name)
                    if field == "weight_offset":
                        seen += 1
        expected = E * 3
        if seen < expected:
            raise RuntimeError(
                f"{self.prefix}: expected {expected} expert tensors for layer {L}, "
                f"found {seen} in {model_path}"
            )

        def _dequant_expert(e: int) -> None:
            for mat in (1, 3, 2):
                d = cache[(e, mat)]
                wq = d["weight"]
                scale = d["weight_scale"].reshape(-1)
                offset = d["weight_offset"].reshape(-1)
                wf = wq.to(torch.float32).mul_(scale.unsqueeze(1)).add_(
                    offset.unsqueeze(1)
                )
                wf = wf.to(torch.bfloat16)
                if mat == 1:
                    w13[e, :I] = wf
                elif mat == 3:
                    w13[e, I:] = wf
                else:
                    w2[e] = wf

        from concurrent.futures import ThreadPoolExecutor

        n_workers = int(os.environ.get("VLLM_CPU_EXPERTS_LOAD_THREADS", "8"))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_dequant_expert, range(E)))

        self.w13 = w13
        self.w2 = w2
        self._loaded = True
        _log(f"layer {L}: CPU experts ready ({E} experts, bf16, "
             f"{(w13.nbytes + w2.nbytes) / 2**30:.2f} GiB)")

    def _maybe_load(self) -> None:
        if self._loaded:
            return
        path = os.environ.get("VLLM_CPU_EXPERTS_PATH", "")
        if not path:
            raise RuntimeError(
                "CPUExpertMoE used but VLLM_CPU_EXPERTS_PATH is not set to the "
                "W8A8 checkpoint directory"
            )
        self.load_from_checkpoint(path)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,  # placeholder, ignored
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._maybe_load()

        dev = hidden_states.device
        M = hidden_states.size(0)

        # ---- routing on NPU (identical semantics to FusedMoE path) ----
        gating, _ = self.gate(hidden_states)
        bias = (
            self.e_score_correction_bias.data
            if self.e_score_correction_bias is not None
            else None
        )
        hash_table = (
            self.hash_indices_table if self.hash_indices_table is not None else None
        )
        if hash_table is not None and input_ids is None:
            raise ValueError("hash-MoE layer requires input_ids")

        # Pure-torch sqrtsoftplus routing (bit-compatible with vllm's
        # _topk_softplus_sqrt_torch; the _moe_C extension op is NOT available
        # in this build, so fused_topk_bias would crash even on NPU).
        topk_weights = torch.empty(
            M, self.top_k, dtype=torch.float32, device=dev
        )
        topk_ids = torch.empty(M, self.top_k, dtype=torch.int32, device=dev)
        _topk_softplus_sqrt_torch(
            topk_weights,
            topk_ids,
            torch.empty(M, self.top_k, dtype=torch.int32, device=dev),
            gating,
            self.renormalize,
            bias,
            input_ids,
            hash_table,
            self.routed_scaling_factor,
        )

        # ---- expert compute on CPU ----
        hs_cpu = hidden_states.to("cpu", dtype=torch.bfloat16)
        ids_cpu = topk_ids.to("cpu")
        wts_cpu = topk_weights.to("cpu", dtype=torch.bfloat16)
        out_cpu = torch.zeros(M, self.hidden_size, dtype=torch.bfloat16)

        flat_e = ids_cpu.reshape(-1)
        k = self.top_k
        flat_t = (
            torch.arange(M, dtype=torch.int64).unsqueeze(1).expand(M, k).reshape(-1)
        )
        flat_w = wts_cpu.reshape(-1)

        order = torch.argsort(flat_e, stable=True)
        sorted_e = flat_e[order]
        # boundaries of equal-expert runs
        uniq, counts = torch.unique_consecutive(sorted_e, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts

        clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        for u in range(uniq.numel()):
            e = int(uniq[u])
            s, c = int(starts[u]), int(counts[u])
            sel = order[s : s + c]
            tok = flat_t[sel]
            wgt = flat_w[sel]
            x = hs_cpu[tok]  # [T, H]
            h = F.linear(x, self.w13[e])  # [T, 2I]
            g, up = h.chunk(2, dim=-1)
            act = F.silu(g) * up
            if clamp is not None:
                act = act.clamp(-clamp, clamp)
            y = F.linear(act, self.w2[e])  # [T, H]
            out_cpu.index_add_(0, tok, y * wgt.unsqueeze(1))

        out = out_cpu.to(dev)

        # ---- shared expert on NPU ----
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
