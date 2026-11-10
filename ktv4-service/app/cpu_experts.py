# SPDX-License-Identifier: Apache-2.0
"""CPU-resident INT8 experts, partitioned by expert ID across TP workers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CPUExpertMoE(nn.Module):
    """Own one disjoint expert shard and return its partial routed output.

    Checkpoint weights stay INT8 in CPU RAM. The CPU weight-only GEMM accepts
    FP32 activations without adding activation quantization error. The owning
    Ascend MoE adds its TP-sharded shared expert and reduces the partial sums.
    Tensors deliberately are not Parameters/buffers: model device moves must
    never move the expert weights onto NPU.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_rank: int,
        tp_size: int,
        swiglu_limit: float | None,
        layer_index: int,
    ) -> None:
        super().__init__()
        if num_experts % tp_size:
            raise ValueError("Expert count must be divisible by TP size")
        self.num_local_experts = num_experts // tp_size
        self.expert_start = tp_rank * self.num_local_experts
        self.expert_end = self.expert_start + self.num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.layer_index = layer_index
        self.tp_rank = tp_rank
        self._seen: set[tuple[int, int, str]] = set()
        self._ready = False
        self.w13 = self.w2 = None
        self.s13 = self.s2 = None
        self.o13 = self.o2 = None

    def _allocate(self) -> None:
        if self.w13 is not None:
            return
        count = self.num_local_experts
        hidden = self.hidden_size
        width = self.intermediate_size
        self.w13 = torch.empty(
            count, 2 * width, hidden, dtype=torch.int8, device="cpu"
        )
        self.w2 = torch.empty(
            count, hidden, width, dtype=torch.int8, device="cpu"
        )
        self.s13 = torch.empty(count, 2 * width, device="cpu", dtype=torch.float32)
        self.s2 = torch.empty(count, hidden, device="cpu", dtype=torch.float32)
        self.o13 = torch.empty_like(self.s13)
        self.o2 = torch.empty_like(self.s2)

    @torch.no_grad()
    def load_tensor(
        self, expert: int, matrix: int, field: str, tensor: torch.Tensor
    ) -> bool:
        if not self.expert_start <= expert < self.expert_end:
            return False
        if matrix not in (1, 2, 3) or field not in (
            "weight", "weight_scale", "weight_offset"
        ):
            raise ValueError(f"Unsupported expert tensor: {matrix}/{field}")
        self._allocate()
        index = expert - self.expert_start
        if field == "weight":
            if tensor.dtype != torch.int8:
                raise ValueError("CPU expert loader expects INT8 W8A8 weights")
            dest = self.w2[index] if matrix == 2 else self.w13[index]
        else:
            stacks = (self.s13, self.s2) if field == "weight_scale" else (
                self.o13, self.o2
            )
            dest = stacks[1 if matrix == 2 else 0][index]
            tensor = tensor.reshape(-1)
        if matrix == 1:
            dest = dest[:self.intermediate_size]
        elif matrix == 3:
            dest = dest[self.intermediate_size:]
        if dest.shape != tensor.shape:
            raise ValueError(f"Expert tensor shape {tensor.shape} != {dest.shape}")
        dest.copy_(tensor)
        self._seen.add((expert, matrix, field))
        return True

    def finish_loading(self) -> None:
        expected = self.num_local_experts * 9
        if len(self._seen) != expected:
            raise RuntimeError(
                f"Layer {self.layer_index}/rank {self.tp_rank}: loaded "
                f"{len(self._seen)}/{expected} expert tensors"
            )
        self._ready = True
        size = (self.w13.nbytes + self.w2.nbytes) / 2**30
        print(
            f"[CPUExpertMoE] rank={self.tp_rank} layer={self.layer_index} "
            f"experts=[{self.expert_start},{self.expert_end}) "
            f"INT8 CPU resident={size:.3f} GiB",
            flush=True,
        )

    @staticmethod
    def _linear(x, weight, scale, offset):
        # This CPU kernel multiplies floating activations by INT8 weights;
        # scale is per output row. Include the affine offset exactly.
        output = torch._weight_int8pack_mm(x, weight, scale)
        return output.add_(x.sum(dim=-1, keepdim=True) * offset)

    @torch.no_grad()
    def forward(self, hidden_states, topk_ids, topk_weights):
        if not self._ready:
            raise RuntimeError("CPU experts used before all weights were loaded")
        device, dtype = hidden_states.device, hidden_states.dtype
        x = hidden_states.to(device="cpu", dtype=torch.float32)
        ids = topk_ids.to(device="cpu", dtype=torch.int64)
        weights = topk_weights.to(device="cpu", dtype=torch.float32)
        output = torch.zeros(
            x.shape[0], self.hidden_size, dtype=torch.float32, device="cpu"
        )
        local = (ids >= self.expert_start) & (ids < self.expert_end)
        token, slot = torch.where(local)
        selected = ids[token, slot]
        for expert in torch.unique(selected).tolist():
            mask = selected == expert
            tokens = token[mask]
            mixing = weights[tokens, slot[mask]].unsqueeze(1)
            index = expert - self.expert_start
            gate_up = self._linear(
                x[tokens], self.w13[index], self.s13[index], self.o13[index]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            if self.swiglu_limit is not None and self.swiglu_limit > 0:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            activated = F.silu(gate) * up
            result = self._linear(
                activated, self.w2[index], self.s2[index], self.o2[index]
            )
            output.index_add_(0, tokens, result * mixing)
        return output.to(device=device, dtype=dtype)
