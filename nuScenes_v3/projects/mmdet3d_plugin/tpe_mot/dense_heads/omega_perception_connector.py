"""VGGT-Omega global-memory adapter for perception expert input tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        )


class OmegaPerceptionConnector(nn.Module):
    """Inject Omega camera/register tokens into perception expert input tokens.

    Query is the task-token sequence produced by ``embed_perception``. Key and
    value are the final VGGT-Omega cached-layer tokens before ``patch_start_idx``:
    one camera token and register tokens per input view.
    """

    def __init__(
        self,
        perception_dim: int,
        omega_dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        residual_scale_init: float = 1e-3,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if perception_dim % num_heads != 0:
            raise ValueError(
                f"perception_dim ({perception_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.perception_dim = perception_dim
        self.omega_dim = omega_dim
        self.debug = debug
        self.query_norm = Qwen2RMSNorm(perception_dim, eps=1e-6)
        self.memory_norm = Qwen2RMSNorm(omega_dim, eps=1e-6)
        self.ffn_norm = Qwen2RMSNorm(perception_dim, eps=1e-6)
        self.memory_k_proj = nn.Linear(omega_dim, perception_dim, bias=bias)
        self.memory_v_proj = nn.Linear(omega_dim, perception_dim, bias=bias)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=perception_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=bias,
            batch_first=True,
        )
        self.out_proj = nn.Linear(perception_dim, perception_dim, bias=bias)
        self.dropout = nn.Dropout(attention_dropout)
        self.ffn = SwiGLUFFN(perception_dim, mlp_ratio, bias, attention_dropout)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.beta = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, perception_embeds: torch.Tensor, omega_global_tokens: torch.Tensor) -> torch.Tensor:
        if perception_embeds.dim() != 3 or omega_global_tokens.dim() != 3:
            raise ValueError(
                "Expected perception_embeds and omega_global_tokens to be [B, N, C] tensors."
            )
        if perception_embeds.shape[0] != omega_global_tokens.shape[0]:
            raise ValueError(
                f"Batch mismatch: perception={perception_embeds.shape[0]}, "
                f"omega={omega_global_tokens.shape[0]}."
            )
        if perception_embeds.shape[-1] != self.perception_dim:
            raise ValueError(
                f"Unexpected perception dim {perception_embeds.shape[-1]}, expected {self.perception_dim}."
            )
        if omega_global_tokens.shape[-1] != self.omega_dim:
            raise ValueError(
                f"Unexpected Omega dim {omega_global_tokens.shape[-1]}, expected {self.omega_dim}."
            )

        visual_dtype = perception_embeds.dtype
        memory = omega_global_tokens.to(device=perception_embeds.device, dtype=visual_dtype)
        query = self.query_norm(perception_embeds)
        memory = self.memory_norm(memory)
        key = self.memory_k_proj(memory)
        value = self.memory_v_proj(memory)
        attended, _ = self.cross_attn(query=query, key=key, value=value, need_weights=False)
        enhanced = perception_embeds + self.alpha * self.dropout(self.out_proj(attended))
        enhanced = enhanced + self.beta * self.ffn(self.ffn_norm(enhanced))

        if self.debug:
            print(
                "[VGGT-Omega Perception Connector] "
                f"perception={tuple(perception_embeds.shape)}, memory={tuple(memory.shape)}, "
                f"alpha={float(self.alpha.detach()):.6g}, beta={float(self.beta.detach()):.6g}"
            )
        return enhanced
