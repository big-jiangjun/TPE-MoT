from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        return self.dropout(x)


class OmegaPrefixConnector(nn.Module):
    """
    Qwen3 deepstack connector for VGGT-Omega aggregator tokens.

    Query:
      - Qwen deepstack visual tokens.

    Key/value:
      - global tokens: Omega camera/register tokens before patch_start_idx.
      - local tokens: Omega patch tokens after temporal/spatial merge.

    Output shape is identical to the Qwen visual token input shape.
    """

    def __init__(
        self,
        clip_dim: int,
        vggt_dim: int,
        language_dim: int,
        spatial_embeds_layer_idx: int,
        visual_temporal_merge_size: int,
        visual_spatial_merge_size: int,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        hidden_act: str = "gelu",
        bias: bool = False,
        vggt_patch_size: int = 16,
        debug: bool = False,
    ) -> None:
        super().__init__()

        if language_dim % num_heads != 0:
            raise ValueError(
                f"language_dim ({language_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.clip_dim = clip_dim
        self.vggt_dim = vggt_dim
        self.language_dim = language_dim
        self.spatial_embeds_layer_idx = spatial_embeds_layer_idx
        self.visual_temporal_merge_size = visual_temporal_merge_size
        self.visual_spatial_merge_size = visual_spatial_merge_size
        self.hidden_act = hidden_act
        self.vggt_patch_size = vggt_patch_size
        self.debug = debug

        print(f"Using VGGT-Omega spatial_embeds_layer_idx: {self.spatial_embeds_layer_idx}")
        print(f"Using VGGT-Omega patch_size: {self.vggt_patch_size}")

        self.omega_token_dim = self.vggt_dim * 2
        self.merged_dim = (
            self.omega_token_dim
            * self.visual_temporal_merge_size
            * (self.visual_spatial_merge_size ** 2)
        )

        self.visual_proj = (
            nn.Identity()
            if self.clip_dim == self.language_dim
            else nn.Linear(self.clip_dim, self.language_dim, bias=bias)
        )

        self.q_norm = Qwen2RMSNorm(self.language_dim, eps=1e-6)
        self.global_norm = Qwen2RMSNorm(self.omega_token_dim, eps=1e-6)
        self.local_norm = Qwen2RMSNorm(self.merged_dim, eps=1e-6)
        self.ffn_norm = Qwen2RMSNorm(self.language_dim, eps=1e-6)

        self.global_k_proj = nn.Linear(self.omega_token_dim, self.language_dim, bias=bias)
        self.global_v_proj = nn.Linear(self.omega_token_dim, self.language_dim, bias=bias)
        self.local_k_proj = nn.Linear(self.merged_dim, self.language_dim, bias=bias)
        self.local_v_proj = nn.Linear(self.merged_dim, self.language_dim, bias=bias)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.language_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=bias,
            batch_first=True,
        )

        self.o_proj = nn.Linear(self.language_dim, self.language_dim, bias=bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.mlp = SwiGLUFFN(
            hidden_size=self.language_dim,
            mlp_ratio=mlp_ratio,
            bias=bias,
            dropout=attention_dropout,
        )


    def _infer_vggt_patch_grid(
        self,
        patch_count: int,
        sample_tchw: torch.Tensor,
    ) -> Tuple[int, int]:
        height, width = sample_tchw.shape[-2], sample_tchw.shape[-1]
        h0 = math.ceil(height / self.vggt_patch_size)
        w0 = math.ceil(width / self.vggt_patch_size)
        if h0 * w0 == patch_count:
            return h0, w0

        aspect = height / max(width, 1)
        best = None
        best_score = None
        for h in range(1, int(math.sqrt(patch_count)) + 1):
            if patch_count % h != 0:
                continue
            w = patch_count // h
            for hh, ww in ((h, w), (w, h)):
                merge_penalty = 0
                if hh % self.visual_spatial_merge_size != 0:
                    merge_penalty += 1
                if ww % self.visual_spatial_merge_size != 0:
                    merge_penalty += 1
                score = abs((hh / max(ww, 1)) - aspect) + 10.0 * merge_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    best = (hh, ww)

        if best is None:
            raise ValueError(
                f"Cannot infer VGGT-Omega patch grid for P={patch_count}, H={height}, W={width}."
            )
        return best

    def _compute_qwen_visual_token_len_from_rows(self, rows: List[torch.Tensor]) -> int:
        token_len = 0
        for row in rows:
            t, h, w = row.tolist()
            if h % self.visual_spatial_merge_size != 0 or w % self.visual_spatial_merge_size != 0:
                raise ValueError(
                    f"Qwen grid_thw row {tuple(row.tolist())} is not divisible by "
                    f"visual_spatial_merge_size={self.visual_spatial_merge_size}."
                )
            token_len += int(t) * (int(h) // self.visual_spatial_merge_size) * (
                int(w) // self.visual_spatial_merge_size
            )
        return token_len

    def preprocess_spatial_embeds(
        self,
        spatial_embeds_list: List[List[Optional[torch.Tensor]]],
        patch_start_idx: List[int],
        grid_thw: torch.Tensor,
        image_tchw: Optional[List[torch.Tensor]] = None,
        video_tchw: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        if image_tchw is None and video_tchw is None:
            raise ValueError(
                "This connector expects image_tchw or video_tchw to infer the VGGT-Omega patch grid."
            )

        is_image_mode = image_tchw is not None
        source_tchw = image_tchw if is_image_mode else video_tchw
        all_local_tokens = []
        all_global_tokens = []
        visual_token_lens = []
        grid_idx = 0

        for i, spatial_embeds_item in enumerate(spatial_embeds_list):
            selected_layer = spatial_embeds_item[self.spatial_embeds_layer_idx]
            if selected_layer is None:
                raise ValueError(
                    f"VGGT-Omega cached layer {self.spatial_embeds_layer_idx} is None. "
                    "Use a cached layer, e.g. -1 for the final cached layer."
                )

            raw_omega_tokens = selected_layer.unsqueeze(0)
            global_tokens = raw_omega_tokens[:, :, : patch_start_idx[i], :]
            patch_tokens = raw_omega_tokens[:, :, patch_start_idx[i] :, :]

            batch_size, num_frames, patch_count, hidden_dim = patch_tokens.shape
            if batch_size != 1:
                raise ValueError(f"Expected per-sample Omega batch size 1, got {batch_size}.")
            if hidden_dim != self.omega_token_dim:
                raise ValueError(
                    f"Unexpected VGGT-Omega hidden dim: got {hidden_dim}, expected {self.omega_token_dim}."
                )
            if global_tokens.shape[1] != num_frames:
                raise ValueError("Global token and patch token temporal lengths do not match.")

            original_num_frames = num_frames
            if num_frames % self.visual_temporal_merge_size != 0:
                pad_frames = self.visual_temporal_merge_size - (num_frames % self.visual_temporal_merge_size)
                patch_tokens = torch.cat(
                    [patch_tokens, patch_tokens[:, -1:].expand(-1, pad_frames, -1, -1)],
                    dim=1,
                )
                global_tokens = torch.cat(
                    [global_tokens, global_tokens[:, -1:].expand(-1, pad_frames, -1, -1)],
                    dim=1,
                )
                num_frames = patch_tokens.shape[1]

            consumed_rows = []
            accumulated_t = 0
            if is_image_mode:
                if original_num_frames < self.visual_temporal_merge_size:
                    target_t = original_num_frames
                else:
                    remaining_rows = len(grid_thw) - grid_idx
                    merged_t = num_frames // self.visual_temporal_merge_size
                    target_t = merged_t if remaining_rows == merged_t else original_num_frames
            else:
                target_t = num_frames // self.visual_temporal_merge_size
            while accumulated_t < target_t:
                if grid_idx >= len(grid_thw):
                    raise ValueError(
                        f"VGGT-Omega sample {i} does not have enough grid_thw rows. "
                        f"accumulated_t={accumulated_t}, target_t={target_t}."
                    )
                row = grid_thw[grid_idx]
                consumed_rows.append(row)
                accumulated_t += int(row[0].item())
                grid_idx += 1

            npatch_t = num_frames // self.visual_temporal_merge_size
            if accumulated_t != target_t:
                raise ValueError(
                    f"Sample {i} temporal rows mismatch: grid_thw accumulated_t={accumulated_t}, "
                    f"expected target_t={target_t}."
                )

            visual_token_len = self._compute_qwen_visual_token_len_from_rows(consumed_rows)

            sample_tchw = source_tchw[i]
            npatch_h, npatch_w = self._infer_vggt_patch_grid(patch_count, sample_tchw)
            if npatch_h * npatch_w != patch_count:
                raise ValueError(
                    f"Failed to infer a valid VGGT-Omega patch grid for sample {i}: "
                    f"P={patch_count}, inferred=({npatch_h}, {npatch_w})."
                )
            if npatch_h % self.visual_spatial_merge_size != 0 or npatch_w % self.visual_spatial_merge_size != 0:
                raise ValueError(
                    f"Inferred VGGT-Omega grid ({npatch_h}, {npatch_w}) is not divisible by "
                    f"visual_spatial_merge_size={self.visual_spatial_merge_size}."
                )

            local_tokens = (
                patch_tokens.view(batch_size, num_frames, npatch_h, npatch_w, hidden_dim)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
            local_tokens = (
                local_tokens.view(
                    batch_size,
                    npatch_t,
                    self.visual_temporal_merge_size,
                    hidden_dim,
                    npatch_h // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                    npatch_w // self.visual_spatial_merge_size,
                    self.visual_spatial_merge_size,
                )
                .permute(0, 1, 4, 6, 5, 7, 3, 2)
                .contiguous()
            )
            local_tokens = local_tokens.reshape(
                batch_size * npatch_t * npatch_h * npatch_w,
                hidden_dim * self.visual_temporal_merge_size,
            )
            local_tokens = local_tokens.view(
                -1,
                hidden_dim * self.visual_temporal_merge_size * (self.visual_spatial_merge_size ** 2),
            )

            global_tokens = global_tokens.reshape(-1, hidden_dim)

            if self.debug:
                print("\n[VGGT-Omega Connector Debug - preprocess]")
                print("sample idx:", i)
                print("patch_start_idx:", patch_start_idx[i])
                print("selected layer shape:", tuple(raw_omega_tokens.shape))
                print("global token shape:", tuple(global_tokens.shape))
                print("patch token shape before merge:", (batch_size, num_frames, patch_count, hidden_dim))
                print("source_tchw shape:", tuple(sample_tchw.shape))
                print("consumed grid_thw rows:", consumed_rows)
                print("visual_token_len from Qwen grid_thw:", visual_token_len)
                print("Omega patch grid:", (npatch_h, npatch_w))
                print("local token count after merge:", local_tokens.shape[0])

            all_local_tokens.append(local_tokens)
            all_global_tokens.append(global_tokens)
            visual_token_lens.append(visual_token_len)

        return all_local_tokens, all_global_tokens, visual_token_lens

    def _split_visual_embeds(
        self,
        visual_embeds: torch.Tensor,
        visual_token_lens: List[int],
    ) -> List[torch.Tensor]:
        if visual_embeds.dim() == 2:
            total_tokens = sum(visual_token_lens)
            if visual_embeds.shape[0] != total_tokens:
                raise ValueError(
                    f"Visual token number mismatch: got {visual_embeds.shape[0]}, "
                    f"but Qwen grid_thw implies {total_tokens}."
                )
            return list(torch.split(visual_embeds, visual_token_lens, dim=0))

        if visual_embeds.dim() == 3:
            if visual_embeds.shape[0] != len(visual_token_lens):
                raise ValueError(
                    f"Batch size mismatch: visual batch={visual_embeds.shape[0]}, "
                    f"spatial batch={len(visual_token_lens)}."
                )
            visual_list = []
            for i, token_len in enumerate(visual_token_lens):
                if visual_embeds.shape[1] < token_len:
                    raise ValueError(
                        f"Visual seq len too short for sample {i}: got {visual_embeds.shape[1]}, "
                        f"need {token_len}."
                    )
                visual_list.append(visual_embeds[i, :token_len])
            return visual_list

        raise ValueError(
            f"Unsupported visual_embeds ndim={visual_embeds.dim()}, expected 2 or 3."
        )

    def _merge_back(
        self,
        fused_visual_list: List[torch.Tensor],
        reference_visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if reference_visual_embeds.dim() == 2:
            return torch.cat(fused_visual_list, dim=0)

        if reference_visual_embeds.dim() == 3:
            output = reference_visual_embeds.clone()
            for i, fused_visual in enumerate(fused_visual_list):
                output[i, : fused_visual.shape[0]] = fused_visual
            return output

        raise ValueError(
            f"Unsupported reference_visual_embeds ndim={reference_visual_embeds.dim()}, expected 2 or 3."
        )

    def _project_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.shape[-1] == self.language_dim:
            return visual_tokens
        if visual_tokens.shape[-1] == self.clip_dim:
            return self.visual_proj(visual_tokens)
        raise ValueError(
            f"Unexpected visual hidden dim={visual_tokens.shape[-1]}, "
            f"expected {self.language_dim} or {self.clip_dim}."
        )

    def forward(
        self,
        image_embeds: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
        spatial_embeds_list: List[List[Optional[torch.Tensor]]] = None,
        patch_start_idx: List[int] = None,
        grid_thw: torch.Tensor = None,
        image_tchw: Optional[List[torch.Tensor]] = None,
        video_tchw: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert video_embeds is not None or image_embeds is not None, (
            "Either video_embeds or image_embeds must be provided."
        )

        visual_embeds = image_embeds if image_embeds is not None else video_embeds
        visual_dtype = visual_embeds.dtype
        visual_device = visual_embeds.device

        local_token_list, global_token_list, visual_token_lens = self.preprocess_spatial_embeds(
            spatial_embeds_list=spatial_embeds_list,
            patch_start_idx=patch_start_idx,
            grid_thw=grid_thw,
            image_tchw=image_tchw,
            video_tchw=video_tchw,
        )

        visual_list = self._split_visual_embeds(visual_embeds, visual_token_lens)
        fused_visual_list = []

        for sample_idx, (visual_tokens, local_tokens, global_tokens) in enumerate(
            zip(visual_list, local_token_list, global_token_list)
        ):
            visual_tokens = visual_tokens.to(device=visual_device, dtype=visual_dtype)
            local_tokens = local_tokens.to(device=visual_device, dtype=visual_dtype)
            global_tokens = global_tokens.to(device=visual_device, dtype=visual_dtype)

            visual_tokens = self._project_visual_tokens(visual_tokens)

            local_tokens = self.local_norm(local_tokens)
            local_k = self.local_k_proj(local_tokens)
            local_v = self.local_v_proj(local_tokens)

            global_tokens = self.global_norm(global_tokens)
            global_k = self.global_k_proj(global_tokens)
            global_v = self.global_v_proj(global_tokens)

            q = self.q_norm(visual_tokens).unsqueeze(0)
            k = torch.cat([global_k, local_k], dim=0).unsqueeze(0)
            v = torch.cat([global_v, local_v], dim=0).unsqueeze(0)

            attn_out, _ = self.cross_attn(query=q, key=k, value=v, need_weights=False)
            attn_out = self.o_proj(attn_out.squeeze(0))

            fused_tokens = visual_tokens + self.attn_dropout(attn_out)

            ffn_out = self.mlp(self.ffn_norm(fused_tokens))
            fused_tokens = fused_tokens + ffn_out

            if self.debug:
                print("[VGGT-Omega Connector Debug - forward]")
                print("sample_idx:", sample_idx)
                print("visual_tokens:", tuple(visual_tokens.shape))
                print("global_tokens:", tuple(global_tokens.shape))
                print("local_tokens:", tuple(local_tokens.shape))
                print("K/V:", tuple(k.shape))
                print("fused_tokens:", tuple(fused_tokens.shape))

            fused_visual_list.append(fused_tokens)

        return self._merge_back(fused_visual_list, visual_embeds)

    def print_trainable_parameters(self) -> None:
        is_connector_trainable = any(param.requires_grad for param in self.parameters())
        print(f"OmegaPrefixConnector trainable: {is_connector_trainable}")


SpatialMLMCrossAttentionConnector = OmegaPrefixConnector
