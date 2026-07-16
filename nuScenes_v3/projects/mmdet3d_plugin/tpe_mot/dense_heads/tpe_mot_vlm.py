# Copyright 2026 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import copy
import math
from typing import Literal, Optional, Dict, Any
import glob
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
import deepspeed
import numpy as np

from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.auto import CONFIG_MAPPING
from transformers import AutoConfig, AutoProcessor

from qwen_vl_utils import smart_resize
from PIL import Image

from .flex_attention_opt import flex_attention_forward_optimized

# This file lives at projects/mmdet3d_plugin/tpe_mot/dense_heads.
REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
VGGT_OMEGA_ROOT = THIRD_PARTY_ROOT / "vggt-omega"
for _path in (THIRD_PARTY_ROOT, VGGT_OMEGA_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from .omega_prefix_connector import OmegaPrefixConnector
    from .omega_spatial_encoder import (
        VGGTOmegaSpatialEncoderConfig,
        VGGTOmegaSpatialEncoderPreTrainedModel,
    )
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape,
        _crop_to_supported_aspect_ratio,
        _max_size_target_shape,
        _pad_images_to_common_size,
    )
except ImportError:
    OmegaPrefixConnector = None
    VGGTOmegaSpatialEncoderConfig = None
    VGGTOmegaSpatialEncoderPreTrainedModel = None
    _balanced_target_shape = None
    _crop_to_supported_aspect_ratio = None
    _max_size_target_shape = None
    _pad_images_to_common_size = None

NUSCENES_VIEW_TOKENS = [
    "<FRONT_VIEW>",
    "<FRONT_LEFT_VIEW>",
    "<FRONT_RIGHT_VIEW>",
    "<BACK_LEFT_VIEW>",
    "<BACK_RIGHT_VIEW>",
    "<BACK_VIEW>",
]

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _unwrap_lm(language_model):
    if hasattr(language_model, "base_model"):
        return language_model.base_model.model
    return language_model


# --- Code Change ---
# 删除 qwen3_action_expert 参数，从 models 列表中移除。
def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    qwen3_vl,
    qwen3_perception_expert,
    attn_implementation: str = "eager",
    q_len_rounded: int = None,
    deepstack_visual_embeds=None,
    visual_pos_masks=None,
):
    base_lm = _unwrap_lm(qwen3_vl.language_model)
    models = [base_lm, qwen3_perception_expert]
# --- End Code Change ---
# Original:
# def compute_layer_complete(
#     layer_idx,
#     inputs_embeds,
#     attention_mask,
#     position_ids,
#     qwen3_vl,
#     qwen3_perception_expert,
#     qwen3_action_expert,
#     attn_implementation: str = "eager",
#     q_len_rounded: int = None,
#     deepstack_visual_embeds=None,
#     visual_pos_masks=None,
# ):
#     base_lm = _unwrap_lm(qwen3_vl.language_model)
#     models = [base_lm, qwen3_perception_expert, qwen3_action_expert]

    query_states = []
    key_states = []
    value_states = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]

        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )

    cos, sin = base_lm.rotary_emb(dummy_tensor, position_ids)

    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    batch_size = query_states.shape[0]
    head_dim = base_lm.layers[layer_idx].self_attn.head_dim
    num_heads = base_lm.layers[layer_idx].self_attn.config.num_attention_heads

    scaling = base_lm.layers[layer_idx].self_attn.scaling

    if attn_implementation == "flex":
        att_output = flex_attention_forward_optimized(
            query_states,
            key_states,
            value_states,
            block_mask=attention_mask,
            scaling=scaling,
            q_len_rounded=q_len_rounded
        )

    elif attn_implementation == "sdpa":
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                raise ValueError(
                    "SDPA backend expects an additive attention_mask (float) like OpenPI (0 for allow, -inf for block)."
                )
            if attention_mask.dim() != 4:
                raise ValueError(
                    f"SDPA backend expects 4D attention_mask (B,1|H,Q,K); got {tuple(attention_mask.shape)}"
                )
            if attention_mask.shape[-2] != query_states.shape[2] or attention_mask.shape[-1] != key_states.shape[2]:
                raise ValueError(
                    "SDPA attention_mask shape mismatch: "
                    f"mask(Q,K)=({attention_mask.shape[-2]},{attention_mask.shape[-1]}) "
                    f"but Q,K=({query_states.shape[2]},{key_states.shape[2]})."
                )

        num_kv_heads = base_lm.layers[layer_idx].self_attn.config.num_key_value_heads
        n_rep = num_heads // num_kv_heads

        if n_rep * num_kv_heads != num_heads:
            raise ValueError(f"Invalid GQA config: num_heads={num_heads} not divisible by num_kv_heads={num_kv_heads}")

        if n_rep > 1:
            key_states = repeat_kv(key_states, n_rep)
            value_states = repeat_kv(value_states, n_rep)

        sdpa_kwargs = {
            "dropout_p": 0.0,
            "is_causal": False,
        }
        try:
            att_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                scale=scaling,
                **sdpa_kwargs,
            )
        except TypeError:
            att_output = F.scaled_dot_product_attention(
                query_states * scaling,
                key_states,
                value_states,
                attn_mask=attention_mask,
                **sdpa_kwargs,
            )

        att_output = att_output.transpose(1, 2).contiguous()
        att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    elif attn_implementation == "eager":
        att_output, _ = modeling_qwen3_vl.eager_attention_forward(
            base_lm.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling,
        )
        att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    else:
        raise ValueError(f"Unknown attn_implementation: {attn_implementation}")

    outputs_embeds = []
    start_pos = 0

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]

        curr_att_out = att_output[:, start_pos:end_pos]
        if curr_att_out.dtype != layer.self_attn.o_proj.weight.dtype:
            curr_att_out = curr_att_out.to(layer.self_attn.o_proj.weight.dtype)

        out_emb = layer.self_attn.o_proj(curr_att_out)

        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()

        out_emb = layer.post_attention_layernorm(out_emb)

        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)

        out_emb = layer.mlp(out_emb)

        out_emb = out_emb + after_first_residual

        outputs_embeds.append(out_emb)
        start_pos = end_pos

    if (deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)):
        vlm_out = outputs_embeds[0]
        ds_feat = deepstack_visual_embeds[layer_idx]
        ds_feat = ds_feat.to(device=vlm_out.device, dtype=vlm_out.dtype)
        vlm_out = vlm_out.clone()
        vlm_out[visual_pos_masks] = vlm_out[visual_pos_masks] + ds_feat
        outputs_embeds[0] = vlm_out

    return outputs_embeds




class TPEMoTVisionLanguageModel(nn.Module):
    # --- Code Change ---
    # 删除 action_expert_config 参数。
    def __init__(
        self,
        vlm_config,
        perception_expert_config,
        pretrained_path,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        train_vlm: bool = False,
        lora_cfg: Optional[Dict[str, Any]] = None,
        vggt_omega_prefix_fusion_cfg: Optional[Dict[str, Any]] = None,
        vggt_omega_perception_fusion_cfg: Optional[Dict[str, Any]] = None,
    ):
    # --- End Code Change ---
    # Original:
    # def __init__(
    #     self,
    #     vlm_config,
    #     perception_expert_config,
    #     action_expert_config,
    #     pretrained_path,
    #     precision: Literal["bfloat16", "float32"] = "bfloat16",
    #     train_vlm: bool = False,
    #     lora_cfg: Optional[Dict[str, Any]] = None,
    # ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        is_8b = (vlm_config.hidden_size == 4096)
        if is_8b:
            vlm_config_hf.text_config.tie_word_embeddings = False
            vlm_config_hf.tie_word_embeddings = False
            vlm_config_hf.vision_config.deepstack_visual_indexes = [8, 16, 24]
            vlm_config_hf.vision_config.depth = 27
            vlm_config_hf.vision_config.hidden_size = 1152
            vlm_config_hf.vision_config.intermediate_size = 4304
            vlm_config_hf.vision_config.out_hidden_size = 4096
        else:
            vlm_config_hf.text_config.tie_word_embeddings = True
            vlm_config_hf.tie_word_embeddings = True
            vlm_config_hf.vision_config.deepstack_visual_indexes = [5, 11, 17]
            vlm_config_hf.vision_config.depth = 24
            vlm_config_hf.vision_config.hidden_size = 1024
            vlm_config_hf.vision_config.intermediate_size = 4096
            vlm_config_hf.vision_config.out_hidden_size = 2048

        self.qwen3_vl = Qwen3VLForConditionalGeneration(config=vlm_config_hf)

        safetensor_files = sorted(
            glob.glob(os.path.join(pretrained_path, "*.safetensors"))
        )

        state_dict = {}

        for file in safetensor_files:
            sd = load_file(file, device="cpu")

            for k, v in sd.items():
                if "action_preprocessor.normalizer" in k:
                    continue

                new_key = k

                if new_key.startswith("model.layers."):
                    new_key = new_key.replace(
                        "model.layers.",
                        "model.language_model.layers.",
                        1
                    )

                elif new_key.startswith("model.embed_tokens."):
                    new_key = new_key.replace(
                        "model.embed_tokens.",
                        "model.language_model.embed_tokens.",
                        1
                    )

                elif new_key.startswith("model.norm."):
                    new_key = new_key.replace(
                        "model.norm.",
                        "model.language_model.norm.",
                        1
                    )

                elif new_key.startswith("visual."):
                    new_key = "model.visual." + new_key[len("visual.") :]

                state_dict[new_key] = v

        self.qwen3_vl.load_state_dict(state_dict, strict=False)

        perception_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        perception_expert_config_hf.hidden_size = perception_expert_config.hidden_size
        perception_expert_config_hf.intermediate_size = perception_expert_config.intermediate_size
        perception_expert_config_hf.num_attention_heads = perception_expert_config.num_attention_heads
        perception_expert_config_hf.num_hidden_layers = vlm_config.num_hidden_layers
        perception_expert_config_hf.num_key_value_heads = perception_expert_config.num_key_value_heads
        perception_expert_config_hf.max_position_embeddings = self.qwen3_vl.config.text_config.max_position_embeddings
        perception_expert_config_hf.rope_scaling = self.qwen3_vl.config.text_config.rope_scaling
        self.qwen3_perception_expert = Qwen3VLTextModel(config=perception_expert_config_hf)
        self.qwen3_perception_expert.embed_tokens = None

        # --- Code Change ---
        # 删除 action_expert_config_hf 和 self.qwen3_action_expert 的创建。
        # --- End Code Change ---
        # Original:
        # action_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        # action_expert_config_hf.head_dim=action_expert_config.head_dim
        # action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        # action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        # action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        # action_expert_config_hf.num_hidden_layers=vlm_config.num_hidden_layers
        # action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        # action_expert_config_hf.max_position_embeddings = self.qwen3_vl.config.text_config.max_position_embeddings
        # action_expert_config_hf.rope_scaling = self.qwen3_vl.config.text_config.rope_scaling
        # self.qwen3_action_expert = Qwen3VLTextModel(config=action_expert_config_hf)
        #
        # self.qwen3_action_expert.embed_tokens = None

        self._lora_enabled = False
        if lora_cfg is not None:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError(
                    "peft is required for LoRA training. "
                    "Install it with: pip install peft"
                )
            lora_config = LoraConfig(**lora_cfg)
            self.qwen3_vl.model.language_model = get_peft_model(
                self.qwen3_vl.model.language_model, lora_config
            )
            self.qwen3_vl.model.language_model.print_trainable_parameters()
            self._lora_enabled = True

        if not train_vlm:
            for p in self.qwen3_vl.parameters():
                p.requires_grad = False

        # --- Code Change ---
        # 从层数断言中移除 action expert。
        _vlm_config = self.qwen3_vl.config
        assert _vlm_config.text_config.num_hidden_layers == self.qwen3_perception_expert.config.num_hidden_layers
        # --- End Code Change ---
        # Original:
        # _vlm_config = self.qwen3_vl.config
        # assert _vlm_config.text_config.num_hidden_layers == self.qwen3_perception_expert.config.num_hidden_layers == self.qwen3_action_expert.config.num_hidden_layers

        self.processor = AutoProcessor.from_pretrained(pretrained_path)
        tokenizer = self.processor.tokenizer

        existing_vocab = tokenizer.get_vocab()
        tokens_to_add = [t for t in NUSCENES_VIEW_TOKENS if t not in existing_vocab]
        if len(tokens_to_add) > 0:
            tokenizer.add_tokens(tokens_to_add)
            self.qwen3_vl.resize_token_embeddings(len(tokenizer))

        self._init_vggt_omega_fusion(
            vggt_omega_prefix_fusion_cfg,
            vggt_omega_perception_fusion_cfg,
            vlm_config_hf,
        )

        # --- Code Change ---
        # 删除 action expert 的 _init_weights（模块已不存在）。
        # --- End Code Change ---
        # Original:
        # if hasattr(self.qwen3_action_expert, "_init_weights"):
        #     self.qwen3_action_expert.apply(self.qwen3_action_expert._init_weights)

        self.to_bfloat16_for_selected_params(precision)

        self._vla_attn_impl = "flex"



    @property
    def vlm_base(self):
        return self.qwen3_vl

    def _init_vggt_omega_fusion(
        self,
        prefix_cfg: Optional[Dict[str, Any]],
        perception_cfg: Optional[Dict[str, Any]],
        vlm_config_hf,
    ) -> None:
        prefix_cfg = dict(prefix_cfg or {})
        perception_cfg = dict(perception_cfg or {})
        self.vggt_omega_prefix_enabled = bool(prefix_cfg.get("enabled", False))
        self.vggt_omega_perception_enabled = bool(perception_cfg.get("enabled", False))
        self.vggt_omega_fuse_deepstack_layer = int(prefix_cfg.get("fuse_deepstack_layer", 0))
        self.vggt_omega_spatial_encoder = None
        self.vggt_omega_connector = None
        self.vggt_omega_freeze_spatial_encoder = True
        self.vggt_omega_global_layer_idx = int(perception_cfg.get("spatial_embeds_layer_idx", -1))

        if not (self.vggt_omega_prefix_enabled or self.vggt_omega_perception_enabled):
            return

        required = (
            OmegaPrefixConnector,
            VGGTOmegaSpatialEncoderConfig,
            VGGTOmegaSpatialEncoderPreTrainedModel,
            _balanced_target_shape,
            _crop_to_supported_aspect_ratio,
            _max_size_target_shape,
            _pad_images_to_common_size,
        )
        if any(item is None for item in required):
            raise ImportError(
                "VGGT-Omega fusion is enabled, but its local modules or the "
                "third_party/vggt-omega package could not be imported."
            )

        omega_cfg = prefix_cfg if self.vggt_omega_prefix_enabled else perception_cfg
        other_cfg = perception_cfg if self.vggt_omega_prefix_enabled else prefix_cfg
        primary_weight = omega_cfg.get("pretrained_weight")
        other_weight = other_cfg.get("pretrained_weight")
        if primary_weight and other_weight and str(primary_weight) != str(other_weight):
            raise ValueError(
                "Prefix and perception Omega fusion must share one pretrained_weight "
                "because they use a single spatial encoder."
            )

        image_resolution = int(omega_cfg.get("image_resolution", 256))
        patch_size = int(omega_cfg.get("patch_size", 16))
        self.vggt_omega_image_resolution = image_resolution
        self.vggt_omega_patch_size = patch_size
        self.vggt_omega_preprocess_mode = omega_cfg.get("preprocess_mode", "balanced")
        self.vggt_omega_freeze_spatial_encoder = bool(
            omega_cfg.get("freeze_spatial_encoder", True)
        )

        spatial_cfg = dict(omega_cfg.get("spatial_config", {}))
        spatial_cfg.setdefault("img_size", image_resolution)
        spatial_cfg.setdefault("patch_size", patch_size)
        spatial_cfg.setdefault("embed_dim", 1024)
        spatial_cfg.setdefault("enable_camera", True)
        spatial_cfg.setdefault("enable_depth", True)
        spatial_cfg.setdefault("enable_alignment", True)
        spatial_config = VGGTOmegaSpatialEncoderConfig(**spatial_cfg)
        self.vggt_omega_spatial_encoder = VGGTOmegaSpatialEncoderPreTrainedModel(spatial_config)

        if bool(omega_cfg.get("require_pretrained", False)) and not primary_weight:
            raise ValueError(
                "VGGT_OMEGA_PATH must be set when require_pretrained=True."
            )
        if primary_weight and str(primary_weight).lower() not in ("none", "null"):
            weight_path = Path(str(primary_weight))
            if weight_path.is_file():
                self.vggt_omega_spatial_encoder.load_pretrained_weights(str(weight_path))
            elif bool(omega_cfg.get("require_pretrained", False)):
                raise FileNotFoundError(f"VGGT-Omega weight not found: {weight_path}")
            else:
                print(f"[VGGT-Omega] Weight not found; using random spatial encoder: {weight_path}")

        if self.vggt_omega_freeze_spatial_encoder:
            self.vggt_omega_spatial_encoder.requires_grad_(False)
            self.vggt_omega_spatial_encoder.eval()

        if self.vggt_omega_prefix_enabled:
            connector_cfg = dict(prefix_cfg.get("connector_config", {}))
            self.vggt_omega_connector = OmegaPrefixConnector(
                clip_dim=vlm_config_hf.vision_config.out_hidden_size,
                vggt_dim=spatial_config.embed_dim,
                language_dim=vlm_config_hf.text_config.hidden_size,
                spatial_embeds_layer_idx=int(connector_cfg.get("spatial_embeds_layer_idx", -1)),
                visual_temporal_merge_size=vlm_config_hf.vision_config.temporal_patch_size,
                visual_spatial_merge_size=vlm_config_hf.vision_config.spatial_merge_size,
                num_heads=int(connector_cfg.get("num_heads", 8)),
                attention_dropout=float(connector_cfg.get("attention_dropout", 0.0)),
                mlp_ratio=float(connector_cfg.get("mlp_ratio", 4.0)),
                hidden_act=connector_cfg.get("hidden_act", "gelu"),
                bias=bool(connector_cfg.get("bias", False)),
                vggt_patch_size=spatial_config.patch_size,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vggt_omega_freeze_spatial_encoder and self.vggt_omega_spatial_encoder is not None:
            self.vggt_omega_spatial_encoder.eval()
        return self

    def _build_vggt_omega_video_tchw(self, frames: list[np.ndarray]) -> torch.Tensor:
        if not frames:
            raise ValueError("VGGT-Omega video fusion requires at least one frame.")
        if self.vggt_omega_preprocess_mode not in ("balanced", "max_size"):
            raise ValueError("vggt_omega preprocess_mode must be 'balanced' or 'max_size'.")
        if self.vggt_omega_image_resolution % self.vggt_omega_patch_size != 0:
            raise ValueError("VGGT-Omega image_resolution must be divisible by patch_size.")

        spatial_merge_size = self.qwen3_vl.visual.spatial_merge_size
        tensors = []
        shapes = set()
        for frame in frames:
            image = _crop_to_supported_aspect_ratio(Image.fromarray(frame).convert("RGB"))
            width, height = image.size
            aspect_ratio = height / max(width, 1)
            if self.vggt_omega_preprocess_mode == "balanced":
                target_h, target_w = _balanced_target_shape(
                    aspect_ratio, self.vggt_omega_image_resolution, self.vggt_omega_patch_size
                )
            else:
                target_h, target_w = _max_size_target_shape(
                    aspect_ratio, self.vggt_omega_image_resolution, self.vggt_omega_patch_size
                )
            patch_h = max(spatial_merge_size, int(np.ceil(target_h / self.vggt_omega_patch_size / spatial_merge_size)) * spatial_merge_size)
            patch_w = max(spatial_merge_size, int(np.ceil(target_w / self.vggt_omega_patch_size / spatial_merge_size)) * spatial_merge_size)
            image = image.resize(
                (patch_w * self.vggt_omega_patch_size, patch_h * self.vggt_omega_patch_size),
                Image.Resampling.BICUBIC,
            )
            tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).float()
            tensor = tensor.permute(2, 0, 1).contiguous().div_(255.0)
            tensors.append(tensor)
            shapes.add(tuple(tensor.shape[-2:]))
        if len(shapes) > 1:
            tensors = _pad_images_to_common_size(tensors, shapes)
        return torch.stack(tensors, dim=0)

    def _fuse_vggt_omega_video_deepstack(
        self,
        deepstack_embs,
        grid_thw: torch.Tensor,
        omega_video_tchw: list[torch.Tensor],
        batch_size: int,
        num_cameras: int,
    ):
        if not (self.vggt_omega_prefix_enabled or self.vggt_omega_perception_enabled):
            return deepstack_embs, None
        expected_sequences = batch_size * num_cameras
        if len(omega_video_tchw) != expected_sequences:
            raise ValueError(
                "VGGT-Omega video sequence count mismatch: "
                f"got {len(omega_video_tchw)}, expected B*V={expected_sequences}."
            )
        if any(video.shape[0] != 4 for video in omega_video_tchw):
            raise ValueError("VGGT-Omega v2 fusion expects four frames per camera sequence.")

        omega_video_tchw = [
            video.to(device=self.qwen3_vl.device, dtype=self.qwen3_vl.dtype)
            for video in omega_video_tchw
        ]
        if self.vggt_omega_freeze_spatial_encoder:
            with torch.no_grad():
                spatial_embeds_list, patch_start_idx = self.vggt_omega_spatial_encoder(omega_video_tchw)
        else:
            spatial_embeds_list, patch_start_idx = self.vggt_omega_spatial_encoder(omega_video_tchw)
        if len(spatial_embeds_list) != expected_sequences or len(patch_start_idx) != expected_sequences:
            raise RuntimeError("VGGT-Omega output count does not match the B*6 input sequence count.")

        fused_deepstack = list(deepstack_embs)
        if self.vggt_omega_prefix_enabled:
            layer_idx = self.vggt_omega_fuse_deepstack_layer
            if layer_idx < 0 or layer_idx >= len(fused_deepstack):
                raise IndexError(f"Invalid VGGT-Omega deepstack layer index: {layer_idx}")
            fused_layer = self.vggt_omega_connector(
                video_embeds=fused_deepstack[layer_idx],
                spatial_embeds_list=spatial_embeds_list,
                patch_start_idx=patch_start_idx,
                grid_thw=grid_thw,
                video_tchw=omega_video_tchw,
            )
            if fused_layer.shape != fused_deepstack[layer_idx].shape:
                raise ValueError("VGGT-Omega prefix connector must preserve deepstack shape.")
            fused_deepstack[layer_idx] = fused_layer

        omega_global_tokens = None
        if self.vggt_omega_perception_enabled:
            global_per_sequence = []
            global_token_count = None
            for sequence_idx, (cached_layers, patch_start) in enumerate(
                zip(spatial_embeds_list, patch_start_idx)
            ):
                selected_layer = cached_layers[self.vggt_omega_global_layer_idx]
                if selected_layer is None:
                    raise ValueError("Selected VGGT-Omega global cached layer is None.")
                if selected_layer.shape[0] != 4:
                    raise ValueError(
                        f"Omega sequence {sequence_idx} has T={selected_layer.shape[0]}, expected T=4."
                    )
                sequence_globals = selected_layer[:, :patch_start, :]
                if global_token_count is None:
                    global_token_count = sequence_globals.shape[1]
                elif sequence_globals.shape[1] != global_token_count:
                    raise ValueError("All cameras must expose the same Omega global-token count.")
                global_per_sequence.append(sequence_globals)
            global_memory = torch.stack(global_per_sequence, dim=0)
            omega_global_tokens = global_memory.reshape(
                batch_size,
                num_cameras * 4 * global_token_count,
                global_memory.shape[-1],
            )

        self._last_vggt_omega_video_info = {
            "spatial_encoder_calls": 1,
            "sequence_count": expected_sequences,
            "sequence_shapes": [list(video.shape) for video in omega_video_tchw],
            "batch_size": batch_size,
            "num_cameras": num_cameras,
            "global_memory_shape": (
                list(omega_global_tokens.shape) if omega_global_tokens is not None else None
            ),
        }
        return fused_deepstack, omega_global_tokens

    def merge_lora(self) -> None:
        if not self._lora_enabled:
            return

        merged_lm = self.qwen3_vl.model.language_model.merge_and_unload()
        self.qwen3_vl.model.language_model = merged_lm
        self._lora_enabled = False

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def embed_image(self, image_paths: list[list[str]], chunk_size: int = 6):
        flat_image_paths = [path for sample in image_paths for path in sample]

        pil_images = []
        for path in flat_image_paths:
            img = Image.open(path)
            img = to_rgb(img)
            w, h = img.size
            img = img
            pil_images.append(img)

        all_embs = []
        all_grids = []

        for i in range(0, len(pil_images), chunk_size):
            chunk = pil_images[i : i + chunk_size]

            inputs = self.processor.image_processor(
                images=chunk, return_tensors="pt", do_resize=False
            )
            pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
            grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            embs_list, _ = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid
            )

            all_embs.append(embs_list)
            all_grids.append(grid)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        return image_features, feature_lens.tolist(), all_grids

    def embed_image_tensor(self, images_tensor: torch.Tensor, chunk_size: int = 6):
        imgs = images_tensor

        if imgs.shape[2] == 3:
            imgs = imgs[:, :, [2, 1, 0], :, :]

        imgs = torch.clamp(imgs, 0, 255).byte()

        B, N, C, H, W = imgs.shape

        flat_imgs = imgs.view(-1, C, H, W)

        pil_images = []
        for i in range(flat_imgs.shape[0]):
            img_np = flat_imgs[i].permute(1, 2, 0).cpu().numpy()
            pil_images.append(Image.fromarray(img_np))

        all_embs = []
        all_grids = []
        all_deepstack_features = [[] for _ in range(len(self.qwen3_vl.config.vision_config.deepstack_visual_indexes))]

        for i in range(0, len(pil_images), chunk_size):
            chunk = pil_images[i : i + chunk_size]

            chunk_resized = []
            for img in chunk:
                w, h = img.size
                img = img
                chunk_resized.append(img)

            inputs = self.processor.image_processor(
                images=chunk_resized, return_tensors="pt", do_resize=False
            )

            pix = inputs["pixel_values"].to(self.qwen3_vl.device, self.qwen3_vl.dtype)
            grid = inputs["image_grid_thw"].to(self.qwen3_vl.device)

            embs_list, deepstack_embs, raw_feature_list = self.qwen3_vl.visual(
                hidden_states=pix, grid_thw=grid
            )

            all_embs.append(embs_list)
            all_grids.append(grid)

            for ds_idx, ds_feat in enumerate(deepstack_embs):
                all_deepstack_features[ds_idx].append(ds_feat)

            if i == 0:
                all_raw_features = [[] for _ in range(len(raw_feature_list))]

            for raw_idx, raw_feat in enumerate(raw_feature_list):
                all_raw_features[raw_idx].append(raw_feat)

            del pix, grid

        image_features = torch.cat(all_embs, dim=0)
        all_grids = torch.cat(all_grids, dim=0)

        merge_size = self.qwen3_vl.visual.spatial_merge_size
        feature_lens = (all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        deepstack_features = [torch.cat(ds_list, dim=0) for ds_list in all_deepstack_features]

        raw_features = [torch.cat(raw_list, dim=0) for raw_list in all_raw_features]

        return image_features, feature_lens.tolist(), all_grids, deepstack_features, raw_features

# 输入形状 (B, T=4, V=6, 3, H, W)
# B：batch 批次
# T=4：4 帧时序 t-3、t-2、t-1、cur
# V=6：6 个物理相机（前 / 左前 / 右前 / 后 / 左后 / 右后）
# Qwen3-VL 内置 Conv3D，temporal_patch_size=2
# 4 帧输入会被融合压缩成 2 个时序块 patch0、patch1
# patch0：融合 t-3、t-2（历史两帧）
# patch1：融合 t-1、cur（近两帧 + 当前）
# 两种模式开关 perception_multiframe
# False = 方案 B：只用 patch1（当前帧特征，6 路相机）
# True = 路径 A：patch0 + patch1 都保留，6 相机 ×2 块 = 12 伪相机
# 三类输出特征：
# image_features：给 LLM 做图文规划，每相机固定 504 token ViT 最后一层全局语义 Token，传统单路多模态方案，一次性全部塞进 LLM 最开头，作为图文 prompt 基础视觉序列；只有高层全局场景语义，丢失微小物体、纹理细节。
# deepstack_features：ViT 中间层特征，辅助 LLM 理解分层   残差注入，并行、逐层注入 LLM 每一层隐藏状态：
# raw_feature：底层视觉特征，送入 Deformable 3D 检测 / 地图头
    # --- Code Change ---
    # Reason: 新增视频 tensor 编码方法，调用 video_processor 让 ViT Conv3D 原生处理时序。
    # Qwen3-VL ViT 内置 temporal_patch_size=2，4 帧 → 2 temporal patches。
    # 支持两种感知路径（由 self.perception_multiframe 开关，默认 False=方案 B）:
    #   方案 B: raw 切到最后一个 temporal patch（当前帧 1008 tokens/cam），deepstack 保全量 504。
    #   路径 A: raw 不切，2 个 temporal patch 拆成 2 组，与 6 相机拼成 12 路伪相机。
    # PERF: 首版用逐 camera 串行循环；后续可改为 batch 所有 camera 一次 ViT forward。
    def embed_video_tensor(self, images_tensor: torch.Tensor, return_omega_global_tokens: bool = False):
        """视频输入编码: (B, T=4, V=6, 3, H, W) → visual tokens via Conv3D."""
        B, T, V, C, H, W = images_tensor.shape
        multiframe = getattr(self, "perception_multiframe", False)

        # BGR → RGB + clamp（与 embed_image_tensor 一致，6D 的 channel 在 dim=3）
        if images_tensor.shape[3] == 3:
            images_tensor = images_tensor[:, :, :, [2, 1, 0], :, :]
        images_tensor = torch.clamp(images_tensor, 0, 255).byte()

        all_embs = []
        all_grids = []
        all_deepstack_features = [[] for _ in range(len(
            self.qwen3_vl.config.vision_config.deepstack_visual_indexes
        ))]
        all_raw_features = None
        omega_video_tchw = []

        merge_size = self.qwen3_vl.visual.spatial_merge_size

        # 路径 A: raw 按 "patch-major → 12 相机" 排列
        raw_per_bv = []  # 每个 (b,v) 的 [patch0_raw, patch1_raw] list per raw_idx

        # PERF: 逐 camera 串行循环。后续优化: 将所有 camera 的 pixel_values cat 后一次 ViT forward。
        for b in range(B):
            for v in range(V):
                # 提取 T 帧 → list of numpy (H, W, C)
                frames = []
                for t in range(T):
                    frame_np = images_tensor[b, t, v].permute(1, 2, 0).cpu().numpy()
                    frames.append(frame_np)
                if self.vggt_omega_prefix_enabled or self.vggt_omega_perception_enabled:
                    omega_video_tchw.append(self._build_vggt_omega_video_tchw(frames))

                inputs = self.processor.video_processor(
                    videos=[frames], return_tensors="pt", do_resize=False
                )
                pix = inputs["pixel_values_videos"].to(
                    self.qwen3_vl.device, self.qwen3_vl.dtype
                )
                grid = inputs["video_grid_thw"].to(self.qwen3_vl.device)
                # grid: [[T_out, H_patches, W_patches]], e.g. [[2, 24, 42]]

                # ViT Conv3D 前向（visual() 同时处理 image/video grid_thw）
                embs, deepstack_embs, raw_feature_list = self.qwen3_vl.visual(
                    hidden_states=pix, grid_thw=grid
                )

                grid_t = grid[0, 0].item()  # T_out (video=2, image=1)

                # LLM prefix: 保留全部 temporal patches（完整 504 tokens）
                all_embs.append(embs)
                all_grids.append(grid)

                # deepstack: 两种模式均保全量（driving_deepstack=True 注入 LLM）
                for ds_idx, ds_feat in enumerate(deepstack_embs):
                    all_deepstack_features[ds_idx].append(ds_feat)

                # raw: 按模式处理
                if all_raw_features is None:
                    all_raw_features = [[] for _ in range(len(raw_feature_list))]
                per_bv_patches = []
                for raw_idx, raw_feat in enumerate(raw_feature_list):
                    per_patch = raw_feat.shape[0] // grid_t  # 2016 // 2 = 1008
                    if multiframe:
                        # 路径 A: 拆成 [patch0, patch1]，暂存，稍后按 12 相机顺序拼
                        patches = [raw_feat[k * per_patch:(k + 1) * per_patch]
                                   for k in range(grid_t)]
                        per_bv_patches.append(patches)
                    else:
                        # 方案 B: 只取最后一个 patch（patch1=当前帧）
                        all_raw_features[raw_idx].append(raw_feat[-per_patch:])
                if multiframe:
                    raw_per_bv.append(per_bv_patches)

                del pix, grid

        image_features = torch.cat(all_embs, dim=0)       # (B*V*504, 2048)
        all_grids = torch.cat(all_grids, dim=0)            # (B*V, 3)

        # feature_lens: 必须乘 grid_t（T 维度）。图片: 1*24*42/4=252, 视频: 2*24*42/4=504
        feature_lens = (all_grids[:, 0] * all_grids[:, 1] * all_grids[:, 2]) // (merge_size * merge_size)

        deepstack_features = [torch.cat(ds_list, dim=0) for ds_list in all_deepstack_features]
        deepstack_features, omega_global_tokens = self._fuse_vggt_omega_video_deepstack(
            deepstack_features,
            all_grids,
            omega_video_tchw,
            B,
            V,
        )
        # 未切片: each (B*V*504, 2048)，与 LLM prefix 的 504 image token 对齐

        if multiframe:
            # 路径 A-native: 把 raw 拼成 12 相机布局（patch-major: 先 6 个 patch0, 再 6 个 patch1）
            num_raw = len(all_raw_features)
            grid_t_val = int(all_grids[0, 0].item())
            for raw_idx in range(num_raw):
                ordered = []
                for b in range(B):
                    for pk in range(grid_t_val):          # patch-major
                        for v in range(V):
                            ordered.append(raw_per_bv[b * V + v][raw_idx][pk])
                all_raw_features[raw_idx] = ordered
            raw_features = [torch.cat(r, dim=0) for r in all_raw_features]
            # each (B*12*1008, 1024)，匹配下游 num_views=12
        else:
            raw_features = [torch.cat(raw_list, dim=0) for raw_list in all_raw_features]
            # 方案 B: each (B*V*1008, 1024)，匹配下游 num_views=6

        if return_omega_global_tokens:
            if omega_global_tokens is None:
                raise RuntimeError("VGGT-Omega perception fusion requested but no global tokens were produced.")
            return (
                image_features,
                feature_lens.tolist(),
                all_grids,
                deepstack_features,
                raw_features,
                omega_global_tokens,
            )
        return image_features, feature_lens.tolist(), all_grids, deepstack_features, raw_features
    # --- End Code Change ---

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        return_middle_layers: Optional[list[int]] = None,
        q_len_rounded: int | None = None,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
    ):
        middle_layer_outputs: dict[int, torch.Tensor] = {}
        if return_middle_layers is not None:
            return_middle_layers = sorted(set(return_middle_layers))

        # --- Code Change ---
        # 将 len(inputs_embeds)==3 改为 ==2，删除 suffix_output=None。
        if len(inputs_embeds) == 2 and inputs_embeds[1] is None:
            prefix_output = self.qwen3_vl.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                deepstack_visual_embeds=deepstack_visual_embeds,
                visual_pos_masks=visual_pos_masks,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            middle_output = None
        # --- End Code Change ---
        # Original:
        # if len(inputs_embeds) == 3 and inputs_embeds[1] is None and inputs_embeds[2] is None:
        #     prefix_output = self.qwen3_vl.language_model.forward(
        #         inputs_embeds=inputs_embeds[0],
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         use_cache=use_cache,
        #         deepstack_visual_embeds=deepstack_visual_embeds,
        #         visual_pos_masks=visual_pos_masks,
        #     )
        #     prefix_past_key_values = prefix_output.past_key_values
        #     prefix_output = prefix_output.last_hidden_state
        #     middle_output = None
        #     suffix_output = None

        # --- Code Change ---
        # 将 len(inputs_embeds)==3 改为 ==2，删除 suffix-only 分支。
        elif len(inputs_embeds) == 2 and inputs_embeds[0] is None:
            middle_output = self.qwen3_perception_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            prefix_past_key_values = middle_output.past_key_values
            middle_output = middle_output.last_hidden_state
            prefix_output = None
        # --- End Code Change ---
        # Original:
        # elif len(inputs_embeds) == 3 and inputs_embeds[0] is None and inputs_embeds[2] is None:
        #     middle_output = self.qwen3_perception_expert.forward(
        #         inputs_embeds=inputs_embeds[1],
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         use_cache=use_cache,
        #     )
        #     prefix_past_key_values = middle_output.past_key_values
        #     middle_output = middle_output.last_hidden_state
        #     prefix_output = None
        #     suffix_output = None
        #
        # elif len(inputs_embeds) == 3 and inputs_embeds[0] is None and inputs_embeds[1] is None:
        #     suffix_output = self.qwen3_action_expert.forward(
        #         inputs_embeds=inputs_embeds[2],
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         use_cache=use_cache,
        #     )
        #     suffix_output = suffix_output.last_hidden_state
        #     prefix_output = None
        #     middle_output = None
        #     prefix_past_key_values = None

        else:
            # --- Code Change ---
            # 从 models 列表中移除 action expert；梯度检查点检测改用
            # perception_expert（VLM 之外仅存的一个 expert）。
            models = [_unwrap_lm(self.qwen3_vl.language_model), self.qwen3_perception_expert]
            # --- End Code Change ---
            # Original:
            # models = [_unwrap_lm(self.qwen3_vl.language_model), self.qwen3_perception_expert, self.qwen3_action_expert]
            num_layers = self.qwen3_vl.config.text_config.num_hidden_layers

            # --- Code Change ---
            use_gradient_checkpointing = (
                hasattr(self.qwen3_perception_expert, "gradient_checkpointing")
                and self.qwen3_perception_expert.gradient_checkpointing
                and self.training
            )
            # --- End Code Change ---
            # Original:
            # use_gradient_checkpointing = (
            #     hasattr(self.qwen3_action_expert, "gradient_checkpointing")
            #     and self.qwen3_action_expert.gradient_checkpointing
            #     and self.training
            # )

            attn_implementation = getattr(self, '_vla_attn_impl', getattr(self.qwen3_vl.config, "_attn_implementation", "eager"))

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    # --- Code Change ---
                    # 从 compute_layer_complete 调用中删除 self.qwen3_action_expert。
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_vl,
                        self.qwen3_perception_expert,
                        attn_implementation,
                        q_len_rounded,
                        deepstack_visual_embeds,
                        visual_pos_masks,
                        use_reentrant=False,
                    )
                    # --- End Code Change ---
                else:
                    # --- Code Change ---
                    # 从 compute_layer_complete 调用中删除 self.qwen3_action_expert。
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        self.qwen3_vl,
                        self.qwen3_perception_expert,
                        attn_implementation=attn_implementation,
                        q_len_rounded=q_len_rounded,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        visual_pos_masks=visual_pos_masks,
                    )
                    # --- End Code Change ---

                if return_middle_layers is not None and layer_idx in return_middle_layers:
                    middle_layer_outputs[layer_idx] = inputs_embeds[1]

            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, use_reentrant=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            # --- Code Change ---
            # 将 outputs_embeds 解包从 3 元素改为 2 元素（无 suffix）。
            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            # --- End Code Change ---
            # Original:
            # prefix_output = outputs_embeds[0]
            # middle_output = outputs_embeds[1]
            # suffix_output = outputs_embeds[2]
            prefix_past_key_values = None

            if return_middle_layers is not None and len(middle_layer_outputs) > 0:
                for k, v in list(middle_layer_outputs.items()):
                    middle_layer_outputs[k] = models[1].norm(v)

        # --- Code Change ---
        # 返回 [prefix, middle] — 2 元素而非 3 元素。
        return [prefix_output, middle_output], prefix_past_key_values, middle_layer_outputs
        # --- End Code Change ---
        # Original:
        # return [prefix_output, middle_output, suffix_output], prefix_past_key_values, middle_layer_outputs
