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

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

import torch
from torch import nn
import torch.nn.functional as F
import logging
import deepspeed
from mmdet.models import HEADS
from mmdet.models.builder import build_head, build_loss
from timm.models.layers import Mlp
from einops import rearrange
from .tpe_mot_vlm import TPEMoTVisionLanguageModel
from torch.nn.utils.rnn import pad_sequence
from .flex_attention_opt import build_blockmask_unidrive
from .constants import (
    NUSCENES_SYSTEM_PROMPT,
    NUSCENES_USER_PROMPT_TEMPLATE,
    NUSCENES_VIEW_TOKENS,
    TARGET_SENSOR_ORDER,
    OPENPI_ATTENTION_MASK_VALUE,
    DEFAULT_PERM_INDICES,
    _NAV_CMD_FIXED,
)
from .utils import (
    make_att_2d_masks,
    sample_beta,
    create_sinusoidal_pos_embedding,
    permute_metas_per_camera_fields,
)
from .modules import OccLatentDecoder, DenseDepthNet


# --- Code Change ---
# 删除 CollisionLoss、GTMapBoundLoss、GTMapDirectionLoss 导入 ——
# 所有 collision/map 辅助 loss 代码已删除。
# --- End Code Change ---
# Original:
# from projects.mmdet3d_plugin.losses.collision_loss import CollisionLoss
# from projects.mmdet3d_plugin.losses.plan_map_loss import GTMapBoundLoss, GTMapDirectionLoss
from projects.mmdet3d_plugin.models.detection3d.target import SparseBox3DTarget
from projects.mmdet3d_plugin.models.detection3d.losses import SparseBox3DLoss
from projects.mmdet3d_plugin.models.detection3d.detection3d_blocks import SparseBox3DEncoder
from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget
from projects.mmdet3d_plugin.models.map.loss import SparseLineLoss
from projects.mmdet3d_plugin.models.map.map_blocks import SparsePoint3DEncoder
from projects.mmdet3d_plugin.models.map.decoder import SparsePoint3DDecoder
from projects.mmdet3d_plugin.ops import feature_maps_format
from projects.mmdet3d_plugin.core.box3d import *

from .tpe_mot_sparse_decoder import TPEMoTSparseDecoder
from .omega_perception_connector import OmegaPerceptionConnector

@dataclass
class DrivingBatch:
    images: torch.Tensor
    image_masks: Dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    command: Optional[torch.Tensor]
    ego_status: Optional[torch.Tensor]
    view_token_ids: Optional[torch.Tensor] = None
    traj_answer_ids: Optional[torch.Tensor] = None
    traj_answer_mask: Optional[torch.Tensor] = None
    traj_labels: Optional[torch.Tensor] = None
    prompt_lens: Optional[List[int]] = None

class QwenConfig:
    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:
    num_hidden_layers = int(variant.split('_')[-1][:-1])
    if variant.startswith("qwen3_vl_8b"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,  #理解专家用这个，完整多模态生成模型，自带 ViT 视觉编码器、图文生成头，原生支持图像 + 文本输入输出
            intermediate_size=6144,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3_8b_expert"):
        return QwenConfig(
            head_dim=128,
            hidden_size=4096,
            intermediate_size=2048,
            num_attention_heads=32,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024, #感知专家和动作专家用这个，单纯的语言模型，原生支持文本输入输出
            intermediate_size=2048, #SwiGLU MLP 升维通道,非线性激活后降维回 hidden_size
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,  #感知、动作专家和主 VLM 层数完全对齐
            num_key_value_heads=8, #每个注意力头的维度为 head_dim，hidden_size 必须能被 num_attention_heads 整除，num_key_value_heads <= num_attention_heads
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

@HEADS.register_module()
class TPEMoTPerceptionHead(nn.Module):
    def __init__(
        self,
        pretrained_path,
        vlm_variant: Literal["2b", "8b"] = "2b",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        num_sample_steps: int = 10,
        ar_loss_weight: float = 0.1,
        train_vlm: bool = False,
        enable_traj_ar: bool = False,
        traj_ar_loss_weight: float = 0.1,
        traj_ar_target: Literal["waypoint", "raw_delta", "norm_delta"] = "waypoint",
        # --- Code Change ---
        # The former action-only arguments are intentionally absent.
        # --- End Code Change ---
        occ_loss_weight: float = 1.0,
        # --- Claude Code ---
        # Reason: with_occ=False 跳过所有 occ tokens/queries/decoder/loss，加速训练
        with_occ: bool = True,
        # --- Claude Code ---
        collision_loss_weight: float = 0.0,
        map_bound_loss_weight: float = 0.0,
        map_dir_loss_weight: float = 0.0,
        map_bound_dis_thresh: float = 1.0,
        map_dir_dis_thresh: float = 2.0,
        x_min: float = -13.97,
        x_max: float = 11.77,
        y_min: float = -2.02,
        y_max: float = 55.79,
        occ_aux_loss_weight: float = 1.0,
        occ_aux_layers_1based: Optional[List[int]] = None,
        attn_implementation: Literal["eager", "sdpa", "flex"] = "flex",
        inference_attn_impl: Literal["eager", "sdpa"] = "eager",
        unified_decoder_cfg: dict = None,
        occworld_vae_config: Optional[dict] = None,
        occworld_vae_path: Optional[str] = None,
        with_depth_supervision: bool = False,
        depth_loss_weight: float = 0.2,
        num_depth_bins: int = 80,
        depth_range: tuple = (1.0, 60.0),
        depth_supervision_source: Literal["input", "output"] = "input",
        feature_source: Literal["raw", "deepstack"] = "deepstack",
        feat_grad: Optional[bool] = None,
        use_tau0_pred: bool = False,
        vlm_fusion_cfg: Optional[dict] = None,
        feature_fusion_cfg: Optional[dict] = None,
        vlm_grad_scale: float = 1.0,
        lora_cfg: Optional[dict] = None,
        lora_merge_save_dir: Optional[str] = None,
        driving_deepstack: bool = False,
        # --- Code Change ---
        # 删除 train_action_expert 参数（action expert 已完全删除）。
        # --- End Code Change ---
        # --- Code Change ---
        # Reason: temporal_cfg 传入视频帧数 + perception_multiframe 开关，供 forward_train/embed_prefix 使用。
        # perception_multiframe=True → 路径 A-native（raw 12 伪相机 + num_views=12）。
        # temporal_cfg is passed from config model.perception_head.temporal_cfg.
        temporal_cfg: Optional[dict] = None,
        # --- End Code Change ---
        vggt_omega_prefix_fusion_cfg: Optional[dict] = None,
        vggt_omega_perception_fusion_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.min_period = min_period
        self.max_period = max_period
        self.num_sample_steps = num_sample_steps

        # --- Code Change ---
        # 删除 enable_knowledge_insulation（仅用于 action expert）。
        # --- End Code Change ---
        self._inference_attn_impl = inference_attn_impl
        self.ar_loss_weight = ar_loss_weight
        self.train_vlm = train_vlm
        self.enable_traj_ar = enable_traj_ar
        self.traj_ar_loss_weight = traj_ar_loss_weight
        assert traj_ar_target in ("waypoint", "raw_delta", "norm_delta"), \
            f"traj_ar_target must be one of waypoint/raw_delta/norm_delta, got {traj_ar_target}"
        self.traj_ar_target = traj_ar_target
        self.occ_loss_weight = occ_loss_weight
        # --- Claude Code ---
        # Reason: with_occ=False 时跳过所有 occ 相关计算
        self.with_occ = with_occ
        # --- Claude Code ---
        # --- Code Change ---
        # 删除 collision_loss_fn、map_bound_loss_fn、map_dir_loss_fn 初始化 ——
        # 这些均依赖 action_out_proj / 预测轨迹。
        self.collision_loss_weight = collision_loss_weight
        self.map_bound_loss_weight = map_bound_loss_weight
        self.map_dir_loss_weight = map_dir_loss_weight
        # --- End Code Change ---

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.occ_aux_loss_weight = occ_aux_loss_weight
        if occ_aux_layers_1based is None:
            occ_aux_layers_1based = [4, 14, 24]
        self.occ_aux_layers = [int(x) - 1 for x in occ_aux_layers_1based]

        # --- Code Change ---
        # No action/planning loss module is constructed in TPE-MoT.
        # --- End Code Change ---

        self.use_tau0_pred = False

        # --- Code Change ---
        # The action expert is intentionally absent from TPE-MoT.
        if vlm_variant == "8b":
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_8b_36l')
            perception_expert_cfg = get_qwen_config('qwen3_8b_expert_36l')
        else:
            qwen3_vl_cfg = get_qwen_config('qwen3_vl_28l')
            perception_expert_cfg = get_qwen_config('qwen3_28l')
        # --- End Code Change ---

        self.lora_merge_save_dir = lora_merge_save_dir

        # --- Code Change ---
        self.qwen3_vl_with_expert = TPEMoTVisionLanguageModel(
            qwen3_vl_cfg,
            perception_expert_cfg,
            pretrained_path,
            precision=dtype,
            train_vlm=train_vlm,
            lora_cfg=lora_cfg,
            vggt_omega_prefix_fusion_cfg=vggt_omega_prefix_fusion_cfg,
            vggt_omega_perception_fusion_cfg=vggt_omega_perception_fusion_cfg,
        )
        # --- End Code Change ---

        self.attn_implementation = attn_implementation
        self.qwen3_vl_with_expert._vla_attn_impl = self.attn_implementation

        # --- Claude Code ---
        # Reason: with_occ=False 时跳过 VAE config/path 校验（occ_decoder 不会创建）
        # Original: if occworld_vae_config is None: raise ValueError(...)
        if self.with_occ:
            if occworld_vae_config is None:
                raise ValueError("occworld_vae_config must be provided via config.")
            if occworld_vae_path is None:
                raise ValueError("occworld_vae_path must be provided via config.")
        # --- Claude Code ---

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg must be provided via config.")

        self.embed_dims = unified_decoder_cfg.get("embed_dims", 256)
        self.vlm_hidden_size = perception_expert_cfg.hidden_size

        self.num_det_queries = unified_decoder_cfg.get("det_instance_bank", {}).get("num_anchor", 900)
        self.num_map_queries = unified_decoder_cfg.get("map_instance_bank", {}).get("num_anchor", 100)
        # --- Claude Code ---
        # Reason: with_occ=False 时 num_occ_queries 置 0
        # Original: self.num_occ_queries = 625
        self.num_occ_queries = 625 if self.with_occ else 0
        # --- Claude Code ---

        self.with_motion = "motion" in unified_decoder_cfg.get("task_select", [])

        if self.with_motion:
            self.num_motion_queries = unified_decoder_cfg.get("motion_instance_bank", {}).get("num_anchor", 900)
            self.motion_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
            self.motion_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        else:
            self.num_motion_queries = 0
            self.motion_proj_up = None
            self.motion_proj_down = None

        self.ego_status_dim = unified_decoder_cfg.get("ego_refine_layer", {}).get("status_dims", 10)

        self.det_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.map_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)

        self.det_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)
        self.map_proj = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        self.ego_proj_up = nn.Linear(self.embed_dims, self.vlm_hidden_size)
        self.ego_proj_down = nn.Linear(self.vlm_hidden_size, self.embed_dims)

        vision_hidden_size = self.qwen3_vl_with_expert.qwen3_vl.config.vision_config.hidden_size
        self.feature_source = feature_source

        if feat_grad is None:
            raise ValueError("feat_grad must be provided via config.")
        self.feat_grad = bool(feat_grad)

        if self.feature_source == "raw":
            proj_input_dim = vision_hidden_size
            num_proj_layers = 4
        elif self.feature_source == "deepstack":
            proj_input_dim = vision_hidden_size * 2
            num_proj_layers = 3
        else:
            raise ValueError(f"Unknown feature_source: {feature_source}")

        self.num_feature_scales = num_proj_layers

        self.feature_map_proj = nn.ModuleList([
            nn.Linear(proj_input_dim, self.embed_dims)
            for _ in range(num_proj_layers)
        ])

        self.fusion_weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dims, num_proj_layers),
                nn.Softmax(dim=-1)
            ) for _ in range(num_proj_layers)
        ])
        # --- Claude Code ---
        # Reason: with_occ=False 时不创建 occ_queries（不注册 nn.Parameter，不在 state_dict 中）
        # Original: self.num_occ_queries = 625
        # Original: self.occ_queries = nn.Parameter(torch.randn(1, self.num_occ_queries, self.vlm_hidden_size))
        self.num_occ_queries = 625 if self.with_occ else 0
        if self.with_occ:
            self.occ_queries = nn.Parameter(torch.randn(1, self.num_occ_queries, self.vlm_hidden_size))
        else:
            self.occ_queries = None
        # --- Claude Code ---

        perception_omega_cfg = dict(vggt_omega_perception_fusion_cfg or {})
        self.vggt_omega_perception_enabled = bool(perception_omega_cfg.get("enabled", False))
        self.vggt_omega_perception_connector = None
        if self.vggt_omega_perception_enabled:
            self.vggt_omega_perception_connector = OmegaPerceptionConnector(
                perception_dim=self.vlm_hidden_size,
                omega_dim=int(perception_omega_cfg.get("omega_dim", 2048)),
                num_heads=int(perception_omega_cfg.get("num_heads", 8)),
                attention_dropout=float(perception_omega_cfg.get("attention_dropout", 0.0)),
                mlp_ratio=float(perception_omega_cfg.get("mlp_ratio", 4.0)),
                bias=bool(perception_omega_cfg.get("bias", False)),
                residual_scale_init=float(perception_omega_cfg.get("residual_scale_init", 1e-3)),
                debug=bool(perception_omega_cfg.get("debug", False)),
            )

        self.with_depth_supervision = with_depth_supervision
        self.depth_loss_weight = depth_loss_weight
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range
        self.depth_supervision_source = depth_supervision_source

        if self.with_depth_supervision:
            self.depth_net = DenseDepthNet(
                embed_dims=self.embed_dims,
                in_channels=qwen3_vl_cfg.hidden_size,
                num_depth_layers=1,
                equal_focal=100,
                max_depth=60,
                loss_weight=depth_loss_weight,
            )

        if unified_decoder_cfg is None:
            raise ValueError("unified_decoder_cfg is required. Legacy det_vla_head/map_vla_head are no longer supported.")

        self.unified_decoder = build_head(unified_decoder_cfg)

        # --- Claude Code ---
        # Reason: with_occ=False 时不创建 OccLatentDecoder（不注册 nn.Module，不在 state_dict 中）
        # Original: self.occ_decoder = OccLatentDecoder(...)
        if self.with_occ:
            self.occ_decoder = OccLatentDecoder(
                qwen_dim=perception_expert_cfg.hidden_size,
                occworld_vae_config=occworld_vae_config,
                pretrained_vae_path=occworld_vae_path,
            )
        else:
            self.occ_decoder = None
        # --- Claude Code ---
        # --- Code Change ---
        # 删除所有 action 投影模块：
        #   self.action_in_proj、self.action_out_proj、
        #   self.status_mlp、self.hist_traj_encoder、
        #   self.action_time_mlp_in, self.action_time_mlp_out
        # --- End Code Change ---

        if dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif dtype == "float32":
            target_dtype = torch.float32
        else:
            target_dtype = torch.float16

        # --- Code Change ---
        # 删除 action 模块的 dtype 转换。
        # --- End Code Change ---

        self.det_proj_up.to(target_dtype)
        self.det_proj.to(target_dtype)
        self.map_proj_up.to(target_dtype)
        self.map_proj.to(target_dtype)
        self.ego_proj_up.to(target_dtype)
        self.ego_proj_down.to(target_dtype)
        if self.motion_proj_up is not None:
            self.motion_proj_up.to(target_dtype)
        if self.motion_proj_down is not None:
            self.motion_proj_down.to(target_dtype)

        self.unified_decoder.to(target_dtype)
        self.feature_map_proj.to(target_dtype)
        self.fusion_weight_generators.to(target_dtype)
        if self.vggt_omega_perception_connector is not None:
            self.vggt_omega_perception_connector.to(target_dtype)

        if hasattr(self.qwen3_vl_with_expert.qwen3_vl, 'lm_head'):
            self.qwen3_vl_with_expert.qwen3_vl.lm_head.requires_grad_(False)

        self.gradient_checkpointing_enable()
        self.gradient_checkpointing_enabled = True

        self.vlm_grad_scale = vlm_grad_scale
        self.driving_deepstack = driving_deepstack

        # --- Code Change ---
        # Reason: 从 temporal_cfg 读取视频帧数 + perception_multiframe 开关，
        # 并透传到持有 embed_video_tensor 的 qwen3_vl_with_expert 子模块。
        self.temporal_cfg = temporal_cfg or {}
        self.num_total_frames = self.temporal_cfg.get("num_total_frames", 1)
        _pmf = self.temporal_cfg.get("perception_multiframe", False)
        self.qwen3_vl_with_expert.perception_multiframe = _pmf
        self.num_perception_views = 12 if _pmf else 6
        # --- End Code Change ---

        self.view_token_str_list = NUSCENES_VIEW_TOKENS
        self.view_token_ids = None

        self._cached_block_mask = None
        self._cached_block_mask_key = None
        self._cached_q_len_rounded = None

        self.adaptive_feature_fusion = torch.compile(
            self.adaptive_feature_fusion, mode="default", fullgraph=False, dynamic=True
        )

    def fuse_perception_with_vggt_omega(self, perception_embs, omega_global_tokens):
        if not self.vggt_omega_perception_enabled:
            return perception_embs
        if self.vggt_omega_perception_connector is None:
            raise RuntimeError("VGGT-Omega perception fusion is enabled but connector is missing.")
        if omega_global_tokens is None:
            raise RuntimeError("VGGT-Omega perception fusion is enabled but global tokens are unavailable.")
        return self.vggt_omega_perception_connector(perception_embs, omega_global_tokens)

    def _get_view_token_ids(self, device):
        if self.view_token_ids is None:
            tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
            ids = []
            for t in self.view_token_str_list:
                tid = tokenizer.convert_tokens_to_ids(t)
                ids.append(tid)
            self.view_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        return self.view_token_ids.to(device)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        gc_kwargs = {"use_reentrant": False}
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
        # --- Code Change ---
        # 删除 qwen3_action_expert 的 gradient_checkpointing（模块已不存在）。
        # --- End Code Change ---

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.qwen3_vl.language_model.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_vl.visual.gradient_checkpointing_disable()
        self.qwen3_vl_with_expert.qwen3_perception_expert.gradient_checkpointing_disable()
        # --- Code Change ---
        # 删除 qwen3_action_expert 的 gradient_checkpointing_disable。
        # --- End Code Change ---

    def merge_and_save_lora(self, save_dir: Optional[str] = None) -> None:
        target = save_dir or self.lora_merge_save_dir
        if target is None:
            logging.warning(
                "[LoRA] merge_and_save_lora called but no save_dir configured. "
                "Set lora_merge_save_dir in the config or pass it explicitly."
            )
            return
        self.qwen3_vl_with_expert.merge_and_save_lora(target)

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # --- Code Change ---
    # 删除 sample_noise()、sample_time()、denorm_delta() —— 仅用于
    # flow matching / ODE 去噪（action expert）。norm_delta() 保留，
    # 因为 _build_driving_batch 中 traj_ar_target='norm_delta' 仍在使用。
    # --- End Code Change ---

    def norm_delta(self, delta_meter: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor([0.0233, 2.2707], device=delta_meter.device, dtype=delta_meter.dtype)
        std = torch.tensor([0.3427, 1.8668], device=delta_meter.device, dtype=delta_meter.dtype)
        return (delta_meter - mu) / (std + 1e-6)

    def embed_perception(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stage1_outs: dict,
    ):
        query_select = self.unified_decoder.query_select
        motion_token_256 = stage1_outs.get('motion_token', None)

        proj_dtype = self.det_proj_up.weight.dtype

        parts_embs = []
        parts_pad  = []
        parts_att  = []
        perception_lengths = {'det': 0, 'map': 0, 'occ': 0, 'ego': 0, 'motion': 0}

        # ── det ──────────────────────────────────────────────────────────────
        if 'det' in query_select and len(stage1_outs.get('det_predictions', [])) > 0:
            det_feat   = stage1_outs['det_instance_feature']
            det_anchor = stage1_outs['det_predictions'][-1]
            det_embs   = self.det_proj_up(det_feat.to(dtype=proj_dtype))
            anchor_embed_256 = self.unified_decoder.det_anchor_encoder(det_anchor)
            anchor_embed_vlm = self.det_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            det_embs = (det_embs + anchor_embed_vlm).to(dtype)
            det_n = det_embs.shape[1]
            parts_embs.append(det_embs)
            parts_pad.append(torch.ones( (batch_size, det_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, det_n), dtype=torch.bool, device=device))
            perception_lengths['det'] = det_n

        # ── map ──────────────────────────────────────────────────────────────
        if 'map' in query_select and len(stage1_outs.get('map_predictions', [])) > 0:
            map_feat   = stage1_outs['map_instance_feature']
            map_anchor = stage1_outs['map_predictions'][-1]
            map_embs   = self.map_proj_up(map_feat.to(dtype=proj_dtype))
            anchor_embed_out = self.unified_decoder.map_anchor_encoder(map_anchor)
            anchor_embed_256 = anchor_embed_out[0] if isinstance(anchor_embed_out, (tuple, list)) else anchor_embed_out
            anchor_embed_vlm = self.map_proj_up(anchor_embed_256.to(dtype=proj_dtype))
            map_embs = (map_embs + anchor_embed_vlm).to(dtype)
            map_n = map_embs.shape[1]
            parts_embs.append(map_embs)
            parts_pad.append(torch.ones( (batch_size, map_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, map_n), dtype=torch.bool, device=device))
            perception_lengths['map'] = map_n

        # ── occ (conditionally included) ──────────────────────────────────────
        # --- Claude Code ---
        # Reason: with_occ=False 时跳过 occ queries 拼接到 perception_embs
        # Original: occ_embs = self.occ_queries.expand(batch_size, -1, -1).to(...)
        if self.with_occ and self.occ_queries is not None:
            occ_embs = self.occ_queries.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
            occ_n = occ_embs.shape[1]
            parts_embs.append(occ_embs)
            parts_pad.append(torch.ones( (batch_size, occ_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, occ_n), dtype=torch.bool, device=device))
            perception_lengths['occ'] = occ_n
        # 注意: occ_len 为 0 时，后续 offsets 自动正确（o0=m1, o1=m1）
        # --- Claude Code ---

        # ── ego ──────────────────────────────────────────────────────────────
        if 'ego' in query_select:
            ego_feat   = stage1_outs['ego_instance_feature']
            ego_anchor = stage1_outs.get('ego_anchor', None)
            ego_embs   = self.ego_proj_up(ego_feat.to(dtype=proj_dtype))
            if hasattr(self.unified_decoder, 'ego_anchor_encoder') and ego_anchor is not None:
                ego_anchor_embed_256 = self.unified_decoder.ego_anchor_encoder(ego_anchor)
                ego_anchor_embed_vlm = self.ego_proj_up(ego_anchor_embed_256.to(dtype=proj_dtype))
                ego_embs = (ego_embs + ego_anchor_embed_vlm).to(dtype)
            else:
                ego_embs = ego_embs.to(dtype)
            ego_n = ego_embs.shape[1]
            parts_embs.append(ego_embs)
            parts_pad.append(torch.ones( (batch_size, ego_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, ego_n), dtype=torch.bool, device=device))
            perception_lengths['ego'] = ego_n

        # ── motion ───────────────────────────────────────────────────────────
        if motion_token_256 is not None:
            motion_token_vlm = self.motion_proj_up(motion_token_256.to(dtype=proj_dtype)).to(dtype)
            motion_n = motion_token_vlm.shape[1]
            parts_embs.append(motion_token_vlm)
            parts_pad.append(torch.ones( (batch_size, motion_n), dtype=torch.bool, device=device))
            parts_att.append(torch.zeros((batch_size, motion_n), dtype=torch.bool, device=device))
            perception_lengths['motion'] = motion_n

        # ── assemble ─────────────────────────────────────────────────────────
        if parts_embs:
            perception_embs      = torch.cat(parts_embs, dim=1)
            perception_pad_masks = torch.cat(parts_pad,  dim=1)
            perception_att_masks = torch.cat(parts_att,  dim=1)
        else:
            perception_embs      = torch.empty((batch_size, 0, self.vlm_hidden_size), dtype=dtype, device=device)
            perception_pad_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
            perception_att_masks = torch.empty((batch_size, 0), dtype=torch.bool, device=device)

        return perception_embs, perception_pad_masks, perception_att_masks, perception_lengths

    def project_and_reshape_features(
        self,
        source_features,
        bsz: int,
        all_image_grids,
        feature_source: str,
    ):
        feature_maps = []

        if source_features is None:
            return feature_maps

        if not isinstance(source_features, list):
            source_features = [source_features]

        projected_features = []
        for i, feat in enumerate(source_features):
            if i < len(self.feature_map_proj):
                feat = feat.to(self.feature_map_proj[i].weight.dtype)
                feat_proj = self.feature_map_proj[i](feat)
                projected_features.append(feat_proj)
            else:
                projected_features.append(feat)

        if all_image_grids is not None and len(all_image_grids) > 0:
            h_grid = int(all_image_grids[0, 1].item())
            w_grid = int(all_image_grids[0, 2].item())
            # num_views = 6
            # --- Code Change ---
            # Reason: 路径 A-native 下 raw 已拼成 12 伪相机（6 相机 × 2 temporal patch）。
            # 方案 B 仍为 6。expected_tokens 随之变为 bsz * num_views * h_grid * w_grid。
            num_views = getattr(self, "num_perception_views", 6)
            # --- End Code Change ---

            for ds_feat in projected_features:
                if ds_feat.dim() != 2:
                    continue

                feat_reshaped = None

                if feature_source == "raw":
                    merge_size = 2
                    h_block = h_grid // merge_size
                    w_block = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_grid * w_grid

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_vis = ds_feat.view(bsz, num_views, h_block, w_block, merge_size, merge_size, -1)
                            feat_vis = feat_vis.permute(0, 1, 2, 4, 3, 5, 6)
                            feat_reshaped = feat_vis.reshape(bsz, num_views, h_grid, w_grid, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                elif feature_source == "deepstack":
                    merge_size = 2
                    h_ds = h_grid // merge_size
                    w_ds = w_grid // merge_size
                    expected_tokens = bsz * num_views * h_ds * w_ds

                    if ds_feat.shape[0] == expected_tokens:
                        try:
                            feat_reshaped = ds_feat.view(bsz, num_views, h_ds, w_ds, -1).permute(0, 1, 4, 2, 3).contiguous()
                        except Exception:
                            feat_reshaped = None

                if feat_reshaped is not None:
                    feature_maps.append(feat_reshaped)

        if len(feature_maps) > 0:
            feature_maps = self.adaptive_feature_fusion(feature_maps)

        return feature_maps

    def adaptive_feature_fusion(self, feature_maps):
        if len(feature_maps) <= 1:
            return feature_maps

        B, N, C, H, W = feature_maps[0].shape
        fused_maps = []

        for i, feat in enumerate(feature_maps):
            feat_flat = feat.view(B * N, C, H, W)

            fusion_weights = self.fusion_weight_generators[i](feat_flat)

            weights = fusion_weights.view(B, N, len(feature_maps), 1, 1, 1)

            current_fused = 0
            for j in range(len(feature_maps)):
                other_feat = feature_maps[j]
                w = weights[:, :, j]

                if other_feat.shape[-2:] != (H, W):
                    ref = F.interpolate(
                        other_feat.flatten(0, 1), size=(H, W), mode='bilinear'
                    ).view(B, N, C, H, W)
                else:
                    ref = other_feat

                current_fused += ref * w

            fused_maps.append(current_fused)

        return fused_maps

    def compute_depth_loss(
        self,
        prefix_embs: torch.Tensor,
        prefix_out: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        all_image_grids: torch.Tensor,
        gt_depth: torch.Tensor,
        focal: Optional[torch.Tensor],
        bsz: int,
    ) -> torch.Tensor:
        if not self.with_depth_supervision or gt_depth is None:
            return torch.tensor(0.0, device=prefix_embs.device)

        feat_for_depth = None
        depth_spatial_shape = None

        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.any():
            if self.depth_supervision_source == "input":
                feat_for_depth = prefix_embs[image_mask]
            elif self.depth_supervision_source == "output":
                feat_for_depth = prefix_out[image_mask]

            if feat_for_depth is not None and all_image_grids is not None and len(all_image_grids) > 0:
                h_grid = int(all_image_grids[0, 1].item())
                w_grid = int(all_image_grids[0, 2].item())
                merge_size = 2
                h_ds, w_ds = h_grid // merge_size, w_grid // merge_size
                expected_tokens = bsz * 6 * h_ds * w_ds

                if feat_for_depth.shape[0] == expected_tokens:
                    feat_for_depth = feat_for_depth.view(bsz * 6, h_ds, w_ds, -1).permute(0, 3, 1, 2)
                    depth_spatial_shape = (h_ds, w_ds)
                else:
                    feat_for_depth = None

        if feat_for_depth is not None and depth_spatial_shape is not None:
            gt_depth = gt_depth.to(feat_for_depth.device)
            num_feat_images = feat_for_depth.shape[0]

            gt_depth_reshaped = rearrange(gt_depth, 'b n h w -> (b n) 1 h w')
            if num_feat_images < gt_depth_reshaped.shape[0]:
                gt_depth_reshaped = gt_depth_reshaped[:num_feat_images]

            H_feat, W_feat = feat_for_depth.shape[-2:]
            gt_depth_resized = F.interpolate(
                gt_depth_reshaped, size=(H_feat, W_feat), mode='nearest'
            ).squeeze(1)

            focal_flat = focal.reshape(-1) if focal is not None else None
            if focal_flat is not None and num_feat_images < focal_flat.shape[0]:
                focal_flat = focal_flat[:num_feat_images]

            return self.depth_net(feat_for_depth, focal=focal_flat, gt_depths=gt_depth_resized)

        return torch.tensor(0.0, device=prefix_embs.device)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def _build_driving_batch(
        self,
        img: torch.Tensor,
        command=None,
        ego_status=None,
        hist_traj=None,
        gt_trajs=None,
    ) -> DrivingBatch:
        device = img.device if img is not None else torch.device("cuda")
        b = int(img.shape[0]) if torch.is_tensor(img) else 1

        permute_indices = [0, 2, 1, 4, 5, 3]
        images = img[:, permute_indices]

        image_masks = {f"cam{i}": torch.ones((b,), device=device, dtype=torch.bool) for i in range(6)}

        view_token_ids = self._get_view_token_ids(device)

        if command is not None:
            if not torch.is_tensor(command):
                 try:
                     command = torch.stack(command)
                 except:
                     command = torch.tensor(command)

            command = command.to(device)
            cmd_idx = command.view(-1).long()

            idx_list = cmd_idx.tolist()
        else:
            idx_list = [2] * b

        nav_cmd_texts = [_NAV_CMD_FIXED.get(i, _NAV_CMD_FIXED[2]) for i in idx_list]

        hist_traj_np = hist_traj.detach().cpu().numpy()

        if not hasattr(self.qwen3_vl_with_expert, "processor") or self.qwen3_vl_with_expert.processor is None:
            raise RuntimeError("QwenVLAPlanningHead expects `self.qwen3_vl_with_expert.processor`")

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        system_ids = tokenizer.encode("system", add_special_tokens=False)
        user_ids = tokenizer.encode("user", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        sys_content_ids = tokenizer.encode(NUSCENES_SYSTEM_PROMPT, add_special_tokens=False)
        sys_part = [im_start_id] + system_ids + [nl_id] + sys_content_ids + [im_end_id, nl_id]

        user_start_part = [im_start_id] + user_ids + [nl_id]

        user_end_assistant_start_part = [im_end_id, nl_id, im_start_id] + assistant_ids + [nl_id]

        input_ids_list = []
        attention_mask_list = []

        for i in range(b):
            points_str = [f"({pt[0]:+07.2f}, {pt[1]:+07.2f})" for pt in hist_traj_np[i]]
            hist_traj_str = f"[PT_HIST, {', '.join(points_str)}]"

            user_prompt_text = NUSCENES_USER_PROMPT_TEMPLATE.format(
                nav_cmd=nav_cmd_texts[i],
                hist_traj_str=hist_traj_str,
            )
            user_content_ids = tokenizer.encode(user_prompt_text, add_special_tokens=False)

            full_ids = sys_part + user_start_part + user_content_ids + user_end_assistant_start_part

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long))

        tokenized_prompt = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        tokenized_prompt_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0).to(device)

        traj_answer_ids = None
        traj_answer_mask = None
        traj_labels = None
        if gt_trajs is not None:
            if self.traj_ar_target == "waypoint":
                gt_trajs_target = torch.cumsum(gt_trajs, dim=1)
            elif self.traj_ar_target == "norm_delta":
                gt_trajs_target = self.norm_delta(gt_trajs)
            else:
                gt_trajs_target = gt_trajs

            gt_trajs_np = gt_trajs_target.detach().cpu().numpy()
            answer_ids_list = []
            for i in range(b):
                pts = gt_trajs_np[i]
                pts_str = ", ".join(f"({p[0]:+07.2f}, {p[1]:+07.2f})" for p in pts)
                answer_text = f"[PT, {pts_str}]"
                answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
                answer_token_ids = answer_token_ids + [im_end_id, nl_id]
                answer_ids_list.append(torch.tensor(answer_token_ids, dtype=torch.long))

            traj_answer_ids = pad_sequence(
                answer_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)

            FIXED_ANSWER_MAX_LEN = 128
            curr_ans_len = traj_answer_ids.shape[1]
            if curr_ans_len < FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = F.pad(
                    traj_answer_ids, (0, FIXED_ANSWER_MAX_LEN - curr_ans_len),
                    value=tokenizer.pad_token_id
                )
            elif curr_ans_len > FIXED_ANSWER_MAX_LEN:
                traj_answer_ids = traj_answer_ids[:, :FIXED_ANSWER_MAX_LEN]

            traj_answer_mask = (traj_answer_ids != tokenizer.pad_token_id).long()
            traj_labels = traj_answer_ids.masked_fill(traj_answer_mask == 0, -100)

        return DrivingBatch(
            images=images,
            image_masks=image_masks,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            command=command,
            ego_status=ego_status,
            view_token_ids=view_token_ids,
            traj_answer_ids=traj_answer_ids,
            traj_answer_mask=traj_answer_mask,
            traj_labels=traj_labels,
        )


    def embed_prefix(self, batch: DrivingBatch):
        images_tensor = batch.images
        # --- Code Change ---
        # 从 self.action_in_proj（已删除）改为 self.det_proj_up。
        device = self.det_proj_up.weight.device
        # --- End Code Change ---

        # --- Code Change ---
        # Reason: 6D tensor (B,T,V,3,H,W) 走 video_processor + Conv3D，5D 走原 image_processor
        if images_tensor.dim() == 6:
            image_outputs = self.qwen3_vl_with_expert.embed_video_tensor(
                images_tensor,
                return_omega_global_tokens=self.vggt_omega_perception_enabled,
            )
        else:
            if self.vggt_omega_perception_enabled:
                raise RuntimeError(
                    "VGGT-Omega perception fusion in v2 requires the 6D video input path."
                )
            image_outputs = self.qwen3_vl_with_expert.embed_image_tensor(images_tensor)
        if self.vggt_omega_perception_enabled:
            image_features, feature_lens, all_image_grids, deepstack_features, raw_features, omega_global_tokens = image_outputs
        else:
            image_features, feature_lens, all_image_grids, deepstack_features, raw_features = image_outputs
            omega_global_tokens = None
        # --- End Code Change ---

        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer
        vision_start_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_start_token_id
        vision_end_id = self.qwen3_vl_with_expert.qwen3_vl.config.vision_end_token_id
        image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
        nl_id = self.qwen3_vl_with_expert.processor.tokenizer.encode("\n", add_special_tokens=False)[0]

        bs = images_tensor.shape[0]
        num_views_per_sample = 6
        view_token_ids = self._get_view_token_ids(device)

        prefix_input_ids_list = []

        # FIXED_PREFIX_MAX_LEN = 3740
        # --- Code Change ---
        # Reason: 视频 3024 img tokens + wrapper + prompt ~4560-5060 >> 3740，必须增大避免每个样本截断
        FIXED_PREFIX_MAX_LEN = 6144
        # --- End Code Change ---

        for b_idx in range(bs):
            sample_input_ids = []

            for v_idx in range(num_views_per_sample):
                img_len = feature_lens[b_idx * num_views_per_sample + v_idx]
                ids = [view_token_ids[v_idx].item(), nl_id, vision_start_id] + \
                    [image_token_id] * img_len + \
                    [vision_end_id, nl_id]
                sample_input_ids.extend(ids)

            prompt_mask = batch.tokenized_prompt_mask[b_idx].bool()
            prompt_ids = batch.tokenized_prompt[b_idx][prompt_mask].tolist()
            sample_input_ids.extend(prompt_ids)

            prefix_input_ids_list.append(torch.tensor(sample_input_ids, dtype=torch.long, device=device))

        prompt_lens_list = [ids.shape[0] for ids in prefix_input_ids_list]

        FIXED_TOTAL_LEN = FIXED_PREFIX_MAX_LEN + (
            batch.traj_answer_ids.shape[1] if batch.traj_answer_ids is not None else 0
        )

        if batch.traj_answer_ids is not None:
            full_ids_list = []
            for b_idx in range(bs):
                p_ids = prefix_input_ids_list[b_idx]
                a_ids = batch.traj_answer_ids[b_idx]
                if p_ids.shape[0] > FIXED_PREFIX_MAX_LEN:
                    p_ids = p_ids[:FIXED_PREFIX_MAX_LEN]
                    prompt_lens_list[b_idx] = FIXED_PREFIX_MAX_LEN
                full_ids_list.append(torch.cat([p_ids, a_ids], dim=0))

            prefix_input_ids = pad_sequence(
                full_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            curr_len = prefix_input_ids.shape[1]
            if curr_len < FIXED_TOTAL_LEN:
                prefix_input_ids = F.pad(prefix_input_ids, (0, FIXED_TOTAL_LEN - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > FIXED_TOTAL_LEN:
                prefix_input_ids = prefix_input_ids[:, :FIXED_TOTAL_LEN]
        else:
            prefix_input_ids = pad_sequence(prefix_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            curr_len = prefix_input_ids.shape[1]
            if curr_len < FIXED_PREFIX_MAX_LEN:
                prefix_input_ids = F.pad(prefix_input_ids, (0, FIXED_PREFIX_MAX_LEN - curr_len), value=tokenizer.pad_token_id)
            elif curr_len > FIXED_PREFIX_MAX_LEN:
                prefix_input_ids = prefix_input_ids[:, :FIXED_PREFIX_MAX_LEN]
                truncated_mask = (prefix_input_ids == image_token_id)
                valid_img_tokens = truncated_mask.sum().item()
                if valid_img_tokens < image_features.shape[0]:
                    image_features = image_features[:valid_img_tokens]

        prompt_only_len = max(prompt_lens_list)

        prefix_pad_masks = (prefix_input_ids != tokenizer.pad_token_id)

        input_embeds = self.qwen3_vl_with_expert.qwen3_vl.get_input_embeddings()(prefix_input_ids)

        image_mask = (prefix_input_ids == image_token_id)

        if image_mask.sum() != image_features.shape[0]:
            target_count = image_mask.sum()
            current_count = image_features.shape[0]
            if current_count > target_count:
                image_features = image_features[:target_count]
            else:
                raise ValueError(f"Visual features mismatch! Feat: {current_count}, Tokens: {target_count}")

        input_embeds = input_embeds.masked_scatter(image_mask.unsqueeze(-1), image_features.to(input_embeds.dtype))

        prefix_att_marks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
        for b_idx in range(bs):
            p_len = prompt_lens_list[b_idx]
            prefix_att_marks[b_idx, :p_len] = 1
        if batch.traj_answer_ids is not None:
            for b_idx in range(bs):
                p_len = prompt_lens_list[b_idx]
                real_ans_len = batch.traj_answer_mask[b_idx].sum().item()
                prefix_att_marks[b_idx, p_len:p_len + real_ans_len] = 1

        batch.prompt_lens = prompt_lens_list

        return input_embeds, prefix_pad_masks, prefix_att_marks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len, omega_global_tokens

    # --- Code Change ---
    # 删除 _maybe_get_status_features() 和 embed_suffix() —— 仅用于
    # action expert 的 suffix token 构建。
    # --- End Code Change ---

    def get_position_ids(self, input_ids, image_grid_thw, pad_masks):
        attention_mask = pad_masks.long()
        position_ids, rope_deltas = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def prepare_for_deformable_aggregation(self, feature_maps):
        if not feature_maps:
            return []
        return feature_maps_format(feature_maps)

    def forward_train(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        gt_occ_dense=None,
        hist_traj=None,
        ar_batch=None,
        **kwargs,
    ):
        if gt_ego_fut_trajs is None:
            raise ValueError("gt_ego_fut_trajs is required")

        # permute_indices = [0, 2, 1, 4, 5, 3]
        # --- Code Change ---
        # Reason: 12 相机时按 patch-major 分两组，各组内部套用原 6 相机 sensor 重排，
        # 与 embed_video_tensor 的 patch-major raw 拼接顺序严格对齐。
        base_perm = [0, 2, 1, 4, 5, 3]
        if getattr(self, "num_perception_views", 6) == 12:
            permute_indices = base_perm + [i + 6 for i in base_perm]  # 前6=patch0, 后6=patch1
        else:
            permute_indices = base_perm
        # --- End Code Change ---
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        _gt_trajs_raw = gt_ego_fut_trajs
        if torch.is_tensor(_gt_trajs_raw) and _gt_trajs_raw.dim() == 4 and _gt_trajs_raw.shape[1] == 1:
            _gt_trajs_raw = _gt_trajs_raw.squeeze(1)

        # --- Code Change ---
        # Reason: 视频模式下 DrivingBatch 用当前帧（img[:, -1]）构建 prompt，
        # 全量 6D tensor 挂到 batch.images 时同步做相机重排（与 _build_driving_batch 内部一致）。
        # 用 6 元素字面量: batch.images 的 V 维恒为 6 物理相机，不可复用被扩成 12 的 permute_indices。
        if img.dim() == 6:
            batch = self._build_driving_batch(
                img=img[:, -1],  # current-last: T=-1 = 当前帧
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
                hist_traj=hist_traj,
                gt_trajs=_gt_trajs_raw if self.enable_traj_ar else None,
            )
            batch.images = img[:, :, [0, 2, 1, 4, 5, 3]]
        else:
            batch = self._build_driving_batch(
                img=img,
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
                hist_traj=hist_traj,
                gt_trajs=_gt_trajs_raw if self.enable_traj_ar else None,
            )
        # --- End Code Change ---

        if isinstance(gt_agent_fut_trajs, list):
            gt_agent_fut_trajs = [t.to(device=img.device) if torch.is_tensor(t) else torch.tensor(t, device=img.device) for t in gt_agent_fut_trajs]
        if isinstance(gt_agent_fut_masks, list):
            gt_agent_fut_masks = [t.to(device=img.device) if torch.is_tensor(t) else torch.tensor(t, device=img.device) for t in gt_agent_fut_masks]

        # --- Code Change ---
        # 删除所有 noise/time/flow-matching 计算和 suffix embedding。
        # Action expert 已完全删除 —— 仅保留 VLM + perception expert。
        # --- End Code Change ---

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len, omega_global_tokens = self.embed_prefix(batch)

        bsz = img.shape[0] if img is not None else 1

        if self.driving_deepstack and deepstack_features is not None:
            image_token_id = self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id
            _visual_pos_masks = (prefix_input_ids == image_token_id)
            _ds_embeds = deepstack_features
        else:
            _visual_pos_masks = None
            _ds_embeds = None

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
        )
        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_device = prefix_embs.device
        head_param_dtype = next(self.unified_decoder.parameters()).dtype
        # --- Code Change ---
        # Reason: 时序视频下 timestamp 被 TemporalFlattenTransform 展开为 T*6 列表。
        # mmcv collate (zip-transpose) 将 B 个样本的 T*6 列表转为 T*6 个 (B,) tensor。
        # 三种格式: (a) 嵌套列表 (b) list of tensors (c) 扁平标量列表 B=1。
        # InstanceBank 期望 (B,) 形状的 1D tensor。
        if isinstance(timestamp, (list, tuple)):
            if len(timestamp) > 0 and isinstance(timestamp[0], (list, tuple)):
                # (a) 嵌套列表: 每样本列表取最后一位
                ts4perception = torch.tensor([float(t[-1]) for t in timestamp], dtype=torch.float32, device=head_device)
            elif len(timestamp) > 0 and torch.is_tensor(timestamp[0]):
                # (b) collate 转置后 list of (B,) tensors: [-1] 即当前帧，(B,) 形状
                ts4perception = timestamp[-1].clone().detach().float().to(head_device)
            else:
                # (c) 扁平标量列表 B=1: 取最后一位
                ts4perception = torch.tensor([float(timestamp[-1])], dtype=torch.float32, device=head_device)
        elif torch.is_tensor(timestamp):
            # 2D (B,T*6) / 1D (T*6,) / 0D 标量
            if timestamp.dim() >= 2:
                ts4perception = timestamp[:, -1].clone().detach().float().to(head_device)
            elif timestamp.dim() == 1:
                ts4perception = timestamp[-1].clone().detach().float().unsqueeze(0).to(head_device)
            else:
                ts4perception = timestamp.clone().detach().float().view(1).to(head_device)
        else:
            ts4perception = timestamp
        # --- End Code Change ---
        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': ts4perception,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_param_dtype),
            'image_wh': image_wh.to(device=head_device, dtype=head_param_dtype),
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)
        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, prefix_embs.device, prefix_embs.dtype, stage1_outs
        )
        perception_embs = self.fuse_perception_with_vggt_omega(
            perception_embs, omega_global_tokens
        )

        if self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            perception_embs = perception_embs.to(dtype=torch.bfloat16)

        if self.vlm_grad_scale != 1.0:
            s = self.vlm_grad_scale
            prefix_embs = prefix_embs * 1.0
            prefix_embs.register_hook(lambda g: g * s)
            perception_embs = perception_embs * 1.0
            perception_embs.register_hook(lambda g: g * s)

        prefix_pos_ids, rope_deltas = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        perception_len = perception_embs.shape[1]
        perception_range = torch.arange(1, perception_len + 1, device=prefix_embs.device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        if perception_len > 0:
            max_perception_pos = perception_pos_ids_1d.max(dim=-1, keepdim=True).values
        else:
            max_perception_pos = max_prefix_pos
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        # --- Code Change ---
        # suffix_len 始终为 0（无 action expert）。
        suffix_len = 0
        position_ids = torch.cat([prefix_pos_ids, perception_pos_ids_3d], dim=2)
        # --- End Code Change ---

        prefix_len = prefix_embs.shape[1]
        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        occ_len = perception_lengths['occ']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        att_mask_input = None
        q_len_rounded = None

        if self.attn_implementation == "flex":
            _prompt_only_len_key = prompt_only_len if self.enable_traj_ar else -1
            _bm_key = (bsz, prefix_len, perception_len, 0, _prompt_only_len_key)
            if self._cached_block_mask is None or self._cached_block_mask_key != _bm_key:
                block_mask, q_len_rounded = build_blockmask_unidrive(
                    bsz=bsz,
                    hq=self.qwen3_vl_with_expert.qwen3_vl.config.text_config.num_attention_heads,
                    prefix_len=prefix_len,
                    perception_len=perception_len,
                    suffix_len=0,
                    device=prefix_embs.device,
                    compile_blockmask=True,
                    prompt_only_len=_prompt_only_len_key,
                )
                self._cached_block_mask = block_mask
                self._cached_q_len_rounded = q_len_rounded
                self._cached_block_mask_key = _bm_key
            block_mask = self._cached_block_mask
            q_len_rounded = self._cached_q_len_rounded
            att_mask_input = block_mask
        else:
            pad_masks = torch.cat([prefix_pad_masks, perception_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, perception_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            if batch.traj_answer_ids is not None:
                for b_idx in range(bsz):
                    p_len = batch.prompt_lens[b_idx]
                    real_ans_len = int(batch.traj_answer_mask[b_idx].sum().item())
                    att_2d_masks[b_idx, prefix_len:, p_len:p_len + real_ans_len] = False
            att_mask_input = self._prepare_attention_masks_4d(att_2d_masks)

        stats = {}
        perception_out = None

        # --- Code Change ---
        # 联合 forward：仅 VLM + perception expert（无 action expert）。
        # Knowledge insulation 路径已完全删除。
        def forward_func(prefix_embs, perception_embs, att_mask, position_ids, _unused_return_middle_layers, q_len_rnd, ds_embeds, vis_pos_masks):
            outputs, _, middle_layer_outputs = self.qwen3_vl_with_expert.forward(
                attention_mask=att_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, perception_embs],
                use_cache=False,
                return_middle_layers=None,
                q_len_rounded=q_len_rnd,
                deepstack_visual_embeds=ds_embeds,
                visual_pos_masks=vis_pos_masks,
            )
            return outputs, middle_layer_outputs

        (outputs_embeds, _middle_layer_outs_unused) = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            perception_embs,
            att_mask_input,
            position_ids,
            None,
            q_len_rounded,
            _ds_embeds,
            _visual_pos_masks,
        )
        prefix_out, perception_out = outputs_embeds
        # --- End Code Change ---

        loss_traj_ar = prefix_embs.sum() * 0.0
        if self.enable_traj_ar and self.train_vlm and batch.traj_labels is not None:
            lm_head = self.qwen3_vl_with_expert.qwen3_vl.lm_head
            answer_len = batch.traj_labels.shape[1]
            loss_sum = prefix_embs.sum() * 0.0
            for i in range(bsz):
                p_len = batch.prompt_lens[i]
                pred_hidden_i = prefix_out[i, p_len - 1: p_len - 1 + answer_len, :]
                pred_logits_i = lm_head(pred_hidden_i.to(lm_head.weight.dtype))
                loss_sum = loss_sum + F.cross_entropy(
                    pred_logits_i,
                    batch.traj_labels[i],
                    ignore_index=-100,
                )
            loss_traj_ar = (loss_sum / bsz) * self.traj_ar_loss_weight

        loss_depth_val = self.compute_depth_loss(
            prefix_embs, prefix_out, prefix_input_ids, all_image_grids, gt_depth, focal, bsz
        )

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        o0 = m1
        o1 = o0 + occ_len
        e0 = o1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]
            # --- Claude Code ---
            # Reason: with_occ=False 时跳过 occ decoder 前向（已知上游双重计算问题，with_occ=False 时两处都跳过）
            # Original: occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            if self.with_occ and self.occ_decoder is not None:
                occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            else:
                occ_logits = None
            # --- Claude Code ---

            target_dtype = self.det_proj.weight.dtype
            proj_dtype = self.det_proj.weight.dtype
            motion_token_256 = stage1_outs.get('motion_token', None)
            ego_feat_stage1 = stage1_outs['ego_instance_feature']

            det_feat_fused = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            map_feat_fused = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            ego_feat_fused = (
                self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if ego_out is not None
                else ego_feat_stage1.to(torch.float32)
            )
            motion_feat_fused = (
                self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                if motion_token_256 is not None
                else None
            )

            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)

            gt_map_labels_val = gt_map_labels.data if hasattr(gt_map_labels, 'data') else gt_map_labels
            gt_map_pts_val = gt_map_pts.data if hasattr(gt_map_pts, 'data') else gt_map_pts
            gt_boxes_list = None
            gt_labels_list = None
            # --- Code Change ---
            # 使用 prefix_embs.device 替代 actions.device（actions 已随 action expert 删除）。
            # --- End Code Change ---
            if gt_bboxes_3d is not None:
                gt_boxes_list = gt_bboxes_3d.data if hasattr(gt_bboxes_3d, 'data') else gt_bboxes_3d
                gt_boxes_list = [x.tensor.to(prefix_embs.device) if hasattr(x, 'tensor') else x.to(prefix_embs.device) for x in gt_boxes_list]
            if gt_labels_3d is not None:
                gt_labels_list = gt_labels_3d.data if hasattr(gt_labels_3d, 'data') else gt_labels_3d
                gt_labels_list = [x.to(prefix_embs.device) for x in gt_labels_list]

            perception_data = {
                'gt_bboxes_3d': gt_boxes_list,
                'gt_labels_3d': gt_labels_list,
                'gt_map_labels': gt_map_labels_val,
                'gt_map_pts': gt_map_pts_val,
                'ego_status': ego_status,
                'gt_agent_fut_trajs': gt_agent_fut_trajs,
                'gt_agent_fut_masks': gt_agent_fut_masks,
            }
            perception_losses = self.unified_decoder.loss(stage1_outs, stage2_outs, perception_data)
            stats.update(perception_losses)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=prefix_embs.device, dtype=torch.float32)
            ego_out = None

        # --- Code Change ---
        # Action-only loss branches are intentionally absent.
        # --- End Code Change ---


        # --- Claude Code ---
        # Reason: with_occ=False 时跳过 occ loss 计算
        # Original: if gt_occ_dense is not None and perception_out is not None:
        if self.with_occ and gt_occ_dense is not None and perception_out is not None:
            occ_target = gt_occ_dense.permute(0, 3, 1, 2).long()
            occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            loss_occ = F.cross_entropy(occ_logits, occ_target)
        else:
            loss_occ = perception_embs.sum() * 0.0
        # --- Claude Code ---
        stats["loss_occ"] = loss_occ

        loss_motion = stats.get("loss_motion", perception_embs.sum() * 0.0)

        ar_loss_total = torch.tensor(0.0, device=prefix_embs.device)
        if self.train_vlm:
            if ar_batch is not None:
                ar_outputs = self.forward_ar_batch(ar_batch)
                ar_loss_total = ar_outputs['loss_ar']
                stats['loss_vlm_raw'] = ar_outputs['loss_vlm_raw']
                stats['ar_perception_ratio'] = 0.0
            else:
                stats['loss_vlm_raw'] = torch.tensor(0.0, device=prefix_embs.device)
                stats['ar_perception_ratio'] = 0.0

        stats["loss_occ"] = self.occ_loss_weight * loss_occ
        stats["loss_depth"] = loss_depth_val

        stats.setdefault("loss_motion", loss_motion)

        if self.train_vlm:
            stats["loss_ar"] = ar_loss_total

        if self.enable_traj_ar and self.train_vlm:
            stats["loss_traj_ar"] = loss_traj_ar

        stats.pop('loss_perception', None)
        stats.pop('loss', None)

        return {"losses": stats}

    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        timestamp=None,
        projection_mat=None,
        image_wh=None,
        gt_depth=None,
        focal=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_labels=None,
        gt_map_pts=None,
        gt_agent_fut_trajs=None,
        gt_agent_fut_masks=None,
        gt_ego_fut_trajs=None,
        gt_ego_fut_masks=None,
        gt_ego_fut_cmd=None,
        ego_status=None,
        num_steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        hist_traj=None,
        use_gt_ego_status: bool = False,
        **kwargs,
    ):
        # permute_indices = [0, 2, 1, 4, 5, 3]
        # --- Code Change ---
        # Reason: 12 相机时按 patch-major 分两组（与 forward_train 一致）
        base_perm = [0, 2, 1, 4, 5, 3]
        if getattr(self, "num_perception_views", 6) == 12:
            permute_indices = base_perm + [i + 6 for i in base_perm]
        else:
            permute_indices = base_perm
        # --- End Code Change ---
        if projection_mat is not None:
            projection_mat = projection_mat[:, permute_indices]
        if image_wh is not None:
            image_wh = image_wh[:, permute_indices]

        if "img_metas" in kwargs and kwargs.get("img_metas") is not None:
            kwargs["img_metas"] = permute_metas_per_camera_fields(
                kwargs.get("img_metas"), permute_indices, TARGET_SENSOR_ORDER
            )

        # --- Code Change ---
        # Reason: 视频模式下 DrivingBatch 用当前帧（img[:, -1]）构建 prompt，
        # 全量 6D 挂到 batch.images 时同步做相机重排（与 forward_train 一致）。
        # 避免 img[:, [0,2,1,4,5,3]] 在 T 维上报 IndexError（原代码无 6D 分支）
        if img.dim() == 6:
            batch = self._build_driving_batch(
                img=img[:, -1],
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
                hist_traj=hist_traj,
            )
            batch.images = img[:, :, [0, 2, 1, 4, 5, 3]]
        else:
            batch = self._build_driving_batch(
                img=img,
                command=gt_ego_fut_cmd,
                ego_status=ego_status,
                hist_traj=hist_traj,
            )
        # --- End Code Change ---
        bsz = batch.tokenized_prompt.shape[0]
        device = batch.tokenized_prompt.device
        dtype = self.qwen3_vl_with_expert.qwen3_vl.language_model.layers[0].self_attn.q_proj.weight.dtype

        # --- Code Change ---
        # 删除噪声采样 —— ODE 去噪循环已随 action expert 删除。
        # --- End Code Change ---

        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        prefix_embs, prefix_pad_masks, prefix_att_masks, all_image_grids, prefix_input_ids, deepstack_features, raw_features, prompt_only_len, omega_global_tokens = self.embed_prefix(batch)

        if prefix_embs.dtype != dtype:
            prefix_embs = prefix_embs.to(dtype)

        source_features = raw_features if self.feature_source == "raw" else deepstack_features
        feature_maps = self.project_and_reshape_features(
            source_features, bsz, all_image_grids, self.feature_source
        )

        feature_maps_daf = self.prepare_for_deformable_aggregation(feature_maps)
        if not self.feat_grad:
            feature_maps_daf = [x.detach() for x in feature_maps_daf]

        head_dtype = next(self.unified_decoder.parameters()).dtype
        head_device = device

        # --- Code Change ---
        # Reason: 时序视频下 timestamp 被 TemporalFlattenTransform 展开为 T*6 列表。
        # mmcv collate (zip-transpose) 将 B 个样本的 T*6 列表转为 T*6 个 (B,) tensor。
        # 三种格式: (a) 嵌套列表 (b) list of tensors (c) 扁平标量列表 B=1。
        # InstanceBank 期望 (B,) 形状的 1D tensor。
        if isinstance(timestamp, (list, tuple)):
            if len(timestamp) > 0 and isinstance(timestamp[0], (list, tuple)):
                # (a) 嵌套列表: 每样本列表取最后一位
                ts4perception = torch.tensor([float(t[-1]) for t in timestamp], dtype=torch.float32, device=head_device)
            elif len(timestamp) > 0 and torch.is_tensor(timestamp[0]):
                # (b) collate 转置后 list of (B,) tensors: [-1] 即当前帧，(B,) 形状
                ts4perception = timestamp[-1].clone().detach().float().to(head_device)
            else:
                # (c) 扁平标量列表 B=1: 取最后一位
                ts4perception = torch.tensor([float(timestamp[-1])], dtype=torch.float32, device=head_device)
        elif torch.is_tensor(timestamp):
            # 2D (B,T*6) / 1D (T*6,) / 0D 标量
            if timestamp.dim() >= 2:
                ts4perception = timestamp[:, -1].clone().detach().float().to(head_device)
            elif timestamp.dim() == 1:
                ts4perception = timestamp[-1].clone().detach().float().unsqueeze(0).to(head_device)
            else:
                ts4perception = timestamp.clone().detach().float().view(1).to(head_device)
        else:
            ts4perception = timestamp
        # --- End Code Change ---

        perception_metas = {
            'img_metas': kwargs.get('img_metas'),
            'timestamp': ts4perception,
            'projection_mat': projection_mat.to(device=head_device, dtype=head_dtype) if projection_mat is not None else None,
            'image_wh': image_wh.to(device=head_device, dtype=head_dtype) if image_wh is not None else None,
        }

        stage1_outs = self.unified_decoder.forward_stage1(feature_maps_daf, perception_metas)

        perception_embs, perception_pad_masks, perception_att_masks, perception_lengths = self.embed_perception(
            bsz, device, dtype, stage1_outs
        )
        perception_embs = self.fuse_perception_with_vggt_omega(
            perception_embs, omega_global_tokens
        )

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d).to(dtype)

        prefix_pos_ids, _ = self.get_position_ids(prefix_input_ids, all_image_grids, prefix_pad_masks)
        max_prefix_pos = prefix_pos_ids.max(dim=0).values.max(dim=-1, keepdim=True).values

        _ds_embeds = deepstack_features if self.driving_deepstack and deepstack_features is not None else None
        _vis_masks = (
            (prefix_input_ids == self.qwen3_vl_with_expert.qwen3_vl.config.image_token_id)
            if _ds_embeds is not None else None
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = self._inference_attn_impl

        # --- Code Change ---
        # 将 inputs_embeds 从 3 元素改为 2 元素，解包从 3 改为 2。
        (_, _), past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            deepstack_visual_embeds=_ds_embeds,
            visual_pos_masks=_vis_masks,
        )
        # --- End Code Change ---

        perception_len = perception_embs.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsz, perception_len, prefix_len)
        perception_att_2d = make_att_2d_masks(perception_pad_masks, perception_att_masks)
        perception_full_att_2d = torch.cat([prefix_pad_2d, perception_att_2d], dim=2)

        perception_full_att_2d_4d = self._prepare_attention_masks_4d(perception_full_att_2d).to(dtype)

        perception_range = torch.arange(1, perception_len + 1, device=device).view(1, -1).expand(bsz, -1)
        perception_pos_ids_1d = max_prefix_pos + perception_range
        perception_pos_ids_3d = torch.stack([perception_pos_ids_1d] * 3, dim=0)

        self.qwen3_vl_with_expert.qwen3_perception_expert.config._attn_implementation = self._inference_attn_impl

        # --- Code Change ---
        # 将 inputs_embeds 从 3 元素改为 2 元素，解包从 3 改为 2。
        (_, perception_out), past_key_values, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=perception_full_att_2d_4d,
            position_ids=perception_pos_ids_3d,
            past_key_values=past_key_values,
            inputs_embeds=[None, perception_embs],
            use_cache=True,
        )
        # --- End Code Change ---

        det_len = perception_lengths['det']
        map_len = perception_lengths['map']
        occ_len = perception_lengths['occ']
        ego_len = perception_lengths['ego']
        motion_len = perception_lengths['motion']

        d0 = 0
        d1 = d0 + det_len
        m0 = d1
        m1 = m0 + map_len
        o0 = m1
        o1 = o0 + occ_len
        e0 = o1
        e1 = e0 + ego_len
        t0 = e1
        t1 = t0 + motion_len

        stage2_outs = None

        if perception_out is not None:
            det_out_vlm = perception_out[:, d0:d1]
            map_out_vlm = perception_out[:, m0:m1]
            occ_out = perception_out[:, o0:o1]
            # --- Claude Code ---
            # Reason: with_occ=False 时跳过 occ decoder 前向
            # Original: occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            if self.with_occ and self.occ_decoder is not None:
                occ_logits = self.occ_decoder(occ_out.to(torch.float32))
            else:
                occ_logits = None
            # --- Claude Code ---
            ego_out = perception_out[:, e0:e1] if ego_len > 0 else None
            motion_out_vlm = perception_out[:, t0:t1]

            target_dtype = self.det_proj.weight.dtype
            proj_dtype = self.det_proj.weight.dtype
            motion_token_256 = stage1_outs.get('motion_token', None)
            ego_feat_stage1 = stage1_outs['ego_instance_feature']

            det_feat_fused = self.det_proj(det_out_vlm.to(proj_dtype)).to(torch.float32)
            map_feat_fused = self.map_proj(map_out_vlm.to(proj_dtype)).to(torch.float32)
            ego_feat_fused = (
                self.ego_proj_down(ego_out.to(proj_dtype)).to(torch.float32)
                if ego_out is not None
                else ego_feat_stage1.to(torch.float32)
            )
            motion_feat_fused = (
                self.motion_proj_down(motion_out_vlm.to(proj_dtype)).to(torch.float32)
                if motion_token_256 is not None
                else None
            )

            vlm_enhanced = {
                'det_feat': det_feat_fused.to(target_dtype),
                'map_feat': map_feat_fused.to(target_dtype),
                'ego_feat': ego_feat_fused.to(target_dtype),
            }
            if motion_feat_fused is not None:
                vlm_enhanced['motion_feat'] = motion_feat_fused.to(target_dtype)

            stage2_outs = self.unified_decoder.forward_stage2(vlm_enhanced, feature_maps_daf, perception_metas)
            det_result, map_result = self.unified_decoder.post_process(stage2_outs)
        else:
            occ_out = torch.zeros((bsz, occ_len, self.vlm_hidden_size), device=device, dtype=torch.float32)
            occ_logits = None  #0630新加：perception_out 为空时 occ 也置空
            ego_out = None
            det_result = None
            map_result = None

        # --- Code Change ---
        # 删除 ODE 去噪循环和 _denoise_step —— action expert 已删除。
        # forward_test 现在对轨迹预测返回 None。
        # --- End Code Change ---

        return {
            "traj": None,
            "det": det_result,
            "map": map_result,
            "occ": occ_logits,
        }

    def forward_ar_batch(self, ar_batch):
        input_ids = ar_batch['ar_input_ids']
        labels    = ar_batch['ar_labels']
        device    = input_ids.device
        tokenizer = self.qwen3_vl_with_expert.processor.tokenizer

        raw_pv = ar_batch.get('ar_pixel_values', None)
        pixel_values_tensor = None
        if raw_pv is not None:
            flat = []
            for item in raw_pv:
                if isinstance(item, (list, tuple)):
                    for t in item:
                        flat.append(t.to(device))
                else:
                    flat.append(item.to(device))
            if flat:
                pixel_values_tensor = torch.cat(flat, dim=0)

        raw_thw = ar_batch.get('ar_image_grid_thw', None)
        flat_image_grid_thw = None
        if raw_thw is not None:
            if isinstance(raw_thw, (list, tuple)):
                parts = [t.to(device) for t in raw_thw if t is not None]
                if parts:
                    flat_image_grid_thw = torch.cat(parts, dim=0)
            elif torch.is_tensor(raw_thw):
                if raw_thw.dim() == 3:
                    flat_image_grid_thw = raw_thw.reshape(-1, 3).to(device)
                else:
                    flat_image_grid_thw = raw_thw.to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        position_ids, _ = self.qwen3_vl_with_expert.vlm_base.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=flat_image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.qwen3_vl_with_expert.qwen3_vl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_tensor,
                image_grid_thw=flat_image_grid_thw,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )

        self.qwen3_vl_with_expert.qwen3_vl.language_model.config._attn_implementation = "flash_attention_2"
        self.qwen3_vl_with_expert.qwen3_vl.visual.config._attn_implementation = "flash_attention_2"
        loss_vlm = outputs.loss

        return dict(
            loss_ar=self.ar_loss_weight * loss_vlm,
            loss_vlm_raw=loss_vlm.detach(),
        )
