from collections import defaultdict
from pathlib import Path
import sys
from typing import List

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled
from contextlib import nullcontext


# This wrapper lives inside TPE-MoT. The VGGT-Omega model package
# remains a third-party dependency at the repository root.
REPO_ROOT = Path(__file__).resolve().parents[5]
OMEGA_ROOT = REPO_ROOT / "third_party" / "vggt-omega"
if OMEGA_ROOT.is_dir() and str(OMEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(OMEGA_ROOT))

from vggt_omega.models import VGGTOmega  # noqa: E402


class VGGTOmegaSpatialEncoderConfig(PretrainedConfig):
    model_type = "vggt_omega_spatial_encoder"
    base_config_key = "spatial_config"

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.enable_camera = enable_camera
        self.enable_depth = enable_depth
        self.enable_alignment = enable_alignment


class VGGTOmegaSpatialEncoderPreTrainedModel(PreTrainedModel):
    config_class = VGGTOmegaSpatialEncoderConfig
    base_model_prefix = "spatial_encoder"

    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vggt_omega_model = VGGTOmega(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            enable_camera=config.enable_camera,
            enable_depth=config.enable_depth,
            enable_alignment=config.enable_alignment,
        ).eval()

    def _init_weights(self, module):
        pass

    def load_pretrained_weights(self, pretrained_weight: str):
        if is_deepspeed_zero3_enabled():
            self.load_pretrained_weights_zero3(pretrained_weight)
        else:
            self._load_pretrained_weights(pretrained_weight)

    def load_pretrained_weights_zero3(self, pretrained_weight):
        if deepspeed is None:
            raise ImportError("DeepSpeed is required for ZeRO-3 VGGT-Omega weight loading.")
        with deepspeed.zero.GatheredParameters(
            list(self.vggt_omega_model.parameters()),
            modifier_rank=0,
        ):
            if deepspeed.comm.get_rank() == 0:
                self._load_pretrained_weights(pretrained_weight)

    def _load_pretrained_weights(self, pretrained_weight):
        print(f"Loading external VGGT-Omega weights from: {pretrained_weight}")
        state_dict = self._read_state_dict(pretrained_weight)

        name, p = next(self.vggt_omega_model.named_parameters())
        print(f"[VGGT-Omega before load] first param: {name}, mean={p.data.float().mean().item():.6f}")

        incompatible = self.vggt_omega_model.load_state_dict(state_dict, strict=False)
        missing_keys = incompatible.missing_keys
        unexpected_keys = incompatible.unexpected_keys

        name, p = next(self.vggt_omega_model.named_parameters())
        print(f"[VGGT-Omega after load] first param: {name}, mean={p.data.float().mean().item():.6f}")
        print("VGGT-Omega load missing keys:", missing_keys)
        print("VGGT-Omega load unexpected keys:", unexpected_keys)

        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "VGGT-Omega pretrained weights do not exactly match the spatial encoder: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )

    def _read_state_dict(self, pretrained_weight):
        weight_path = Path(pretrained_weight)
        if weight_path.suffix == ".safetensors":
            return load_file(str(weight_path), device="cpu")

        state_dict = torch.load(str(weight_path), map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module"):
                nested = state_dict.get(key)
                if isinstance(nested, dict):
                    return nested
        return state_dict

    def preprocess_video_tensors(self, video_tensor: List[torch.Tensor]) -> List[torch.Tensor]:
        return video_tensor

    def _aggregator_autocast_context(self, batch_input: torch.Tensor):
        if not batch_input.is_cuda:
            return nullcontext()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def forward(self, video_tensor: List[torch.Tensor], **kwargs):
        """
        video_tensor: List of [T_i, C, H_i, W_i].

        Returns the same external contract as the original VGGT spatial encoder:
          - final_outputs: List[List[Tensor]], one cached Omega layer list per sample.
          - final_indices: List[int], patch token start index per sample.

        VGGT-Omega's aggregator emits tensors shaped [B, T, N, 2 * embed_dim]
        for cached layers, where tokens before patch_token_start are camera and
        register tokens.
        """
        group_map = defaultdict(list)
        for original_idx, v in enumerate(video_tensor):
            group_map[v.shape].append((original_idx, v))

        final_outputs = [None] * len(video_tensor)
        final_indices = [None] * len(video_tensor)

        for (_T, _C, _H, _W), items in group_map.items():
            indices = [item[0] for item in items]
            tensors = [item[1] for item in items]

            batch_input = torch.stack(tensors)
            if not batch_input.is_cuda:
                batch_input = batch_input.float()
            with self._aggregator_autocast_context(batch_input):
                batch_out, patch_token_start = self.vggt_omega_model.aggregator(batch_input)
            batch_size = len(indices)

            for i in range(batch_size):
                real_idx = indices[i]
                sample_output = []
                for layer_tensor in batch_out:
                    sample_output.append(None if layer_tensor is None else layer_tensor[i])

                final_outputs[real_idx] = sample_output
                final_indices[real_idx] = patch_token_start

        return final_outputs, final_indices

    def forward_predictions(self, video_tensor: torch.Tensor):
        """
        Optional full VGGT-Omega inference path for camera/depth/text-alignment heads.
        The training connector path should use forward(), which exposes aggregator
        tokens instead of prediction dictionaries.
        """
        return self.vggt_omega_model(video_tensor)

    def print_trainable_parameters(self) -> None:
        is_spatial_encoder_trainable = any(param.requires_grad for param in self.parameters())
        print(f"VGGT-Omega Spatial Encoder Trainable: {is_spatial_encoder_trainable}")
