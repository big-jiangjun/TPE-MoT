"""Training-time audit for the frozen VGGT-Omega and its two connectors."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _sample_parameter(parameter, sample_size=256):
    flat = parameter.detach().reshape(-1)
    if flat.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    count = min(int(flat.numel()), sample_size)
    if count == 1:
        indices = torch.zeros(1, device=flat.device, dtype=torch.long)
    else:
        # Avoid torch.linspace here. For billion-scale tensors its floating
        # intermediate values can round above flat.numel() - 1 on CUDA.
        steps = torch.arange(count, device=flat.device, dtype=torch.long)
        indices = steps * (flat.numel() - 1) // (count - 1)
    return flat.index_select(0, indices).float().cpu().clone()


@HOOKS.register_module()
class OmegaFusionTrainingAuditHook(Hook):
    """Assert connector learning and frozen Omega behavior during a short run."""

    def __init__(self, interval=1, output_filename="omega_training_audit.json"):
        self.interval = int(interval)
        self.output_filename = output_filename
        self.targets = {}
        self.initial_samples = {}
        self.audit = {}

    def _rank(self):
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _resolve_targets(self, runner):
        model = _unwrap_model(runner.model)
        perception_head = getattr(model, "perception_head", None)
        if perception_head is None:
            perception_head = getattr(model, "pts_bbox_head", None)
        if perception_head is None:
            raise RuntimeError("VGGT audit could not find the perception head.")
        expert = getattr(perception_head, "qwen3_vl_with_expert", None)
        targets = {
            "prefix_connector": getattr(expert, "vggt_omega_connector", None),
            "perception_connector": getattr(perception_head, "vggt_omega_perception_connector", None),
            "spatial_encoder": getattr(expert, "vggt_omega_spatial_encoder", None),
        }
        if any(module is None for module in targets.values()):
            missing = [name for name, module in targets.items() if module is None]
            raise RuntimeError("VGGT audit missing modules: {}".format(", ".join(missing)))
        return targets

    @staticmethod
    def _parameter_count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    def before_run(self, runner):
        self.targets = self._resolve_targets(runner)
        spatial = self.targets["spatial_encoder"]
        if any(parameter.requires_grad for parameter in spatial.parameters()):
            raise RuntimeError("VGGT spatial encoder must be frozen for this smoke training run.")

        for name, module in self.targets.items():
            self.initial_samples[name] = [
                _sample_parameter(parameter)
                for parameter in module.parameters()
            ]
        self.audit = {
            "interval": self.interval,
            "modules": {
                name: {
                    "parameter_count": self._parameter_count(module),
                    "trainable_parameter_count": sum(
                        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
                    ),
                    "nonzero_gradient_steps": 0,
                    "gradient_present_steps": 0,
                    "spatial_gradient_present_steps": 0,
                }
                for name, module in self.targets.items()
            },
        }
        runner.logger.info("[OmegaTrainAudit] Captured initial connector samples.")

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval:
            return
        for name in ("prefix_connector", "perception_connector"):
            gradients = [
                parameter.grad for parameter in self.targets[name].parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if gradients:
                self.audit["modules"][name]["gradient_present_steps"] += 1
                if any(torch.count_nonzero(gradient).item() for gradient in gradients):
                    self.audit["modules"][name]["nonzero_gradient_steps"] += 1

        spatial_gradients = [
            parameter.grad for parameter in self.targets["spatial_encoder"].parameters()
            if parameter.grad is not None
        ]
        if spatial_gradients:
            self.audit["modules"]["spatial_encoder"]["spatial_gradient_present_steps"] += 1

    def after_run(self, runner):
        failures = []
        for name, module in self.targets.items():
            max_sample_delta = 0.0
            changed_samples = 0
            for initial, parameter in zip(self.initial_samples[name], module.parameters()):
                delta = (_sample_parameter(parameter) - initial).abs()
                if delta.numel():
                    max_sample_delta = max(max_sample_delta, float(delta.max().item()))
                    changed_samples += int(torch.count_nonzero(delta).item())
            self.audit["modules"][name].update(
                sample_max_abs_delta=max_sample_delta,
                changed_sample_values=changed_samples,
            )

        for name in ("prefix_connector", "perception_connector"):
            report = self.audit["modules"][name]
            if report["nonzero_gradient_steps"] == 0:
                failures.append(f"{name} never had a nonzero gradient")
            if report["sample_max_abs_delta"] == 0.0:
                failures.append(f"{name} sampled weights never changed")

        spatial = self.audit["modules"]["spatial_encoder"]
        if spatial["spatial_gradient_present_steps"] != 0:
            failures.append("spatial_encoder received gradients despite freeze_spatial_encoder=True")
        if spatial["sample_max_abs_delta"] != 0.0:
            failures.append("spatial_encoder sampled weights changed despite being frozen")

        self.audit["passed"] = not failures
        self.audit["failures"] = failures
        if self._rank() == 0:
            output_path = Path(runner.work_dir) / self.output_filename
            output_path.write_text(json.dumps(self.audit, indent=2), encoding="utf-8")
            runner.logger.info("[OmegaTrainAudit] %s", json.dumps(self.audit, sort_keys=True))
            runner.logger.info("[OmegaTrainAudit] Wrote %s", output_path)
        if failures:
            raise RuntimeError("VGGT-Omega training audit failed: " + "; ".join(failures))
