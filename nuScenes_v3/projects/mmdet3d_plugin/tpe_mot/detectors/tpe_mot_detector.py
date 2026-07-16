import torch

from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class TPEMoTDetector(BaseDetector):
    """Video perception model with a VLM and sparse perception decoder."""

    def __init__(
        self,
        perception_head=None,
        task_loss_weight=dict(perception=1.0),
        **kwargs,
    ):
        super().__init__()
        self.perception_head = (
            build_head(perception_head) if perception_head is not None else None
        )
        self.task_loss_weight = task_loss_weight
        self.instance_bank = None

        video_input_cfg = kwargs.get("video_input", {})
        self.num_total_frames = video_input_cfg.get("num_total_frames", 1)

    @property
    def with_perception_head(self):
        return self.perception_head is not None

    def load_state_dict(self, state_dict, strict=True):
        """Load legacy v2 checkpoints after the public head rename.

        Existing checkpoints store the active head under ``planning_head.``.
        The mapping is deliberately one-way, so new TPE-MoT checkpoints only
        expose the perception-oriented module name.
        """
        migrated = state_dict.copy()
        legacy_prefix = "planning_head."
        current_prefix = "perception_head."
        for key, value in state_dict.items():
            if key.startswith(legacy_prefix):
                current_key = current_prefix + key[len(legacy_prefix):]
                migrated.setdefault(current_key, value)
                migrated.pop(key, None)
        return super().load_state_dict(migrated, strict=strict)

    def extract_feat(self, img):
        return None

    def simple_test(self, img, **kwargs):
        if not self.with_perception_head:
            raise RuntimeError("perception_head is required.")
        pred = self.perception_head.forward_test(img=img, **kwargs)
        return [{"perception": pred}]

    def aug_test(self, imgs, **kwargs):
        return self.simple_test(imgs[0], **kwargs)

    def forward(self, return_loss=True, ar_batch=None, **kwargs):
        if return_loss:
            if ar_batch is None:
                ar_batch = getattr(self, "_current_ar_batch", None)
            return self.forward_train(ar_batch=ar_batch, **kwargs)
        return self.forward_test(**kwargs)

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        factor = float(self.task_loss_weight.get(prefix, 1.0))
        return {"{}.{}".format(prefix, key): value * factor for key, value in loss_dict.items()}

    def _select_last_from_queue(self, img):
        # Video inputs remain [B, T, V, 3, H, W] when temporal fusion is on.
        # A single-frame configuration still selects the current frame.
        if torch.is_tensor(img) and img.dim() == 6 and self.num_total_frames <= 1:
            return img[:, -1]
        return img

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
        if not self.with_perception_head:
            raise RuntimeError("perception_head is required.")

        img_input = self._select_last_from_queue(img)
        ret = self.perception_head.forward_train(
            img=img_input,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_occ_dense=gt_occ_dense,
            hist_traj=hist_traj,
            ar_batch=ar_batch,
            **kwargs,
        )

        perception_losses = ret["losses"] if isinstance(ret, dict) and "losses" in ret else ret
        losses = self.loss_weighted_and_prefixed(perception_losses, prefix="perception")
        for key, value in list(losses.items()):
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=img.device if img is not None else "cuda")
            losses[key] = torch.nan_to_num(value)
        return losses

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
        gt_occ_dense=None,
        hist_traj=None,
        **kwargs,
    ):
        if not self.with_perception_head:
            raise RuntimeError("perception_head is required.")

        img_input = self._select_last_from_queue(img)
        pred = self.perception_head.forward_test(
            img=img_input,
            timestamp=timestamp,
            projection_mat=projection_mat,
            image_wh=image_wh,
            gt_depth=gt_depth,
            focal=focal,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_map_labels=gt_map_labels,
            gt_map_pts=gt_map_pts,
            gt_agent_fut_trajs=gt_agent_fut_trajs,
            gt_agent_fut_masks=gt_agent_fut_masks,
            gt_ego_fut_trajs=gt_ego_fut_trajs,
            gt_ego_fut_masks=gt_ego_fut_masks,
            gt_ego_fut_cmd=gt_ego_fut_cmd,
            ego_status=ego_status,
            gt_occ_dense=gt_occ_dense,
            hist_traj=hist_traj,
            **kwargs,
        )

        if not isinstance(pred, dict) or ("det" not in pred and "map" not in pred):
            payload = pred if isinstance(pred, dict) else {"perception": pred}
            return [{"img_bbox": payload}]

        det_list = pred.get("det")
        map_list = pred.get("map")
        if isinstance(det_list, list) and isinstance(map_list, list):
            batch_size = max(len(det_list), len(map_list))
        elif isinstance(det_list, list):
            batch_size = len(det_list)
        elif isinstance(map_list, list):
            batch_size = len(map_list)
        else:
            batch_size = 1

        outputs = []
        for index in range(int(batch_size)):
            img_bbox = {}
            if isinstance(det_list, list) and index < len(det_list) and isinstance(det_list[index], dict):
                img_bbox.update(det_list[index])
            if isinstance(map_list, list) and index < len(map_list) and isinstance(map_list[index], dict):
                img_bbox.update(map_list[index])
            outputs.append({"img_bbox": img_bbox})
        return outputs
