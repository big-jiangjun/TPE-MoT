import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            # --- Code Change ---
            # 抑制 np.round(nan) 产生的 "invalid value encountered in cast" 警告
            # NaN 值会被下方 mask 过滤，无害但在 numpy 1.26+ 中会产生噪声
            with np.errstate(invalid='ignore'):
                U = np.round(pts_2d[:, 0]).astype(np.int32)
                V = np.round(pts_2d[:, 1]).astype(np.int32)
            # --- End Code Change ---
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict


# --- Code Change ---
#多帧加载输出嵌套[[t0六图],[t1六图]...]，Resize 循环会遍历内层列表报错,这里将嵌套 4×6 图像、内参、外参、timestamp 展平为一维 24 长度列表，后续处理完成后再 reshape 回 4D 张量；
# Reason: 将嵌套 (T, 6) 结构平铺为 (T*6,) 供 ResizeCropFlipImage/DepthMapGenerator 遍历。
# 展开范围: img / img_shape / lidar2img / cam_intrinsic / timestamp。
# hist_T_global_inv 是 (T,) 层级（每帧一个 4x4），不参与平铺——Adaptor 中按帧索引 [1] 直接取值。
@PIPELINES.register_module()
class TemporalFlattenTransform:
    """将 (T, 6) 嵌套展平为 (T*6,) 列表。

    关键: lidar2img/cam_intrinsic 也平铺到 T*6 长度，
    因为下游 ResizeCropFlipImage 按 for i in range(len(imgs)) 遍历并索引 lidar2img[i]。
    Adaptor 中取最后 6 项（[-6:]）作为当前帧外参（current-last 约定）。
    单帧时 no-op。
    """
    def __call__(self, results):
        img = results.get("img")
        if not isinstance(img, list) or not isinstance(img[0], list):
            return results  # 已是平铺的单帧

        # 平铺 img
        flat_img = []
        for frame_imgs in img:
            flat_img.extend(frame_imgs)
        results["img"] = flat_img

        # 平铺 img_shape: 每帧各相机同 shape
        img_shape = results.get("img_shape")
        if isinstance(img_shape, list):
            flat_shapes = []
            for i, frame_imgs in enumerate(img):
                _shape = img_shape[i] if i < len(img_shape) else frame_imgs[0].shape
                for _ in frame_imgs:
                    flat_shapes.append(_shape)
            results["img_shape"] = flat_shapes

        # 平铺 lidar2img: 每帧 (6,4,4) → T*6 项
        lidar2img = results.get("lidar2img")
        if isinstance(lidar2img, list) and len(lidar2img) > 0 and isinstance(lidar2img[0], list):
            flat_l2i = []
            for frame_l2i in lidar2img:
                flat_l2i.extend(frame_l2i)
            results["lidar2img"] = flat_l2i

        # 平铺 cam_intrinsic
        cam_intrinsic = results.get("cam_intrinsic")
        if isinstance(cam_intrinsic, list) and len(cam_intrinsic) > 0 and isinstance(cam_intrinsic[0], list):
            flat_ci = []
            for frame_ci in cam_intrinsic:
                flat_ci.extend(frame_ci)
            results["cam_intrinsic"] = flat_ci

        # 展开 timestamp: [t-3, t-2, t-1, cur] (per-frame) → T*6 (per-image)
        timestamp = results.get("timestamp")
        if isinstance(timestamp, list) and len(timestamp) > 0 and not isinstance(timestamp[0], list):
            flat_ts = []
            for frame_imgs in img:
                flat_ts.extend([timestamp[0]] * len(frame_imgs))
                timestamp = timestamp[1:] if len(timestamp) > 1 else [timestamp[0]]
            results["timestamp"] = flat_ts

        return results
# --- End Code Change ---


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    # def __init(self):
    #     pass
    # --- Code Change ---
    # Reason: 修复 __init 拼写错误（少两个下划线），支持 num_total_frames + perception_multiframe 参数传入。
    # perception_multiframe 是路径 A-native 的必要开关——Adaptor 需要它来决定是否构建 12 相机外参。
    def __init__(self, num_total_frames=1, perception_multiframe=False):
        self.num_total_frames = num_total_frames
        self.perception_multiframe = perception_multiframe
    # --- End Code Change ---

    def __call__(self, input_dict):
        # --- Code Change ---
        # Reason: 多帧时将平铺的 (T*6, ...) 数据取最后 6 项（当前帧，current-last 约定）构建外参，
        # img reshape 为 (T, 6, 3, H, W)。单帧时保持原逻辑不变。
        if self.num_total_frames > 1:
            input_dict["projection_mat"] = np.float32(
                np.stack(input_dict["lidar2img"][-6:])
            )
            input_dict["image_wh"] = np.ascontiguousarray(
                np.array(input_dict["img_shape"][-6:], dtype=np.float32)[:, :2][:, ::-1]
            )
            if "cam_intrinsic" in input_dict:
                input_dict["cam_intrinsic"] = np.float32(
                    np.stack(input_dict["cam_intrinsic"][-6:])
                )
                input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        else:
        # --- End Code Change ---
            input_dict["projection_mat"] = np.float32(
                np.stack(input_dict["lidar2img"])
            )
            input_dict["image_wh"] = np.ascontiguousarray(
                np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )
            if "cam_intrinsic" in input_dict:
                input_dict["cam_intrinsic"] = np.float32(
                    np.stack(input_dict["cam_intrinsic"])
                )
                input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]

        # --- Code Change ---
        # Reason: 路径 A-native 需 12 相机外参（patch-major: 前6=历史时刻, 后6=当前时刻）。
        # patch1=fused(t-1,cur) 用当前帧 lidar2img（即 [-6:]）。
        # patch0=fused(t-3,t-2) 用 t-2 帧 lidar2img @ T_cur2hist（ego 位姿补偿到当前帧坐标系）。
        # 不平铺的 lidar2img 为 current-last: [t-3(6), t-2(6), t-1(6), cur(6)]，t-2 帧 = [6:12]。
        if self.num_total_frames > 1 and getattr(self, "perception_multiframe", False):
            l2i = input_dict["lidar2img"]                       # list 长度 T*6 = 24
            cur_l2i = np.stack(l2i[-6:])                        # patch1 外参（当前帧）
            hist_l2i = np.stack(l2i[6:12])                      # t-2 帧 lidar2img
            # ego 位姿补偿: 把 anchor(当前帧系) 投到历史相机坐标系。
            # hist_T_global_inv[1] = inv(lidar2global of t-2) = T_world2lidar_hist
            # lidar2global = T_lidar2world of cur
            # T_cur2hist = T_world2lidar_hist @ T_lidar2world_cur
            T_cur2hist = input_dict["hist_T_global_inv"][1] @ input_dict["lidar2global"]
            hist_proj = hist_l2i @ T_cur2hist[None]             # (6,4,4)
            input_dict["projection_mat"] = np.float32(
                np.concatenate([hist_proj, cur_l2i], axis=0))   # (12,4,4) patch-major
            wh_cur = np.array(input_dict["img_shape"][-6:], dtype=np.float32)[:, :2][:, ::-1]
            input_dict["image_wh"] = np.ascontiguousarray(
                np.concatenate([wh_cur, wh_cur], axis=0))       # (12,2) 两组同分辨率
        # --- End Code Change ---

        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        # --- Code Change ---
        # Reason: 多帧时将平铺的 (T*6, 3, H, W) reshape 为 (T, 6, 3, H, W)，T=0 最旧、T=-1 当前。
        # 单帧时保持 (6, 3, H, W)。
        if self.num_total_frames > 1:
            imgs = imgs.reshape(self.num_total_frames, 6, 3,
                                imgs.shape[-2], imgs.shape[-1])
        # --- End Code Change ---
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        if "gt_depth" in input_dict:
            # input_dict["gt_depth"] is a list of numpy arrays (one per scale)
            # We take the first scale (highest resolution or first specified downsample)
            # and convert it to a Tensor wrapped in DataContainer.
            # Shape: (N_views, H, W)
            # --- Code Change ---
            # Reason: 多帧时取最后 6 张深度图（当前帧），避免 depth loss shape 不匹配
            if self.num_total_frames > 1:
                input_dict["gt_depth"] = DC(to_tensor(input_dict["gt_depth"][0][-6:]).float(), stack=True)
            else:
                input_dict["gt_depth"] = DC(to_tensor(input_dict["gt_depth"][0]).float(), stack=True)
            # --- End Code Change ---

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
