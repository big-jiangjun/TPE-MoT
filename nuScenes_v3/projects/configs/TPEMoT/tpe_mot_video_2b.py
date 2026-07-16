_base_ = ["../_base_/default_runtime.py"]

import os
# ===== User Configuration =====
vlm_pretrained_path = os.environ.get("VLM_PRETRAINED_PATH", "/path/to/Qwen3-VL-2B-Instruct")
stage1_model_checkpoint = os.environ.get("TPE_MOT_STAGE1_CHECKPOINT") or None
occworld_vae_path = os.environ.get("OCCWORLD_VAE_PATH", "/path/to/ckpt/occvae_latest.pth")
vggt_omega_path = os.environ.get(
    "VGGT_OMEGA_PATH"
)
deepspeed_config = os.environ.get("DEEPSPEED_CONFIG", "/path/to/zero_configs/adam_zero1_bf16.json")
data_infos_root = os.environ.get("DATA_INFOS_ROOT", "data/infos")
num_gpus = int(os.environ.get("NUM_GPUS", 8))  #婧愪唬鐮佽繖閲屾槸32锛屾牴鎹疄闄呴厤缃幇鍦ㄦ敼涓?锛屼笉鐒朵細鎻愬墠瑙﹀彂鏈€澶ц凯浠ｄ笂闄愶紝璁粌鎻愬墠缁撴潫
# ==============================

# ====================================================================
#  Stage 1: Unified perception decoder (det + map + ego, no motion)
#  Train perception queries first, motion loss disabled.
# ====================================================================

# Update-2023-06-12:
# [Enhance] Update some freezing args of UniAD
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
total_batch_size = 128
batch_size = 8 # --- Code Change --- Phase 4: 瑙嗛妯″紡鏄惧瓨 ~23GB/sample锛宐atch=8 棰勮 ~55-60GB/鍗?num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 30
total_epochs = 30  #杩欓噷璁粌杞暟鏀逛负30杞?checkpoint_epoch_interval = 1

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# input_shape = (960, 544)
# --- Code Change ---
# Reason: 闄嶄綆鍒嗚鲸鐜囨帶鍒惰棰?token 鏁般€?72/1600鈮?.42锛屽悓姝ユ敼 resize_lim 閬垮厤 FOV 瑁佸壀
input_shape = (672, 384)
# --- End Code Change ---
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
map_class_names = [
    'ped_crossing',
    'divider',
    'boundary',
]

num_classes = len(class_names)
num_map_classes = len(map_class_names)
roi_size = (30, 60)

num_sample = 20
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_mode = 6
queue_length = 4

# --- Code Change ---
# Reason: 鏃跺簭瑙嗛杈撳叆鍙傛暟銆俼ueue_length=4 鏄?InstanceBank 鍘嗗彶缂撳瓨锛坉ecoder query锛夛紝
# 浠ヤ笅鏄?VLM 瑙嗚杈撳叆甯ф暟锛堝綋鍓嶅抚 + 鍘嗗彶甯э級锛屼袱鑰呰涔変笉鍚屻€?num_history_frames = 3
num_total_frames = 4
perception_multiframe = True  # 寮€鍚? 浼浉鏈烘椂绌鸿仛鍚?# --- End Code Change ---

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
num_single_frame_decoder_map = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = True
temporal_map = True
decouple_attn = True
decouple_attn_map = False
decouple_attn_motion = True
with_quality_estimation = True

# ego_feature_map_scale = (544 // 16, 960 // 16)
# --- Code Change ---
# Reason: 鍒嗚鲸鐜囨敼涓?672脳384 鍚庡悓姝ユ洿鏂?ego_feature_map_scale = (384 // 16, 672 // 16)  # (24, 42)  楠ㄥ共缃戠粶鍥哄畾 4 娆?2 鍊嶄笅閲囨牱锛屾€荤缉鏀?16 鍊?# --- End Code Change ---

single_frame_layer = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split',
    'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
]
temporal_frame_layer = [
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split',
    'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
]
unified_decoder_operation_order = single_frame_layer * num_single_frame_decoder + \
                                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

unified_decoder_cfg = dict(
    type="TPEMoTSparseDecoder",
    embed_dims=embed_dims,
    task_select=["det", "map", "ego"],
    query_select=["det", "map", "ego"],
    num_stage1_layers=3,
    num_stage2_layers=3,
    num_single_frame_decoder=1,
    cls_threshold_to_reg=0.05,
    decouple_attn=decouple_attn,
    use_vlm_in_stage2=True,
    operation_order=unified_decoder_operation_order,

    det_instance_bank=dict(
        type="InstanceBank",
        num_anchor=900,
        embed_dims=embed_dims,
        anchor="data/kmeans/kmeans_det_900.npy",
        anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
        num_temp_instances=600,
        confidence_decay=0.6,
        feat_grad=False,
    ),
    map_instance_bank=dict(
        type="InstanceBank",
        num_anchor=100,
        embed_dims=embed_dims,
        anchor="data/kmeans/kmeans_map_100.npy",
        anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
        num_temp_instances=0,
        confidence_decay=0.6,
        feat_grad=True,
    ),
    ego_instance_bank=dict(
        type="EgoInstanceBank",
        embed_dims=embed_dims,
        anchor_type='nus',
        num_temp_instances=1,
        feature_map_scale=ego_feature_map_scale,
    ),

    det_anchor_encoder=dict(
        type="SparseBox3DEncoder",
        vel_dims=3,
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    ),
    map_anchor_encoder=dict(
        type="SparsePoint3DEncoder",
        embed_dims=embed_dims,
        num_sample=num_sample,
    ),

    graph_model=dict(
        type="SeparateAttention",
        query_select=["det", "map", "ego"],
        separate_list=[["det"], ["map"]],
        decouple_list=[True, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    temp_graph_model=dict(
        type="TemporalSeparateAttention",
        query_select=["det", "map", "ego"],
        query_list=[["det"], ["map"], ["ego"]],
        key_list=[["det"], ["map"], ["det", "map"]],
        decouple_list=[True, False, False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    inter_graph_model=dict(
        type="InteractiveAttention",
        query_select=["det", "map", "ego"],
        query_list=[["ego"]],
        key_list=[["det", "map"]],
        decouple_list=[False],
        attn=[
            dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                 num_heads=num_groups, batch_first=True, dropout=drop_out),
        ],
    ),

    det_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=12,  # --- Code Change --- Phase 4: 6鈫?2 for 12 pseudo-cameras
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0], [-0.45, 0, 0],
                [0, 0.45, 0], [0, -0.45, 0],
                [0, 0, 0.45], [0, 0, -0.45],
            ],
        ),
    ),
    map_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=12,  # --- Code Change --- Phase 4: 6鈫?2 for 12 pseudo-cameras
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparsePoint3DKeyPointsGenerator",
            embed_dims=embed_dims,
            num_sample=num_sample,
            num_learnable_pts=3,
            fix_height=(0, 0.5, -0.5, 1, -1),
            ground_height=-1.84023,
        ),
    ),
    ego_deformable=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=12,  # --- Code Change --- Phase 4: 6鈫?2 for 12 pseudo-cameras
        attn_drop=0.15,
        use_deformable_func=use_deformable_func,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBox3DKeyPointsGenerator",
            num_learnable_pts=6,
            fix_scale=[
                [0, 0, 0],
                [0.45, 0, 0], [-0.45, 0, 0],
                [0, 0.45, 0], [0, -0.45, 0],
                [0, 0, 0.45], [0, 0, -0.45],
            ],
        ),
    ),

    ffn=dict(
        type="AsymmetricFFN",
        in_channels=embed_dims * 2,
        pre_norm=dict(type="LN"),
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 4,
        num_fcs=2,
        ffn_drop=drop_out,
        act_cfg=dict(type="ReLU", inplace=True),
    ),
    norm_layer=dict(type="LN", normalized_shape=embed_dims),

    det_refine_layer=dict(
        type="SparseBox3DRefinementModule",
        embed_dims=embed_dims,
        num_cls=num_classes,
        refine_yaw=True,
        with_quality_estimation=with_quality_estimation,
    ),
    map_refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=embed_dims,
        num_sample=num_sample,
        num_cls=num_map_classes,
    ),
    ego_refine_layer=dict(
        type="EgoStatusRefinementModule",
        embed_dims=embed_dims,
        status_dims=6,
    ),
    motion_refine_layer=dict(
        type="SparseMotionRefinementModule",
        embed_dims=embed_dims,
        fut_ts=fut_ts,
        fut_mode=fut_mode,
    ),

    det_sampler=dict(
        type="SparseBox3DTarget",
        num_dn_groups=0,
        num_temp_dn_groups=0,
        dn_noise_scale=[2.0] * 3 + [0.5] * 7,
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        cls_wise_reg_weights={
            9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        },
    ),
    map_sampler=dict(
        type="SparsePoint3DTarget",
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=1.0),
                reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
            ),
        ),
        num_cls=num_map_classes,
        num_sample=num_sample,
        roi_size=roi_size,
    ),
    motion_sampler=dict(type='SparseMotionTarget'),

    det_decoder=dict(type="SparseBox3DDecoder"),
    map_decoder=dict(type="SparsePoint3DDecoder"),

    loss_det_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0,
    ),
    loss_det_reg=dict(
        type="SparseBox3DLoss",
        loss_box=dict(type="L1Loss", loss_weight=0.25),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
        loss_yawness=dict(type="GaussianFocalLoss"),
        cls_allow_reverse=[5],
    ),
    loss_map_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
    ),
    loss_map_reg=dict(
        type="SparseLineLoss",
        loss_line=dict(type='LinesL1Loss', loss_weight=10.0, beta=0.01),
        num_sample=num_sample,
        roi_size=roi_size,
    ),
    loss_ego_status=dict(type="L1Loss", loss_weight=1.0),
    loss_motion_cls=dict(
        type="FocalLoss",
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=0.0,
    ),
    loss_motion_reg=dict(type="L1Loss", loss_weight=0.0),

    det_reg_weights=[2.0] * 3 + [1.0] * 7,
    map_reg_weights=[1.0] * 40,

    motion_anchor="data/kmeans/kmeans_motion_6.npy",
)

vlm_lr_mult = 0.5

lora_cfg = dict(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
)

model = dict(
    type='TPEMoTDetector',
    # --- Code Change ---
    # Reason: 鍦?TPE-MoT 椤跺眰浼犲叆 video_input锛屼緵 _select_last_from_queue + forward_train 浣跨敤銆?    # 鍛藉悕 video_input 鑰岄潪 temporal锛岄伩鍏嶄笌 config L79 鐨?temporal=True锛坉ecoder 鏃跺簭娉ㄦ剰鍔涳級娣锋穯銆?    video_input=dict(num_total_frames=num_total_frames),
    # --- End Code Change ---
    perception_head=dict(
        type='TPEMoTPerceptionHead',
        # --- Code Change ---
        # Reason: 浼犲叆 temporal_cfg 渚?forward_train 鎰熺煡瑙嗛妯″紡 + Phase 4 perception_multiframe 寮€鍏?        temporal_cfg=dict(num_total_frames=num_total_frames, perception_multiframe=perception_multiframe), #鎶婂叏灞€鏃跺簭寮€鍏充粠閰嶇疆涓嬪彂鍒?Detector銆丳lanningHead 涓や釜鏍稿績妯″瀷绫伙紝鍏ㄥ眬缁熶竴鎺у埗鍗曞抚 / 瑙嗛妯″紡銆?        # --- End Code Change ---
        pretrained_path=vlm_pretrained_path,
        dtype='bfloat16',
        train_vlm=True,
        with_depth_supervision=False,
        # --- Claude Code ---
        # Reason: 鍏抽棴 occ voxel 棰勬祴锛?25 tokens + OccLatentDecoder锛夛紝鍔犻€熻缁?        with_occ=False,
        # --- Claude Code ---
        depth_loss_weight=0.2,
        occworld_vae_config=dict(
            type='VAERes3D',
            encoder_cfg=dict(
                type='Encoder2D',
                ch=64,
                out_ch=64,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=128,
                double_z=False,
            ),
            decoder_cfg=dict(
                type='Decoder3D',
                ch=64,
                out_ch=128,
                ch_mult=(1, 2, 4, 8),
                num_res_blocks=2,
                attn_resolutions=(50,),
                dropout=0.0,
                resamp_with_conv=True,
                in_channels=128,
                resolution=200,
                z_channels=64,
                give_pre_end=False,
            ),
            num_classes=18,
            expansion=8,
            vqvae_cfg=None,
        ),
        occworld_vae_path=occworld_vae_path,
        feat_grad=True,
        feature_source="raw",
        unified_decoder_cfg=unified_decoder_cfg,
        det_vla_head_cfg=None,
        map_vla_head_cfg=None,
        lora_cfg=lora_cfg,
        driving_deepstack=True,
        vggt_omega_prefix_fusion_cfg=dict(
            enabled=True,
            pretrained_weight=vggt_omega_path,
            require_pretrained=True,
            freeze_spatial_encoder=True,
            fuse_deepstack_layer=0,
            image_resolution=256,
            patch_size=16,
            preprocess_mode="balanced",
            spatial_config=dict(
                img_size=256,
                patch_size=16,
                embed_dim=1024,
                enable_camera=True,
                enable_depth=True,
                enable_alignment=True,
            ),
            connector_config=dict(
                spatial_embeds_layer_idx=-1,
                num_heads=8,
                attention_dropout=0.0,
                mlp_ratio=4.0,
                bias=False,
            ),
        ),
        vggt_omega_perception_fusion_cfg=dict(
            enabled=True,
            pretrained_weight=vggt_omega_path,
            require_pretrained=True,
            freeze_spatial_encoder=True,
            image_resolution=256,
            patch_size=16,
            preprocess_mode="balanced",
            spatial_embeds_layer_idx=-1,
            omega_dim=2048,
            num_heads=8,
            attention_dropout=0.0,
            mlp_ratio=4.0,
            bias=False,
            residual_scale_init=1e-3,
            spatial_config=dict(
                img_size=256,
                patch_size=16,
                embed_dim=1024,
                enable_camera=True,
                enable_depth=True,
                enable_alignment=True,
            ),
        ),
        vlm_fusion_cfg=dict(type='direct'),
        feature_fusion_cfg=dict(type='none'),
    ),
    task_loss_weight=dict(perception=1.0),
)


dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
anno_root = "data/infos/"
info_root = "data/infos/"
file_client_args = dict(backend="disk")

ann_file_train = info_root + "nuscenes_infos_train.pkl"
ann_file_val = info_root + "nuscenes_infos_val.pkl"
ann_file_test = info_root + "nuscenes_infos_val.pkl"

vad_ann_file_train = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_train.pkl")
vad_ann_file_val = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_val.pkl")
vad_ann_file_test = os.path.join(data_infos_root, "vad_nuscenes_infos_temporal_val.pkl")

train_pipeline = [
    # --- Code Change ---
    # Reason: 鍚敤 with_temporal 閫愬抚鍔犺浇宓屽 (T,6)锛孴emporalFlattenTransform 骞抽摵渚涗笅娓稿鐞嗐€?    # TemporalFlattenTransform 蹇呴』鍦?ResizeCropFlipImage 涔嬪墠鈥斺€?    # 鍚﹀垯 ResizeCropFlipImage 閬嶅巻宓屽 list 鑰岄潪 ndarray 鈫?TypeError銆?    dict(type="LoadMultiViewImageFromFiles", to_float32=True, with_temporal=True),
    dict(type="TemporalFlattenTransform"),
    # --- End Code Change ---
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        type="MultiScaleDepthMapGenerator",
        downsample=strides[:num_depth_layers],
    ),
    dict(type='LoadOccWorldLabels', data_root='data/nuscenes', input_dataset='gts'),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=False,
        normalize=False,
        sample_num=num_sample,
        permute=True,
    ),
    # --- Code Change ---
    # Reason: Adaptor 鍔?num_total_frames + perception_multiframe锛堣矾寰?A 闇€瑕侊級锛屽甯ф椂 reshape (T*6,...)鈫?T,6,3,H,W)
    dict(type="NuScenesSparse4DAdaptor", num_total_frames=num_total_frames, perception_multiframe=perception_multiframe),
    # --- End Code Change ---
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_map_labels',
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
            "gt_occ_dense",
            "hist_traj",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

test_pipeline = [
    # --- Code Change ---
    # Reason: 娴嬭瘯 pipeline 鍚屾牱鍚敤澶氬抚銆俆emporalFlattenTransform 蹇呴』鍦?ResizeCropFlipImage 涔嬪墠锛?    # 鍚﹀垯 ResizeCropFlipImage 閬嶅巻宓屽 list 鑰岄潪 ndarray 鈫?TypeError
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, with_temporal=True),
    dict(type="TemporalFlattenTransform"),
    # --- End Code Change ---
    dict(type="ResizeCropFlipImage"),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    # --- Code Change ---
    # Reason: Adaptor 鍔?num_total_frames + perception_multiframe
    dict(type="NuScenesSparse4DAdaptor", num_total_frames=num_total_frames, perception_multiframe=perception_multiframe),
    # --- End Code Change ---
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'ego_status',
            'gt_ego_fut_cmd',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            "hist_traj",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

eval_pipeline = [
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="InstanceNameFilter", classes=class_names),
    dict(type='LoadAnnotations3D_E2E', with_hist_traj=True),
    dict(
        type='VectorizeMap',
        roi_size=roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(
        type='Collect',
        keys=[
            'vectors',
            "gt_bboxes_3d",
            "gt_labels_3d",
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'fut_boxes',
            "hist_traj",
        ],
        meta_keys=['token', 'timestamp']
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    map_classes=map_class_names,
    modality=input_modality,
    version="v1.0-trainval",
)

eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + 'nuscenes_infos_val.pkl',
    vad_ann_file=vad_ann_file_val,
    pipeline=eval_pipeline,
    test_mode=True,
)

data_aug_conf = {
    # "resize_lim": (0.6, 0.6),
    # --- Code Change ---
    # Reason: 672/1600鈮?.42锛屼笌 input_shape 鍚屾缂╁皬锛岀湡姝?resize 鑰岄潪 crop FOV
    "resize_lim": (0.42, 0.42),
    # --- End Code Change ---
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train.pkl",
        vad_ann_file=vad_ann_file_train,
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        # --- Code Change ---
        # Reason: 璁粌闆嗗惎鐢ㄥ甯у姞杞斤紙鍘嗗彶甯?+ 褰撳墠甯э級
        num_history_frames=num_history_frames,
        # --- End Code Change ---
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        vad_ann_file=vad_ann_file_val,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        num_history_frames=num_history_frames,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        vad_ann_file=vad_ann_file_test,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        num_history_frames=num_history_frames,
        test_mode=True,
        eval_config=eval_config,
    ),
    shuffler_sampler=dict(type="GroupInBatchSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

deepspeed = True
deepspeed_config = deepspeed_config

gradient_checkpointing = dict(
    enabled=True,
    checkpoint_activations=True,
    checkpoint_attention=True,
)

optimizer = dict(
    type="AdamW",
    lr=3e-4,   #婧愮爜鐨勫涔犵巼涓?e-4锛岀敱浜巄atchsize璋冨皬浜嗕竴鍗婏紝鎵€浠ュ涔犵巼閫傚綋缂╁皬
    weight_decay=1e-07,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'perception_head.qwen3_vl_with_expert.qwen3_vl': dict(lr_mult=vlm_lr_mult, decay_mult=1.0),
        }
    )
)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=900,  # --- Code Change --- 鍘?900锛坆atch_size=8 鏃?~7%锛夛紝瑙嗛妯″紡 batch_size=1 涓?7384/105480鈮?%
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)

evaluation = dict(
    interval=num_iters_per_epoch * 100,   #姣?00 杞缁冨悗璇勪及涓€娆?    eval_mode=eval_mode,
)

log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, interval=1, warm_up=2000, priority='VERY_HIGH')
]

checkpoint_config = dict(deepspeed=deepspeed, interval=num_iters_per_epoch * checkpoint_epoch_interval, max_keep_ckpts=20) #鏈€杩戜繚鐣欑殑妫€鏌ョ偣鏂囦欢鏁拌缃负20
# A complete Stage-1 model checkpoint initializes the VLM and sparse
# perception decoder together. The launcher sets this for a fresh Stage-2 run.
load_from = stage1_model_checkpoint

find_unused_parameters = True
logger_name = 'mmdet'



