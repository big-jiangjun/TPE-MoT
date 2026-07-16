# TPE-MoT

TPE-MoT is a temporal perception-expert mixture-of-transformers framework for
six-camera autonomous-driving video. It processes four frames per camera and
injects frozen VGGT-Omega spatial priors through two trainable routes:

- **Prefix route:** Omega local and global tokens enhance a selected Qwen3-VL
  deepstack feature before the understanding expert.
- **Perception route:** Omega camera/register global tokens cross-attend to
  detection, map, and ego query embeddings before the perception expert.

The released configuration has no ego planning head and disables the action
expert and occupancy tokens.
The perception sequence is `900 det + 100 map + 1 ego = 1001` tokens.

## Repository layout

```text
projects/configs/TPEMoT/tpe_mot_video_2b.py  Active four-frame, six-camera config
projects/mmdet3d_plugin/tpe_mot/              TPE-MoT detector and model modules
third_party/vggt-omega/                       VGGT-Omega source and license
tools/extract_deepspeed_model.py               DeepSpeed checkpoint extraction
train.sh                                      Training and evaluation entry point
```

No datasets, pretrained weights, experiment logs, or checkpoints are included.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before use.

## Initialization

Fresh TPE-MoT Stage-2 training uses two distinct weight sources:

- `VLM_PRETRAINED_PATH`: the Stage-1 Qwen3-VL directory used while constructing
  the vision-language model.
- `TPE_MOT_STAGE1_CHECKPOINT`: a complete Stage-1 model checkpoint used as
  `load_from`, restoring checkpoint parameters compatible with the current
  sparse perception decoder and VLM. It may be a cleaned `.pth` file or a
  DeepSpeed iteration directory when passed to `train.sh --init-from`.

TPE-MoT reuses the UniDriveVLA Stage-1 weights for this initialization. The
VGGT-Omega spatial encoder is loaded separately, while the two Omega
connectors are newly initialized and jointly trained in Stage-2.

`--resume` resumes a TPE-MoT run and intentionally does not reload Stage-1.

## Setup

TPE-MoT supports the nuScenes training stack only. Create the Python 3.9
conda environment named `TPE_MOT`, install PyTorch, MMCV/MMDetection3D,
DeepSpeed, Qwen3-VL, VGGT-Omega dependencies, and build the custom CUDA ops
according to the [nuScenes installation guide](docs/installation.md).

Prepare nuScenes, CAN bus, Occ3D labels, temporal annotation PKLs, and K-means
anchors using the [nuScenes data preparation guide](docs/data_preparation.md).
The active configuration expects `data/nuscenes/`, `data/infos/`, and
`data/kmeans/` below the repository root.

VGGT-Omega source is included under its FAIR Noncommercial Research License;
review `third_party/vggt-omega/LICENSE` before use or redistribution.

For fresh Stage-2 training, set the following paths after activating `TPE_MOT`:

```bash
export VLM_PRETRAINED_PATH=/path/to/official_weights_stage1
export VGGT_OMEGA_PATH=/path/to/vggt_omega_1b_256_text.pt
export TPE_MOT_STAGE1_CHECKPOINT=/path/to/stage1_model_clean.pth
export DATA_INFOS_ROOT=/path/to/temporal_nuscenes_infos
```

Leave `DATA_INFOS_ROOT` unset when temporal PKLs are stored in `data/infos/`.

## Train

Run the commands below from the `nuScenes_v3` repository root:

```bash
cd /path/to/nuScenes_v3
```

Fresh Stage-2 training from a UniDriveVLA Stage-1 DeepSpeed checkpoint:

```bash
export VLM_PRETRAINED_PATH=/path/to/official_weights_stage1
export VGGT_OMEGA_PATH=/path/to/vggt_omega_1b_256_text.pt

bash train.sh \
  --gpus 8 \
  --exp tpe_mot_stage2 \
  --init-from /path/to/stage1/iter_2000
```

Single GPU:

```bash
bash train.sh --gpus 1 --batch-size 1 --work-dir work_dirs/tpe_mot_2b
```

Eight GPUs:

```bash
bash train.sh --gpus 8 --batch-size 8 --work-dir work_dirs/tpe_mot_2b_8gpu
```

The launcher follows the v2 workflow: `--exp NAME` selects
`work_dirs/NAME`, automatically resumes the newest `iter_xxx/` directory in
that work directory, and accepts `latest` for evaluation or inference. For a
fresh Stage-2 run, pass a checkpoint without exporting it first:

```bash
bash train.sh --gpus 8 --exp tpe_mot_stage2 \
  --init-from /path/to/stage1/iter_2000
```

Resume a DeepSpeed iteration directory:

```bash
bash train.sh --gpus 8 --batch-size 8 \
  --work-dir work_dirs/tpe_mot_2b_8gpu \
  --resume work_dirs/tpe_mot_2b_8gpu/iter_2000
```

Follow logs from the selected experiment:

```bash
bash train.sh --logs --exp tpe_mot_2b_8gpu
bash train.sh --logs --exp tpe_mot_2b_8gpu --grep loss
```

## Evaluate

`train.sh --eval` accepts either a regular `.pth` checkpoint or a DeepSpeed
`iter_xxx/` directory. It writes a non-EMA extracted `.pth` file under the
chosen work directory before starting distributed evaluation.

```bash
bash train.sh --eval work_dirs/tpe_mot_2b_8gpu/iter_2000 \
  --gpus 8 --batch-size 1 \
  --work-dir work_dirs/tpe_mot_2b_8gpu_eval
```

Single-GPU inference uses the same checkpoint conversion path:

```bash
bash train.sh --infer latest --exp tpe_mot_2b_8gpu
```

## Fusion and training contract

For an input `[B, 4, 6, 3, H, W]`, Omega receives `B*6` sequences in
`[t-3, t-2, t-1, current]` order. One frozen Omega forward produces both the
prefix connector inputs and global memory `[B, 6*4*17, 2048]`. The two
connectors are ordinary registered `nn.Module`s and are optimized jointly with
the main model; the Omega spatial encoder remains frozen by default.
