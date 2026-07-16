# nuScenes Data Preparation

This guide prepares the nuScenes inputs required by the active TPE-MoT
configuration, `projects/configs/TPEMoT/tpe_mot_video_2b.py`. Run every command
from the `nuScenes_v3` repository root unless stated otherwise.

## 1. Download nuScenes

Download the nuScenes `v1.0-trainval` and `v1.0-test` releases, the map
expansion, and the CAN bus expansion from the [nuScenes download
page](https://www.nuscenes.org/nuscenes#download). Extract the CAN bus files
under the same nuScenes data root, so that `can_bus/`, `maps/`, `samples/`,
`sweeps/`, and the `v1.0-*` metadata directories are all available below it.

The TPE-MoT configuration reads the dataset from `data/nuscenes/`. A symbolic
link keeps the dataset outside the repository while preserving that path:

```bash
cd /path/to/nuScenes_v3
mkdir -p data
ln -s /path/to/nuscenes data/nuscenes
```

Afterwards, `data/nuscenes/can_bus/` and `data/nuscenes/v1.0-trainval/` should
exist. Use a directory copy instead of a symbolic link on filesystems where
symbolic links are unavailable.

## 2. Download Occupancy Labels

Although the released configuration disables occupancy-token prediction, its
nuScenes pipeline still reads Occ3D labels. Download
[Occ3D-nuScenes](https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc)
and place the extracted `gts/` directory at:

```text
data/nuscenes/gts/
```

## 3. Generate Annotation PKLs

Create the output directories and generate the standard nuScenes info PKLs:

```bash
cd /path/to/nuScenes_v3
mkdir -p data/infos data/kmeans
bash tools/create_data.sh
```

The TPE-MoT video dataset also requires temporal annotation files with the
`vad_nuscenes` prefix. Generate them from the same data root:

```bash
PYTHONPATH=. python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/infos \
  --extra-tag vad_nuscenes \
  --version v1.0 \
  --canbus ./data/nuscenes
```

The first command writes the standard info files used by the base nuScenes
dataset. The second command writes the temporal files read by the TPE-MoT
video pipeline. By default, the configuration reads both sets from
`data/infos/`. To keep temporal PKLs elsewhere, set `DATA_INFOS_ROOT` to that
directory before calling `train.sh`.

## 4. Generate K-means Anchors

The active perception configuration needs detection and map anchors:

```bash
python tools/kmeans/kmeans_det.py
python tools/kmeans/kmeans_map.py
```

## 5. Validate the Layout

Before training, verify that the following required files exist:

```text
data/
  infos/
    nuscenes_infos_train.pkl
    nuscenes_infos_val.pkl
    vad_nuscenes_infos_temporal_train.pkl
    vad_nuscenes_infos_temporal_val.pkl
  kmeans/
    kmeans_det_900.npy
    kmeans_map_100.npy
  nuscenes/
    can_bus/
    gts/
    maps/
    samples/
    sweeps/
    v1.0-trainval/
    v1.0-test/
```

The released model has no occupancy-token or ego-planning output branch, but
the listed Occ3D labels and temporal PKLs remain required by the current data
pipeline.
