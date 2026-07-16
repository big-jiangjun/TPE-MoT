# TPE-MoT nuScenes Training Environment

This guide covers the Python 3.9 environment used for TPE-MoT nuScenes
training, evaluation, and single-sample inference.

## 1. Create the Conda Environment

```bash
conda create -n TPE_MOT python=3.9
conda activate TPE_MOT
```

## 2. Install PyTorch

Detect the CUDA wheel suffix for the installed CUDA toolkit:

```bash
CUDA_VERSION=$(nvcc -V | grep -oP 'release \K[\d.]+' | cut -d. -f1-2 | tr -d '.')
CUDA_SUFFIX="cu${CUDA_VERSION:0:2}${CUDA_VERSION:2:2}"
echo "Detected CUDA wheel suffix: ${CUDA_SUFFIX}"
```

Install the matching PyTorch release:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url "https://download.pytorch.org/whl/${CUDA_SUFFIX}"
```

For example, CUDA 12.1 uses `cu121` and CUDA 11.8 uses `cu118`.

## 3. Install Python Dependencies

From the `nuScenes_v3` repository root:

```bash
pip install transformers==4.57.1
pip install -r requirements/requirements_nusc.txt
pip install deepspeed==0.14.4 peft timm qwen_vl_utils
pip install flash-attn --no-build-isolation
pip install -r third_party/vggt-omega/requirements.txt
```

`transformers==4.57.1` provides the Qwen3-VL modules imported by TPE-MoT; no
copy into the site-packages directory is required.

## 4. Install the MMDetection Stack

TPE-MoT follows the original nuScenes training stack: MMCV 1.7.2 is built with
CUDA ops and MMDetection3D 1.0.0rc6 is installed in editable mode. Use the
TPE-MoT-compatible source trees under `third_party/` if they are included in
your release package; otherwise obtain the matching patched sources before
running these commands.

```bash
cd third_party/mmcv-1.7.2
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MMCV_NO_Compiler_CHECK=1
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
cd ../..

cd third_party/mmdetection3d-1.0.0rc6
pip install -e . --no-build-isolation
cd ../..
```

## 5. Build TPE-MoT Custom Ops

```bash
cd /path/to/nuScenes_v3/projects/mmdet3d_plugin/ops
pip install -e . --no-build-isolation
cd ../../..
```

## 6. Verify the Environment

From the repository root, check that the key packages import successfully:

```bash
python -c "import torch, mmcv, mmdet, mmdet3d, transformers; print(torch.__version__); print(transformers.__version__)"
```

Continue with [nuScenes data preparation](data_preparation.md) before starting
a fresh Stage-2 training run.
