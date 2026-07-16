#!/usr/bin/env bash
# ============================================================
# TPE-MoT unified entry point: training / evaluation / inference / logs
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CONFIG="projects/configs/TPEMoT/tpe_mot_video_2b.py"

MODE="train"
NUM_GPUS=0
EXP_NAME=""
WORK_DIR=""
CKPT=""
LOG_GREP=""
BATCH_SIZE=8
WORKERS=4
RESUME=""
INIT_FROM=""

usage() {
    cat <<'EOF'
Usage:
  bash train.sh [--gpus N] [--exp NAME] [--batch-size N] [--workers N]
  bash train.sh --eval latest|CHECKPOINT [--gpus N] [--exp NAME]
  bash train.sh --infer latest|CHECKPOINT [--exp NAME]
  bash train.sh --logs [--exp NAME] [--grep PATTERN]

Options:
  --config PATH       Config path relative to the repository root.
  --work-dir PATH     Explicit work directory; overrides --exp naming.
  --gpus N            GPU count; defaults to the detected local count.
  --exp NAME          Experiment name under work_dirs/.
  --batch-size N      Samples per GPU during training (default: 8).
  --workers N         Data-loader workers per GPU (default: 4).
  --init-from PATH    Complete Stage-1 .pth or DeepSpeed iter_xxx directory.
  --resume PATH       Resume a TPE-MoT DeepSpeed iteration directory.
  --eval PATH         Multi-GPU evaluation of a checkpoint or iter_xxx directory.
  --infer PATH        Single-GPU inference demo for a checkpoint or iter_xxx directory.
  --logs              Follow the latest training log.
  --grep PATTERN      Filter output while using --logs.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --exp)        EXP_NAME="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --init-from)  INIT_FROM="$2"; shift 2 ;;
        --resume)     RESUME="$2"; shift 2 ;;
        --eval)       MODE="eval"; CKPT="$2"; shift 2 ;;
        --infer)      MODE="infer"; CKPT="$2"; shift 2 ;;
        --logs)       MODE="logs"; shift ;;
        --grep)       LOG_GREP="$2"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

# GPU auto-detection follows the v2 launcher while remaining portable.
if [[ "${NUM_GPUS}" -eq 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    fi
    if [[ "${NUM_GPUS}" -le 0 ]]; then
        NUM_GPUS=1
    fi
fi
export NUM_GPUS

if [[ -z "${DEEPSPEED_CONFIG:-}" ]]; then
    if [[ "${NUM_GPUS}" -ge 8 ]]; then
        export DEEPSPEED_CONFIG="zero_configs/adam_zero1_bf16_video8gpu.json"
    else
        export DEEPSPEED_CONFIG="zero_configs/adam_zero1_bf16.json"
    fi
fi

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="tpe_mot_${NUM_GPUS}gpu"
fi
if [[ -z "${WORK_DIR}" ]]; then
    WORK_DIR="work_dirs/${EXP_NAME}"
fi
mkdir -p "${WORK_DIR}"

find_latest_ckpt() {
    local dir="$1"
    local ckpt=""
    if [[ -L "${dir}/latest" ]]; then
        ckpt="$(readlink -f "${dir}/latest")"
    elif [[ -f "${dir}/latest" ]]; then
        ckpt="${dir}/$(cat "${dir}/latest")"
    else
        ckpt="$(find "${dir}" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' 2>/dev/null | \
            sort -t_ -k2,2n | tail -n 1)"
        [[ -n "${ckpt}" ]] && ckpt="${dir}/${ckpt}"
    fi
    printf '%s\n' "${ckpt}"
}

convert_ckpt() {
    local checkpoint="$1"
    local output="$2"
    "${PYTHON_BIN}" tools/extract_deepspeed_model.py "${checkpoint}" "${output}" >&2
    printf '%s\n' "${output}"
}

if [[ "${MODE}" == "logs" ]]; then
    LATEST_LOG="$(ls -t "${WORK_DIR}"/train_*.log 2>/dev/null | head -n 1 || true)"
    if [[ -z "${LATEST_LOG}" ]]; then
        LATEST_LOG="$(find work_dirs -type f -name 'train_*.log' -printf '%T@ %p\n' 2>/dev/null | \
            sort -nr | head -n 1 | cut -d' ' -f2- || true)"
    fi
    [[ -n "${LATEST_LOG}" ]] || { echo "No training log found." >&2; exit 1; }
    echo "=== ${LATEST_LOG} ==="
    if [[ -n "${LOG_GREP}" ]]; then
        tail -f "${LATEST_LOG}" | grep --color=auto "${LOG_GREP}"
    else
        tail -f "${LATEST_LOG}"
    fi
    exit 0
fi

: "${VLM_PRETRAINED_PATH:?Set VLM_PRETRAINED_PATH to the Stage-1 Qwen3-VL directory.}"
: "${VGGT_OMEGA_PATH:?Set VGGT_OMEGA_PATH to the VGGT-Omega .pt checkpoint.}"

if [[ "${MODE}" == "infer" || "${MODE}" == "eval" ]]; then
    [[ -n "${CKPT}" ]] || { echo "--${MODE} requires a checkpoint path or latest." >&2; exit 2; }
    if [[ "${CKPT}" == "latest" ]]; then
        CKPT="$(find_latest_ckpt "${WORK_DIR}")"
        [[ -n "${CKPT}" ]] || { echo "No checkpoint found in ${WORK_DIR}." >&2; exit 1; }
    fi
    CLEAN_CKPT="${WORK_DIR}/.eval_clean.pth"
    CKPT="$(convert_ckpt "${CKPT}" "${CLEAN_CKPT}")"
fi

if [[ "${MODE}" == "infer" ]]; then
    echo "[TPE-MoT] Single-GPU inference: ${CKPT}"
    "${PYTHON_BIN}" tools/infer_demo.py "${CONFIG}" "${CKPT}" \
        2>&1 | tee "${WORK_DIR}/infer_$(date +%Y%m%d_%H%M%S).log"
    exit 0
fi

if [[ "${MODE}" == "eval" ]]; then
    echo "[TPE-MoT] Evaluation (${NUM_GPUS} GPUs): ${CKPT}"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29501}" \
        tools/test.py "${CONFIG}" "${CKPT}" --launcher pytorch --eval bbox \
        --cfg-options "data.samples_per_gpu=1" "data.workers_per_gpu=${WORKERS}" \
        2>&1 | tee "${WORK_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
    exit 0
fi

# Match v2 behavior: prefer an existing iteration directory unless the caller
# requested a specific resume target. A resumed run never pre-loads Stage-1.
if [[ -z "${RESUME}" ]]; then
    LATEST_CKPT="$(find_latest_ckpt "${WORK_DIR}")"
    if [[ -n "${LATEST_CKPT}" && -d "${LATEST_CKPT}" ]]; then
        RESUME="${LATEST_CKPT}"
        echo "[TPE-MoT] Auto-resume: ${RESUME}"
    fi
fi

if [[ -n "${RESUME}" ]]; then
    export TPE_MOT_STAGE1_CHECKPOINT=""
else
    if [[ -n "${INIT_FROM}" ]]; then
        export TPE_MOT_STAGE1_CHECKPOINT="${INIT_FROM}"
    fi
    : "${TPE_MOT_STAGE1_CHECKPOINT:?Set TPE_MOT_STAGE1_CHECKPOINT or pass --init-from for a fresh Stage-2 run.}"
    if [[ -d "${TPE_MOT_STAGE1_CHECKPOINT}" ]]; then
        STAGE1_CLEAN_CKPT="${WORK_DIR}/.stage1_init.pth"
        export TPE_MOT_STAGE1_CHECKPOINT="$(convert_ckpt "${TPE_MOT_STAGE1_CHECKPOINT}" "${STAGE1_CLEAN_CKPT}")"
    fi
    [[ -f "${TPE_MOT_STAGE1_CHECKPOINT}" ]] || {
        echo "Stage-1 checkpoint not found: ${TPE_MOT_STAGE1_CHECKPOINT}" >&2
        exit 2
    }
fi

echo "============================================"
echo " TPE-MoT Stage-2 Training"
echo "============================================"
echo " Experiment:    ${EXP_NAME}"
echo " Work dir:      ${WORK_DIR}"
echo " Config:        ${CONFIG}"
echo " GPUs:          ${NUM_GPUS}"
echo " Batch/GPU:     ${BATCH_SIZE}"
echo " DeepSpeed:     ${DEEPSPEED_CONFIG}"
echo " VLM weights:   ${VLM_PRETRAINED_PATH}"
echo " Omega weights: ${VGGT_OMEGA_PATH}"
if [[ -n "${RESUME}" ]]; then
    echo " Resume:        ${RESUME}"
else
    echo " Stage-1 init:  ${TPE_MOT_STAGE1_CHECKPOINT}"
fi
echo " Python:        $(command -v "${PYTHON_BIN}")"
echo "============================================"

TRAIN_ARGS=()
if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume-from "${RESUME}")
fi

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29500}" \
    tools/train.py "${CONFIG}" --launcher pytorch --deterministic \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
        "data.samples_per_gpu=${BATCH_SIZE}" \
        "data.workers_per_gpu=${WORKERS}" \
        "log_config.interval=5" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${WORK_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
