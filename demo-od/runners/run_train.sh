#!/usr/bin/env bash
# Train the base four-class NMS-free detector on UA-DETRAC YOLO data.
set -euo pipefail

RUNNERS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd -- "${RUNNERS_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${DEMO_DIR}/.." && pwd)"
TRAIN_PYTHON="${OD_PYTHON:-${REPO_DIR}/venv/bin/python}"
if [[ ! -x "$TRAIN_PYTHON" ]]; then
    TRAIN_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$TRAIN_PYTHON" ]]; then
    echo "Python not found. Set OD_PYTHON to your training environment." >&2
    exit 1
fi

export NO_ALBUMENTATIONS_UPDATE=1
args=(
    -u "$DEMO_DIR/core/train.py"
    --input-dir "${OD_DATASET_DIR:-${REPO_DIR}/ua_detrac_yolo_train}"
    --output-dir "${OD_OUTPUT_DIR:-${DEMO_DIR}/output}"
    --job-id "${OD_JOB_ID:-od_nmsfree_$(date +%Y%m%d_%H%M%S)}"
    --epochs "${OD_EPOCHS:-1}"
    --batch-size "${OD_BATCH_SIZE:-8}"
    --learning-rate "${OD_LEARNING_RATE:-0.0005}"
    --one2many-loss-weight "${OD_ONE2MANY_LOSS_WEIGHT:-0.5}"
    --obj-focal-gamma "${OD_OBJ_FOCAL_GAMMA:-2}"
    --device "${OD_DEVICE:-auto}"
    --progress "${OD_PROGRESS:-auto}"
    --workers "${OD_WORKERS:-4}"
    --log-every-images "${OD_LOG_EVERY_IMAGES:-500}"
    --eval-limit "${OD_EVAL_LIMIT:-2000}"
    --preview-every-images "${OD_PREVIEW_EVERY_IMAGES:-2000}"
    --preview-images "${OD_PREVIEW_IMAGES:-4}"
)
if [[ -n "${OD_PRETRAINED_WEIGHTS:-}" ]]; then
    args+=(--pretrained-weights "$OD_PRETRAINED_WEIGHTS")
fi
exec "$TRAIN_PYTHON" "${args[@]}" "$@"
