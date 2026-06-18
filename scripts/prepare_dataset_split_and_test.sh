#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-dataset_train}"
BACKUP_ROOT="${BACKUP_ROOT:-dataset_train_raw_backup}"
RATIOS="${RATIOS:-7:1:2}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TEST_NUM_FRAMES="${TEST_NUM_FRAMES:-50}"
TEST_SEED="${TEST_SEED:-2026}"
TEST_CANDIDATE_POOL="${TEST_CANDIDATE_POOL:-24}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "[1/3] Backing up ${DATASET_ROOT} -> ${BACKUP_ROOT}"
python scripts/backup_dataset_contents.py \
  --src-dir "${DATASET_ROOT}" \
  --dst-dir "${BACKUP_ROOT}"

echo "[2/3] Splitting ${DATASET_ROOT} into train/val/test"
python scripts/split_dataset_train_val_test.py \
  --dataset-root "${DATASET_ROOT}" \
  --ratios "${RATIOS}" \
  --seed "${SPLIT_SEED}"

echo "[3/3] Selecting final test frames"
python scripts/select_dataset_test_frames.py \
  --test-root "${DATASET_ROOT}/test" \
  --num-frames "${TEST_NUM_FRAMES}" \
  --seed "${TEST_SEED}" \
  --candidate-pool "${TEST_CANDIDATE_POOL}" \
  --no-dry-run

echo "[DONE] Dataset backup, split, and test-frame selection complete."
