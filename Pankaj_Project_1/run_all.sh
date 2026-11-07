#!/usr/bin/env bash
# Runs all three tasks of Project 1 (WikiText-2) end to end.
#   bash run_all.sh              full experiment (AWD-LSTM: 50 epochs)
#   QUICK=1 bash run_all.sh      quick test of the whole pipeline (one short epoch)
#   EPOCHS=30 bash run_all.sh    change the number of AWD-LSTM epochs
# AWD-LSTM training is checkpointed after every epoch and resumes automatically if re-run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
EXTRA=""
if [[ "${QUICK:-0}" == "1" ]]; then
  EPOCHS=${EPOCHS:-1}; EXTRA="--max_batches 30"
  export CKPT_DIR=${CKPT_DIR:-checkpoints_quick} RESULTS_DIR=${RESULTS_DIR:-results_quick}
  export CACHE_DIR=${CACHE_DIR:-cache_quick} LOG_DIR=${LOG_DIR:-logs_quick}
fi
LOG=${LOG_DIR:-logs}
mkdir -p "$LOG"

# run the three tasks in order (task 3 uses the saved outputs of tasks 1 and 2)
echo "=== Task 1: smoothed n-gram LMs ==="
$PY -W ignore src/task1_ngram.py 2>&1 | tee "$LOG/task1.log"

echo "=== Task 2: AWD-LSTM (train, resumable) ==="
$PY -W ignore src/task2_awd_lstm.py train --epochs "${EPOCHS:-50}" $EXTRA 2>&1 | tee -a "$LOG"/task2_train.log
echo "=== Task 2: evaluation ==="
$PY -W ignore src/task2_awd_lstm.py eval 2>&1 | tee "$LOG"/task2_eval.log

echo "=== Task 3: KN + AWD-LSTM hybrid ==="
$PY -W ignore src/task3_hybrid.py 2>&1 | tee "$LOG"/task3.log

echo "=== Figures ==="
$PY src/make_plots.py
echo "Done. Tables: ${RESULTS_DIR:-results}/task{1,2,3}_results.md  Figures: ${RESULTS_DIR:-results}/figures/"
