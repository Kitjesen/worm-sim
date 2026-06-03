#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training"
PY="/home/bsrl/miniconda3/envs/wormv6_np2/bin/python"
TODAY="${TODAY:-20260603}"

cd "$ROOT"
mkdir -p server_logs

ACTIVE="$(
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /src\/v6\/train_v6.py/ {print}'
)"
if [[ -n "$ACTIVE" ]]; then
  echo "[v97] active train_v6.py process detected; not launching duplicate"
  echo "$ACTIVE"
  exit 2
fi

unset WORM_V6_ENABLE_MIXED_COMMAND_COMPOSITION || true
unset WORM_V6_ENABLE_MIXED_PLANAR_SPLIT_PRIOR || true
unset WORM_V6_ENABLE_MIXED_PLANAR_FULL_CHANNEL_SPLIT_PRIOR || true
unset WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE || true
unset WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE || true
unset WORM_V6_ENABLE_CLOSED_LOOP_YAW_FEEDBACK || true

export WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=1
export WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE=1
export WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE=1
export WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT="${WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT:-0.65}"
export WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT="${WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT:-2.50}"

RESUME_MODEL="${RESUME_MODEL:-runs/worm_v6_ppo_flat_random_v96_server_mixed_component_sign_from_v95final_np2/progress_best_model.zip}"
RUN_LABEL="${RUN_LABEL:-flat_random_v97_server_symmetric_mixed_gate_from_v96progress_np2}"
TRAIN_LOG="${TRAIN_LOG:-server_logs/v97_flat_symmetric_mixed_gate_${TODAY}.log}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-3689272}"
TRAIN_CHUNK_TIMESTEPS="${TRAIN_CHUNK_TIMESTEPS:-120000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SCAN_DIR="${SCAN_DIR:-record/current/flat_omni_v97_server_symmetric_mixed_gate_scan}"

nohup env PYTHONUNBUFFERED=1 \
  WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE="$WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE" \
  WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE="$WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE" \
  WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE="$WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE" \
  WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT="$WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT" \
  WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT="$WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT" \
  "$PY" src/v6/train_v6.py \
  --terrain flat \
  --gait-mode random \
  --run-label "$RUN_LABEL" \
  --command-curriculum mixed_composition_repair \
  --resume "$RESUME_MODEL" \
  --allow-contract-resume \
  --allow-curriculum-resume \
  --timesteps "$TARGET_TIMESTEPS" \
  --train-chunk-timesteps "$TRAIN_CHUNK_TIMESTEPS" \
  --n-envs 8 \
  --device cpu \
  --learning-rate "$LEARNING_RATE" \
  --policy-net-arch 512,256,128 \
  --value-net-arch 512,256,128 \
  --encoder-pos-noise 0.01 \
  --encoder-vel-noise 0.02 \
  --imu-gravity-noise 0.01 \
  --imu-gyro-noise 0.01 \
  --action-delay-steps 1 \
  --action-saturation 0.9 \
  --directional-eval-freq-steps 5000 \
  --directional-eval-seconds 6.0 \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID="$!"
RUN_DIR="runs/worm_v6_ppo_${RUN_LABEL}"
PID_FILE="server_logs/v97_flat_symmetric_mixed_gate_train_pids_${TODAY}.env"

cat > "$PID_FILE" <<EOF
TRAIN_PID=${TRAIN_PID}
RUN_DIR=${RUN_DIR}
SCAN_DIR=${SCAN_DIR}
TRAIN_LOG=${TRAIN_LOG}
TARGET_TIMESTEPS=${TARGET_TIMESTEPS}
TRAIN_CHUNK_TIMESTEPS=${TRAIN_CHUNK_TIMESTEPS}
LEARNING_RATE=${LEARNING_RATE}
WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE=${WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE}
WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE=${WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE}
WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE=${WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE}
WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT=${WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT}
WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT=${WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT}
EOF

echo "train_pid=${TRAIN_PID}"
echo "run_dir=${RUN_DIR}"
echo "train_log=${TRAIN_LOG}"
echo "pid_file=${PID_FILE}"

nohup env \
  TRAIN_PID="$TRAIN_PID" \
  RUN_DIR="$RUN_DIR" \
  SCAN_DIR="$SCAN_DIR" \
  TRAIN_LOG="$TRAIN_LOG" \
  WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE="$WORM_V6_ENABLE_MIXED_PLANAR_HARDCASE_GATE" \
  WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE="$WORM_V6_ENABLE_MIXED_PLANAR_CONTINUOUS_GATE" \
  WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE="$WORM_V6_ENABLE_MIXED_PLANAR_AUTHORITY_REBALANCE" \
  WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT="$WORM_V6_MIXED_PLANAR_PRIOR_AUTHORITY_MULT" \
  WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT="$WORM_V6_MIXED_PLANAR_RESIDUAL_SCALE_MULT" \
  bash scripts/monitor_v97_flat_symmetric_mixed_gate_remote.sh \
  > "server_logs/v97_flat_symmetric_mixed_gate_monitor_stdout_${TODAY}.log" 2>&1 &
echo "monitor_pid=$!"
