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
  echo "[v96] active train_v6.py process detected; not launching duplicate"
  echo "$ACTIVE"
  exit 2
fi

unset WORM_V6_ENABLE_SLOPE_FORWARD_AXIS_PROFILE || true
unset WORM_V6_ENABLE_SLOPE_MIXED_PLANAR_PRIMITIVE || true
unset WORM_V6_ENABLE_CLOSED_LOOP_YAW_FEEDBACK || true

RESUME_MODEL="${RESUME_MODEL:-runs/worm_v6_ppo_flat_random_v95_server_mixed_positive_vx_hardcase_from_v92bfinal_np2/final_model.zip}"
RUN_LABEL="${RUN_LABEL:-flat_random_v96_server_mixed_component_sign_from_v95final_np2}"
TRAIN_LOG="${TRAIN_LOG:-server_logs/v96_flat_mixed_component_sign_${TODAY}.log}"
TARGET_TIMESTEPS="${TARGET_TIMESTEPS:-3569272}"
TRAIN_CHUNK_TIMESTEPS="${TRAIN_CHUNK_TIMESTEPS:-100000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SCAN_DIR="${SCAN_DIR:-record/current/flat_omni_v96_server_mixed_component_sign_scan}"

nohup env PYTHONUNBUFFERED=1 \
  "$PY" src/v6/train_v6.py \
  --terrain flat \
  --gait-mode random \
  --run-label "$RUN_LABEL" \
  --command-curriculum robust_forward_left_diagonal_repair \
  --resume "$RESUME_MODEL" \
  --allow-contract-resume \
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
PID_FILE="server_logs/v96_flat_mixed_component_sign_train_pids_${TODAY}.env"

cat > "$PID_FILE" <<EOF
TRAIN_PID=${TRAIN_PID}
RUN_DIR=${RUN_DIR}
SCAN_DIR=${SCAN_DIR}
TRAIN_LOG=${TRAIN_LOG}
TARGET_TIMESTEPS=${TARGET_TIMESTEPS}
TRAIN_CHUNK_TIMESTEPS=${TRAIN_CHUNK_TIMESTEPS}
LEARNING_RATE=${LEARNING_RATE}
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
  bash scripts/monitor_v96_flat_mixed_component_sign_remote.sh \
  > "server_logs/v96_flat_mixed_component_sign_monitor_stdout_${TODAY}.log" 2>&1 &
echo "monitor_pid=$!"
