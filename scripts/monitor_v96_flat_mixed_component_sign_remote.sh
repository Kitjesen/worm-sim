#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bsrl/hongsenpang/codex_runs/worm-sim-v6-server-training"
PY="/home/bsrl/miniconda3/envs/wormv6_np2/bin/python"
TODAY="${TODAY:-20260603}"
RUN_DIR="${RUN_DIR:-runs/worm_v6_ppo_flat_random_v96_server_mixed_component_sign_from_v95final_np2}"
SCAN_DIR="${SCAN_DIR:-record/current/flat_omni_v96_server_mixed_component_sign_scan}"
TRAIN_PID="${TRAIN_PID:-}"

cd "$ROOT"
mkdir -p "$SCAN_DIR" server_logs

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "server_logs/v96_flat_mixed_component_sign_monitor_${TODAY}.log"
}

wait_for_pid() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]]; then
    log "skip pid wait for ${label}: no pid provided"
    return 0
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    log "${label} pid=${pid} is already stopped"
    return 0
  fi
  log "waiting for ${label} pid=${pid}"
  while ps -p "$pid" >/dev/null 2>&1; do
    sleep 60
  done
  log "${label} pid=${pid} finished"
}

scan_one() {
  local label="$1"
  local model_name="$2"
  local condition="$3"
  shift 3

  local model="${RUN_DIR}/${model_name}.zip"
  local scan_json="${SCAN_DIR}/${label}_scan.json"
  local scan_csv="${SCAN_DIR}/${label}_scan.csv"
  local analysis_json="${SCAN_DIR}/${label}_analysis.json"
  local analysis_md="${SCAN_DIR}/${label}_analysis.md"

  if [[ ! -f "$model" ]]; then
    log "skip ${label}: missing ${model}"
    return 0
  fi

  log "scan ${label} using ${model}"
  "$PY" src/v6/scan_command_tracking_v6.py \
    --run-dir "$RUN_DIR" \
    --model "$model" \
    --terrain flat \
    --gait-mode random \
    --time 6 \
    --seed 20260603 \
    --eval-condition "$condition" \
    --json-out "$scan_json" \
    --csv-out "$scan_csv" \
    "$@"

  "$PY" src/v6/analyze_command_scan_v6.py \
    --scan "$scan_json" \
    --out-json "$analysis_json" \
    --out-md "$analysis_md"
  log "done ${label}"
}

wait_for_pid "$TRAIN_PID" train

pids=()
for model_name in final_model progress_best_model best_model; do
  scan_one "v96_${model_name}_nominal" "$model_name" nominal &
  pids+=("$!")
  scan_one "v96_${model_name}_robust" "$model_name" robust \
    --encoder-pos-noise 0.01 \
    --encoder-vel-noise 0.02 \
    --imu-gravity-noise 0.01 \
    --imu-gyro-noise 0.01 \
    --action-delay-steps 1 \
    --action-saturation 0.9 &
  pids+=("$!")
done

scan_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    scan_status=1
  fi
done

summary_args=()
for item in \
  v96_final_model_nominal \
  v96_final_model_robust \
  v96_progress_best_model_nominal \
  v96_progress_best_model_robust \
  v96_best_model_nominal \
  v96_best_model_robust; do
  scan_json="${SCAN_DIR}/${item}_scan.json"
  analysis_json="${SCAN_DIR}/${item}_analysis.json"
  if [[ -f "$scan_json" && -f "$analysis_json" ]]; then
    summary_args+=(--item "$item" "$scan_json" "$analysis_json")
  fi
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  "$PY" src/v6/summarize_scan_acceptance_v6.py \
    "${summary_args[@]}" \
    --title "Worm V6 V96 flat mixed-component sign acceptance" \
    --out-json "${SCAN_DIR}/v96_flat_mixed_component_sign_summary.json" \
    --out-csv "${SCAN_DIR}/v96_flat_mixed_component_sign_summary.csv" \
    --out-md "${SCAN_DIR}/v96_flat_mixed_component_sign_summary.md"
else
  log "summary skipped: no scan/analysis pairs produced"
  scan_status=1
fi

log "monitor complete scan_status=${scan_status}"
exit "$scan_status"
