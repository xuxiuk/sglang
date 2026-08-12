#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/sglang-dspark-csd
BASE=${RUN_ROOT_OVERRIDE:-$ROOT/runs/dspark_csd/final_results}
TABLE=$ROOT/runs/dspark_csd/formal_redpajama_20260724/tables/dspark_csd_merged.json
SPS_TABLE=$ROOT/runs/dspark_csd/dspark_sps_h20_tp8_dp8_bs64_input512.json
MATRIX=$ROOT/scripts/run_dspark_accuracy81920_matrix.sh

mkdir -p "$BASE"
exec > >(tee -a "$BASE/supervisor.log") 2>&1

echo "[$(date -Is)] run started"
echo "git_head=$(git -C "$ROOT" rev-parse HEAD)"
echo "table=$TABLE"
echo "table_sha256=$(sha256sum "$TABLE" | awk '{print $1}')"

common=(
  ROOT="$ROOT"
  MODEL=/data/model/DeepSeek-V4-Flash-DSpark
  CSD_TABLE="$TABLE"
  SPS_TABLE="$SPS_TABLE"
  METHODS="bare plain dynamic entropy"
  RAGGED_VERIFY_MODE=static
  GPU_SET=0,1,2,3,4,5,6,7
  TP_SIZE=8 DP_SIZE=8 EP_SIZE=1
  ENABLE_DP_ATTENTION=1 ENABLE_DP_LM_HEAD=1
  MOE_A2A_BACKEND=none MOE_RUNNER_BACKEND=flashinfer_mxfp4
  CHUNKED_PREFILL_SIZE=2048
  MEM_FRACTION_STATIC=0.88
  SWA_FULL_TOKENS_RATIO=0.2
  SWA_EVICTION_INTERVAL=128
  MAX_TOKENS=81920 LCB_MAX_TOKENS=81920 MAX_MODEL_LENGTH=96000
  CSD_PROB_RATIO=0.3
  ENTROPY_THRESHOLD=1.5638477802276611
  REQUEST_TIMEOUT_SECONDS=86400 MAX_RETRIES=0
  NCCL_NET=Socket
)

echo "[$(date -Is)] phase=lcb start"
env "${common[@]}" \
  RUN_ROOT="$BASE" TASKS=lcb_avg4 \
  MAX_RUNNING_REQUESTS=48 EVAL_THREADS=48 \
  PORT=36000 DIST_INIT_ADDR=127.0.0.1:36010 NCCL_PORT=36020 \
  bash "$MATRIX"
mv "$BASE/config.txt" "$BASE/config_lcb.txt"
echo "[$(date -Is)] phase=lcb complete"

echo "[$(date -Is)] phase=remaining start (AIME avg@16/pass@16; Math500 and GSM8K avg@4/pass@4)"
env "${common[@]}" \
  RUN_ROOT="$BASE" \
  TASKS="aime25_avg16 math500_avg4 gsm8k_avg4" \
  MAX_RUNNING_REQUESTS=64 EVAL_THREADS=64 \
  PORT=37000 DIST_INIT_ADDR=127.0.0.1:37010 NCCL_PORT=37020 \
  bash "$MATRIX"
mv "$BASE/config.txt" "$BASE/config_aime16_math_gsm8k.txt"
echo "[$(date -Is)] phase=remaining complete"
echo "[$(date -Is)] run complete"
