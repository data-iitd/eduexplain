#!/usr/bin/env bash
# Whole demo evaluation, unattended (tmux vx-exp): SetCoRe-LW/E2E -> OptiSet -> stage1/2/3 for 35 videos.
# Every step skips what is already finished, so re-running this script resumes.
set -uo pipefail
EXP=/mnt/storage/aryan/VideoExplain/experiments/demo_eval
cd $EXP
log() { echo "$(date '+%F %T') $*"; }
log "waiting for SetCoRe-LW / SetCoRe-E2E (tmux vx-exp-s1)"
while tmux has-session -t vx-exp-s1 2>/dev/null && ! grep -q "SETCORE DONE" logs/setcore.log; do sleep 30; done
if [ ! -s stage1/e2e_s0/predictions.jsonl ] || [ ! -s stage1/lw_s0/predictions.jsonl ]; then
  log "SetCoRe did not finish: re-running run_setcore.sh"; ./run_setcore.sh >> logs/setcore.log 2>&1
fi
if [ ! -s stage1/optiset/optiset_summary.json ]; then
  log "OptiSet (Llama-3.1-70B via vLLM :8003)"
  mkdir -p stage1/optiset
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false \
  VLLM_GPU_PEERS=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1 \
  VLLM_SLEEP_LEVELS=http://127.0.0.1:8002/v1=2,http://127.0.0.1:8003/v1=2 \
  /mnt/storage/aryan/set2story/venv/bin/python -u optiset_vllm.py \
    --benchmark stage1/eval_queries_gold.json --resources /mnt/storage/aryan/setCore/all_resources_unique_ids_clean1_merged.json \
    --embedding-model /mnt/storage/aryan/setCore/models/bge-large-en-v1.5 --llm-model Llama-3.1-70B-Instruct \
    --retrieval-fields summary --prompt-fields summary --top-k 20 --num-subqueries 3 --temperature 0.2 \
    --output stage1/optiset/optiset_summary.json > logs/optiset.log 2>&1 || log "OptiSet FAILED (see logs/optiset.log)"
fi
log "stage 1 / 2 / 3"
/mnt/storage/aryan/VideoExplain/venv/bin/python -u run_demo_eval.py all
log "run_all finished"
