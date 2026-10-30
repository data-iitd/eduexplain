#!/usr/bin/env bash
# Stage 3 only, with RAM freed: gemma :8002 and Llama-70B :8003 stopped (the 70B pins ~18 GB even asleep), the
# 27B/9B servers restarted (long-lived vLLM leaks pinned host memory). Large figures stalled Manim in memory reclaim
# (render timeouts) when RAM and swap were full. Afterwards gemma and Llama are started again for the website.
set -uo pipefail
EXP=/mnt/storage/aryan/VideoExplain/experiments/demo_eval; VX=/mnt/storage/aryan/VideoExplain
log() { echo "$(date '+%F %T') $*"; }
up() { curl -sf --max-time 5 "http://127.0.0.1:$1/v1/models" >/dev/null; }
for s in vx-llama vx-gemma vx-vllm; do tmux kill-session -t $s 2>/dev/null; done
while pgrep -u aryan -f "VLLM::EngineCore|venv-vllm/bin/python" >/dev/null; do sleep 5; done
log "vLLM stopped"; free -g | head -2
tmux new-session -d -s vx-vllm "cd /mnt/storage/aryan/VideoModule && MODEL_ROOT=/mnt/storage/aryan/VideoModule/models ./serve_models.sh 2>&1 | tee $VX/logs/vllm.log"
until up 8000 && up 8001; do sleep 20; done
log "27B/9B serving"; free -g | head -2
cd $EXP && /mnt/storage/aryan/VideoExplain/venv/bin/python -u run_demo_eval.py stage3
log "stage 3 finished; restoring gemma + Llama"
cd $VX && GEMMA=1 ./start_all.sh; until up 8002; do sleep 20; done
LLAMA=1 ./start_all.sh; until up 8003; do sleep 20; done
log "gemma + Llama serving again"
