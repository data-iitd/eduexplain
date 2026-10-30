#!/usr/bin/env bash
# Start the whole VideoExplain stack in tmux (survives SSH disconnects):
#   vx-vllm      vLLM generator :8000 + vision verifier :8001 (only if not already serving)
#   vx-backend   FastAPI pipeline API :8800
#   vx-frontend  Vite dev server :5173 (proxies /api to :8800)
#   vx-dummy     optional dummy OpenAI endpoint :8009 (VX_DUMMY=1)
#   vx-gemma     optional gemma-3-12b-it :8002 for Stage 2 (GEMMA=1)
#   vx-llama     optional Llama-3.1-70B-Instruct :8003 for Stage 1 (LLAMA=1)
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
mkdir -p logs
up() { curl -sf --max-time 5 "http://127.0.0.1:$1/health" >/dev/null || curl -sf --max-time 5 "http://127.0.0.1:$1/v1/models" >/dev/null; }

if up 8000 && up 8001; then
  echo "vLLM already serving on :8000/:8001"
elif up 8000 || up 8001; then
  echo "WARNING: only one vLLM server is up; not starting serve_models.sh (it would collide). Check ports 8000/8001."
else
  tmux has-session -t vx-vllm 2>/dev/null || \
    tmux new-session -d -s vx-vllm "cd /mnt/storage/aryan/VideoModule && MODEL_ROOT=/mnt/storage/aryan/VideoModule/models ./serve_models.sh 2>&1 | tee $ROOT/logs/vllm.log"
  echo "starting vLLM in tmux session vx-vllm (~10 min until both servers are ready)"
fi

if [[ "${VX_DUMMY:-0}" == 1 ]] && ! tmux has-session -t vx-dummy 2>/dev/null; then
  tmux new-session -d -s vx-dummy "cd $ROOT/backend && $ROOT/venv/bin/python dummy_llm.py --port 8009 2>&1 | tee $ROOT/logs/dummy.log"
fi
# set2story's original LLM (gemma-3-12b-it) on :8002, a third sleepable server on the shared GPU (GEMMA=1 starts it;
# it must start while :8000/:8001 are asleep). When it serves, Stage 2 uses it and every client sleeps it first.
# It sleeps at level 2 (reload ~1-4 min once per job) so its ~25 GB are not pinned in RAM.
if [[ "${GEMMA:-0}" == 1 ]] && ! up 8002 && ! tmux has-session -t vx-gemma 2>/dev/null; then
  for p in 8000 8001; do curl -s -X POST --max-time 120 "http://127.0.0.1:$p/sleep?level=1" >/dev/null; done
  curl -s -X POST --max-time 120 "http://127.0.0.1:8003/sleep?level=2" >/dev/null
  tmux new-session -d -s vx-gemma "$ROOT/serve_gemma.sh 2>&1 | tee $ROOT/logs/vllm-gemma.log"
  echo "starting gemma-3-12b-it on :8002 (tmux vx-gemma, ~3-5 min); restart the backend once it is ready"
fi
# setCore's original selector (Llama-3.1-70B-Instruct, AWQ INT4 + CPU offload) on :8003, a fourth sleepable server
# that sleeps at LEVEL 2 (weights reloaded from disk on wake; no room to park them in RAM). LLAMA=1 starts it; like
# gemma it must start while the others are asleep. When it serves, Stage 1's LLM selection uses it.
if [[ "${LLAMA:-0}" == 1 ]] && ! up 8003 && ! tmux has-session -t vx-llama 2>/dev/null; then
  for p in 8000 8001; do curl -s -X POST --max-time 120 "http://127.0.0.1:$p/sleep?level=1" >/dev/null; done
  curl -s -X POST --max-time 120 "http://127.0.0.1:8002/sleep?level=2" >/dev/null
  tmux new-session -d -s vx-llama "$ROOT/serve_llama70b.sh 2>&1 | tee $ROOT/logs/vllm-llama70b.log"
  echo "starting Llama-3.1-70B on :8003 (tmux vx-llama, ~3-8 min); restart the backend once it is ready"
fi
# Every sleepable server on the GPU goes into VLLM_GPU_PEERS (each client sleeps all others before generating).
PEER_URLS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"
S2S_SLEEP="$PEER_URLS"
ENVS=""
LEVELS=""   # level-2 sleep (weights dropped, reloaded on wake): RAM cannot park all four models
if up 8002; then
  PEER_URLS="$PEER_URLS,http://127.0.0.1:8002/v1"
  LEVELS="http://127.0.0.1:8002/v1=2"
  ENVS="$ENVS S2S_GEN_URL=http://127.0.0.1:8002/v1 S2S_GEN_MODEL=gemma-3-12b-it"
  echo "gemma-3-12b-it serving on :8002 -> Stage 2 (set2story) uses it"
fi
if up 8003; then
  PEER_URLS="$PEER_URLS,http://127.0.0.1:8003/v1"
  S2S_SLEEP="$S2S_SLEEP,http://127.0.0.1:8003/v1"
  LEVELS="${LEVELS:+$LEVELS,}http://127.0.0.1:8003/v1=2"
  ENVS="$ENVS VX_STAGE1_LLM_URL=http://127.0.0.1:8003/v1 VX_STAGE1_LLM_MODEL=Llama-3.1-70B-Instruct"
  echo "Llama-3.1-70B-Instruct serving on :8003 -> Stage 1 (setCore selection) uses it"
fi
PEERS=""
if up 8002 || up 8003; then
  PEERS="export VLLM_GPU_PEERS=$PEER_URLS S2S_SLEEP_URLS=$S2S_SLEEP VLLM_SLEEP_LEVELS=$LEVELS $ENVS;"
fi
tmux has-session -t vx-backend 2>/dev/null || \
  tmux new-session -d -s vx-backend "$PEERS cd $ROOT/backend && while true; do $ROOT/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8800 2>&1 | tee -a $ROOT/logs/backend.log; echo 'backend exited; restarting in 5 s' | tee -a $ROOT/logs/backend.log; sleep 5; done"
tmux has-session -t vx-frontend 2>/dev/null || \
  tmux new-session -d -s vx-frontend "cd $ROOT/frontend && while true; do PATH=/home/aryan/.local/node/bin:\$PATH npm run dev -- --host 0.0.0.0 --port 5173 2>&1 | tee -a $ROOT/logs/frontend.log; sleep 5; done"
tmux ls
echo "Website: http://$(hostname -I | awk '{print $1}'):5173   (API: :8800/api/health)"
