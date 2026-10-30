#!/usr/bin/env bash
# set2story's original LLM, gemma-3-12b-it (BF16, ~24 GB), as a third sleepable vLLM endpoint on :8002.
# It shares the A100 with the Qwen generator (:8000) and verifier (:8001): every client puts the other servers to
# sleep before it generates (VLLM_GPU_PEERS / S2S_SLEEP_URLS), so only one model is resident at a time.
# Start only while :8000 and :8001 are asleep (they hold ~1.2 GB then).
set -euo pipefail
VM=/mnt/storage/aryan/VideoModule
export VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES="${GEMMA_GPU:-0}" CUDA_HOME=/usr/local/cuda-12.9
export PATH="$VM/venv-vllm/bin:$CUDA_HOME/bin:$PATH"
exec vllm serve /mnt/storage/aryan/set2story/models/gemma-3-12b-it --served-model-name gemma-3-12b-it \
  --host 0.0.0.0 --port "${GEMMA_PORT:-8002}" --dtype bfloat16 --max-model-len "${GEMMA_MAX_LEN:-8192}" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --enable-sleep-mode --enforce-eager --language-model-only \
  --gpu-memory-utilization "${GEMMA_GPU_MEMORY_UTILIZATION:-0.72}"
