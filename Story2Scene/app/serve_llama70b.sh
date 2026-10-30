#!/usr/bin/env bash
# setCore's original selector, Llama-3.1-70B-Instruct, as a fourth sleepable vLLM endpoint on :8003.
# 70B in BF16 is ~141 GB, so it runs as the AWQ INT4 checkpoint (hugging-quants, ~39.8 GB) with the part that does
# not fit the A100 40GB kept in CPU RAM (--cpu-offload-gb). Stage 1's selection writes only a few ids, so the slower
# offloaded decode is affordable.
# It sleeps at LEVEL 2 (weights dropped, reloaded from disk on wake): a level-1 sleep would park ~35 GB of pinned
# RAM next to the other three sleeping models, which this machine does not have. Clients learn the level from
# VLLM_SLEEP_LEVELS (model_loader.sleep_level_for). Start only while :8000/:8001/:8002 are asleep.
set -euo pipefail
VM=/mnt/storage/aryan/VideoModule
export VLLM_SERVER_DEV_MODE=1 CUDA_VISIBLE_DEVICES="${LLAMA_GPU:-0}" CUDA_HOME=/usr/local/cuda-12.9
export PATH="$VM/venv-vllm/bin:$CUDA_HOME/bin:$PATH"
# Level-2 wake re-reads the weights from this path. The local-disk copy reads at ~560 MB/s (~70 s per wake) against
# ~85 MB/s from NFS (~7 min), so it is used when present:
#   rsync -a --exclude .cache /mnt/storage/aryan/setCore/models/Llama-3.1-70B-Instruct-AWQ-INT4 /var/tmp/vx-models/
MODEL=/mnt/storage/aryan/setCore/models/Llama-3.1-70B-Instruct-AWQ-INT4
LOCAL=/var/tmp/vx-models/Llama-3.1-70B-Instruct-AWQ-INT4
[[ -f "$LOCAL/model.safetensors.index.json" ]] && MODEL=$LOCAL
exec vllm serve "$MODEL" \
  --served-model-name Llama-3.1-70B-Instruct --host 0.0.0.0 --port "${LLAMA_PORT:-8003}" \
  --quantization awq_marlin --dtype float16 --max-model-len "${LLAMA_MAX_LEN:-8192}" \
  --max-num-seqs 2 --max-num-batched-tokens 2048 --enable-sleep-mode --enforce-eager \
  --cpu-offload-gb "${LLAMA_CPU_OFFLOAD_GB:-18}" \
  --gpu-memory-utilization "${LLAMA_GPU_MEMORY_UTILIZATION:-0.79}"
