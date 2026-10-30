#!/usr/bin/env bash
# Start the generator and verifier as two vLLM OpenAI-compatible endpoints.
#
# Default layout (this machine: ONE A100 40GB, driver 550 / CUDA 12.4):
#   * vLLM comes from venv-vllm/ (vLLM 0.31.0 cu129 build; the cu130 build in venv38 needs driver >= 580).
#   * Both servers share GPU 0 with sleep mode: the pipeline client puts the idle model to sleep
#     (weights -> CPU RAM, KV cache freed) before calling the other one (model_loader.share_gpu).
#   * Qwen3.8-27B is 52GB in BF16, so it is quantized to FP8 at load time (~27GB). The verifier stays BF16.
# On a node with two big GPUs set GEN_GPU=0 VER_GPU=1 SHARE_GPU=0 GEN_QUANTIZATION= and run the pipeline
# with --no-gpu-swap.
set -euo pipefail
cd "$(dirname "$0")"

VLLM_BIN="${VLLM_BIN:-$PWD/venv-vllm/bin/vllm}"
MODEL_ROOT="${MODEL_ROOT:-$PWD/models}"
GEN_MODEL_PATH="${GEN_MODEL_PATH:-$MODEL_ROOT/Qwen3.8-27B}"
VER_MODEL_PATH="${VER_MODEL_PATH:-$MODEL_ROOT/Qwen3.5-9B}"
GEN_MODEL_NAME="${GEN_MODEL_NAME:-Qwen3.8-27B}"
VER_MODEL_NAME="${VER_MODEL_NAME:-Qwen3.5-9B}"
GEN_PORT="${GEN_PORT:-8000}"
VER_PORT="${VER_PORT:-8001}"
GEN_GPU="${GEN_GPU:-0}"
VER_GPU="${VER_GPU:-0}"
SHARE_GPU="${SHARE_GPU:-1}"                      # 1 = sleep-mode swapping on one GPU
DTYPE="${DTYPE:-bfloat16}"
GEN_QUANTIZATION="${GEN_QUANTIZATION:-fp8}"      # empty = serve BF16 (needs ~60GB+ GPU)
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.78}"   # leaves room for the sleeping verifier (~2.9GB) + other GPU users when it wakes
VER_GPU_MEMORY_UTILIZATION="${VER_GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# The pipeline sends one request at a time. Qwen3.5 is a hybrid (linear-attention) model whose per-sequence
# recurrent state is reserved for every possible sequence, so the vLLM default (hundreds) does not fit 40GB.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"   # smaller prefill chunks -> lower activation peak
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"

[[ -x "$VLLM_BIN" ]] || { echo "ERROR: vllm not found at $VLLM_BIN (set VLLM_BIN)" >&2; exit 1; }
export PATH="$(dirname "$VLLM_BIN"):$PATH"   # ninja/nvcc used by torch.compile live next to vllm
# JIT kernels (flashinfer) must be compiled with a toolkit matching the cu129 wheels, not the default CUDA 12.4.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
[[ -d "$GEN_MODEL_PATH" ]] || { echo "ERROR: generator model path not found: $GEN_MODEL_PATH" >&2; exit 1; }
[[ -d "$VER_MODEL_PATH" ]] || { echo "ERROR: verifier model path not found: $VER_MODEL_PATH" >&2; exit 1; }
n_gpus=$(nvidia-smi -L | wc -l)
for g in "$GEN_GPU" "$VER_GPU"; do
  (( g < n_gpus )) || { echo "ERROR: GPU $g does not exist (this machine has $n_gpus GPU(s))" >&2; exit 1; }
done

common=(--host 0.0.0.0 --dtype "$DTYPE" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
if [[ "$SHARE_GPU" == 1 ]]; then
  export VLLM_SERVER_DEV_MODE=1                          # exposes /sleep, /wake_up, /is_sleeping
  common+=(--enable-sleep-mode)
fi
gen_extra=(--language-model-only)   # generator is text-only: the vision encoder is not loaded
# A sleeping server still holds its CUDA graphs / non-cumem buffers on the GPU. The verifier only writes short
# JSON reviews, so it runs eager (no graphs) to keep its sleeping footprint small enough for the generator to wake.
ver_extra=(${VER_EXTRA_ARGS---enforce-eager})
# VER_VISION=1: the verifier keeps its vision encoder (Qwen3.5-9B is natively multimodal) to ground storyboard
# figures (FIG_xxxx) - boxes for the parts on_screen asks to annotate - and to look at rendered figure frames.
VER_VISION="${VER_VISION:-1}"
if [[ "$VER_VISION" == 1 ]]; then
  ver_extra+=(--limit-mm-per-prompt '{"image":2,"video":0}')
else
  ver_extra+=(--language-model-only)
fi
[[ -n "$GEN_QUANTIZATION" ]] && gen_extra+=(--quantization "$GEN_QUANTIZATION")

wait_ready() {  # port pid name
  local t=0
  until curl -sf "http://127.0.0.1:$1/health" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "ERROR: $3 exited during startup; see logs/vllm-$3.log" >&2; tail -20 "logs/vllm-$3.log" >&2; exit 1; }
    (( t >= STARTUP_TIMEOUT )) && { echo "ERROR: $3 not ready after ${STARTUP_TIMEOUT}s" >&2; exit 1; }
    sleep 5; t=$((t + 5))
  done
  echo "$3 ready on :$1 (${t}s)"
}

mkdir -p logs
GEN_PID= VER_PID=
cleanup() { kill $GEN_PID $VER_PID 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "Starting generator $GEN_MODEL_NAME on GPU $GEN_GPU, port $GEN_PORT (quantization: ${GEN_QUANTIZATION:-none})"
CUDA_VISIBLE_DEVICES="$GEN_GPU" "$VLLM_BIN" serve "$GEN_MODEL_PATH" --served-model-name "$GEN_MODEL_NAME" \
  --port "$GEN_PORT" --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" "${common[@]}" "${gen_extra[@]}" \
  >"logs/vllm-generator.log" 2>&1 &
GEN_PID=$!
if [[ "$SHARE_GPU" == 1 ]]; then
  # The verifier can only allocate its memory once the generator has released the GPU.
  wait_ready "$GEN_PORT" "$GEN_PID" generator
  curl -sf -X POST "http://127.0.0.1:$GEN_PORT/sleep?level=1" >/dev/null && echo "generator put to sleep"
fi

echo "Starting verifier $VER_MODEL_NAME on GPU $VER_GPU, port $VER_PORT"
CUDA_VISIBLE_DEVICES="$VER_GPU" "$VLLM_BIN" serve "$VER_MODEL_PATH" --served-model-name "$VER_MODEL_NAME" \
  --port "$VER_PORT" --gpu-memory-utilization "$VER_GPU_MEMORY_UTILIZATION" "${common[@]}" "${ver_extra[@]}" \
  >"logs/vllm-verifier.log" 2>&1 &
VER_PID=$!
wait_ready "$VER_PORT" "$VER_PID" verifier
[[ "$SHARE_GPU" == 1 ]] || wait_ready "$GEN_PORT" "$GEN_PID" generator

echo "Generator: http://127.0.0.1:$GEN_PORT/v1 (PID $GEN_PID)"
echo "Verifier:  http://127.0.0.1:$VER_PORT/v1 (PID $VER_PID)"
echo "Logs: logs/vllm-generator.log and logs/vllm-verifier.log. Ctrl-C stops both."
wait
