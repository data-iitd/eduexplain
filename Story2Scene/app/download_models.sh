#!/usr/bin/env bash
# Download the models of every stage into its own models/ directory (idempotent; run inside tmux).
#   ./download_models.sh            # everything that is still missing
# Gated models (Llama, Gemma) need: `hf auth login` once, and accepting the licence on the model page.
set -uo pipefail
HF=/mnt/storage/aryan/VideoModule/venv-vllm/bin/hf
LOG=/mnt/storage/aryan/VideoExplain/logs/downloads.log
mkdir -p "$(dirname "$LOG")"

# repo                                         target dir                                                    needed by
MODELS=(
  "BAAI/bge-large-en-v1.5                      /mnt/storage/aryan/setCore/models/bge-large-en-v1.5           stage1 retrieval (downloaded)"
  "sentence-transformers/all-roberta-large-v1  /mnt/storage/aryan/set2story/models/all-roberta-large-v1      stage2 encoder (downloaded)"
  "cross-encoder/nli-deberta-v3-large          /mnt/storage/aryan/set2story/models/nli-deberta-v3-large      stage2 NLI (also in the HF cache)"
  "google/gemma-3-12b-it                       /mnt/storage/aryan/set2story/models/gemma-3-12b-it            stage2 original LLM (GATED; optional: Qwen3.8-27B is used meanwhile)"
  "meta-llama/Llama-3.1-70B-Instruct           SKIP                                                           stage1 original selector (GATED, ~141 GB BF16: cannot run on one 40 GB A100; Qwen3.8-27B is used)"
)

$HF auth whoami >/dev/null 2>&1 && echo "HF login: $($HF auth whoami 2>/dev/null | head -1)" || echo "HF login: NOT logged in (gated models will fail; run: $HF auth login)"
for row in "${MODELS[@]}"; do
  read -r repo dir _ <<<"$row"
  note=$(echo "$row" | awk '{for (i=3; i<=NF; i++) printf "%s ", $i}')
  if [[ "$dir" == SKIP ]]; then echo "skip  $repo  - $note"; continue; fi
  if [[ -n "$(ls -A "$dir" 2>/dev/null | grep -E 'safetensors|\.bin$|onnx')" ]]; then echo "ok    $repo  -> $dir"; continue; fi
  echo "get   $repo  -> $dir  ($note)"
  if $HF download "$repo" --local-dir "$dir" >>"$LOG" 2>&1; then
    echo "      done"
  else
    echo "      FAILED (gated? accept the licence at https://huggingface.co/$repo and run '$HF auth login'); see $LOG"
  fi
done
