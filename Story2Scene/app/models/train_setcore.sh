#!/usr/bin/env bash
# Train the two served SetCoRe checkpoints for the web app (setcore-37E8/model/serve.py, presets = paper rows). CPU only.
#   LW  : --fit-on train (the 321 benchmark queries stay held out)
#   E2E : --fit-on all   (serve.py's deployment example)
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false
S=/mnt/storage/aryan/setCore
PY=/mnt/storage/aryan/set2story/venv/bin/python
OUT=/mnt/storage/aryan/VideoExplain/models
ARGS="--resources $S/all_resources_unique_ids_clean1_merged.json --queries $S/gold_sets_final_validated_new_merged.json
      --train-queries $S/final_training_set.json --encoder-name $S/models/bge-large-en-v1.5
      --cache-dir /mnt/storage/aryan/VideoExplain/experiments/demo_eval/cache --seed 0"
cd /mnt/storage/aryan/external/setcore-37E8
[ -s $OUT/setcore_lw/manifest.json ]  || $PY -m model.serve train --model-type lightweight $ARGS --fit-on train --out $OUT/setcore_lw
[ -s $OUT/setcore_e2e/manifest.json ] || $PY -m model.serve train --model-type e2e $ARGS --fit-on all --out $OUT/setcore_e2e
echo TRAIN DONE
