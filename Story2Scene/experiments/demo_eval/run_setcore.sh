#!/usr/bin/env bash
# Stage 1, SetCoRe-LW and SetCoRe-E2E from the SetCoRe repository (anonymous.4open.science/r/setcore-37E8),
# seed 0, grouped 5-fold CV over the 321 gold queries, on the shared BGE top-100 pool. CPU only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false
EXP=/mnt/storage/aryan/VideoExplain/experiments/demo_eval
REPO=/mnt/storage/aryan/external/setcore-37E8
S=/mnt/storage/aryan/setCore
PY=/mnt/storage/aryan/set2story/venv/bin/python
R=$S/all_resources_unique_ids_clean1_merged.json
Q=$S/gold_sets_final_validated_new_merged.json
T=$S/final_training_set.json
BGE=$S/models/bge-large-en-v1.5
POOL=$EXP/pools/bge_matched_top100.json
if [ ! -s $POOL ]; then
  (cd $REPO/baselines && $PY bge_retriever.py --gold $Q --resources $R --bge-model $BGE --top-k 100 \
     --output $POOL)
fi
cd $REPO
if [ ! -s $EXP/stage1/lw_s0/predictions.jsonl ]; then
  $PY -m model.pipeline run --resources $R --queries $Q --train-queries $T --candidates $POOL \
    --role-train cv --uq-mode match --kappa 8 --con-source learned --con-group chapter \
    --con-learned-size topm --con-learned-tau 0.7 --res-concept-hidden 512 --res-concept-lr 1e-3 \
    --res-concept-wd 1e-5 --res-concept-epochs 400 --seed 0 \
    --encoder-name $BGE --cache-dir $EXP/cache --out $EXP/stage1/lw_s0
fi
if [ ! -s $EXP/stage1/e2e_s0/predictions.jsonl ]; then
  $PY -m model.pipeline run --resources $R --queries $Q --candidates $POOL \
    --role-train cv --selector e2e --e2e-cv \
    --e2e-d-model 128 --e2e-layers 1 --e2e-dropout 0.3 --e2e-lr 1e-4 \
    --e2e-gold-match best --e2e-size-source role --seed 0 \
    --encoder-name $BGE --cache-dir $EXP/cache --out $EXP/stage1/e2e_s0
fi
echo SETCORE DONE
