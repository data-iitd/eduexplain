"""Config + persistence helpers for serving the setCore lightweight model on unseen queries.

Training (once):  python scx_lw_train.py            -> models/lw/role_head_s0.npz
Serving:          backend/setcore_worker.py (loads Index from caches + the saved role head)
"""
import json, os, sys
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
LW = os.path.join(S, "models", "lw")
sys.path.append(S)   # append: icra/select.py must not shadow the stdlib

RES = os.path.join(S, "all_resources_unique_ids_clean1_merged.json")
GOLD = os.path.join(S, "gold_sets_final_validated_new_merged.json")
TRAIN = os.path.join(S, "final_training_set.json")
BGE = os.path.join(S, "models", "bge-large-en-v1.5")
ROLE = os.path.join(LW, "role_head_s0.npz")
# EXACTLY run_cardinality.sh's CONCEPT flags (+ seed 0, K 100, encoder, cache dir)
CONCEPT = dict(uq_mode="match", kappa=8, con_source="learned", con_group="chapter",
               con_learned_size="topm", con_learned_tau=0.7, res_concept_hidden=512,
               res_concept_lr=1e-3, res_concept_wd=1e-5, res_concept_epochs=400)


def make_cfg(queries_path, train_path=None):
    from icra.config import Config
    return Config(resources_path=RES, queries_path=queries_path, train_queries_path=train_path,
                  encoder_name=BGE, cache_dir=os.path.join(LW, "cache"), seed=0, K=100,
                  role_train="cv", **CONCEPT)


def tiny_queries_file():
    p = os.path.join(LW, "inference_queries.json")
    if not os.path.exists(p):
        with open(GOLD, encoding="utf-8") as f:
            json.dump(json.load(f)[:1], open(p, "w", encoding="utf-8"))
    return p


def save_role(head, meta):
    np.savez(ROLE, meta=json.dumps(meta), **head.P)


def load_role(in_dim, cfg):
    from icra.heads import RoleHead
    head = RoleHead(in_dim, cfg.n_max, cfg.role_hidden, cfg.role_lr, cfg.role_weight_decay,
                    cfg.role_epochs, cfg.lam, cfg.role_loss, cfg.seed)
    z = np.load(ROLE)
    head.P = {k: z[k] for k in head.P}
    return head
