"""Train ONE final class-balanced RoleSequencePredictor on all queries (CPU). Run with the set2story venv python.

Mirrors set2story/training/train_cv.py (best-match loss, class-balanced weights, Adam, batch 16, role_emb_dim 32,
early stopping on RS-F1/RSS/macro-F1 selection score, patience 20, max 150 epochs). The CV grid winner is not stored,
so a small grid {lr, hidden, dropout} x CB beta is searched on a 15% stratified validation split, then the best config
is retrained on ALL data for the best epoch count. Writes models/role_predictor_final.pt and role_fallback.json.
"""
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/storage/aryan/set2story")
for d in ("Methodology", "Metrics", "Baseline", "training", "Runs"):
    sys.path.insert(0, str(ROOT / d))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from Role_Metrics import ROLES, best_match, canonical_role, role_presence_stats
from role_predictor import RoleSequencePredictor, best_match_loss, class_balanced_weights
from encoder import SentenceTransformerEncoder

MODELS = ROOT / "models"
FIXED = {"role_emb_dim": 32, "batch_size": 16, "max_epochs": 150, "patience": 20, "val_frac": 0.15}
GRID = {"lr": [1e-3, 3e-3], "hidden": [128, 256], "dropout": [0.0, 0.2]}
CB_BETAS = [0.99, 0.999, 0.9999]


def load():
    qs = json.load(open(ROOT / "inputs/Test_Set.json"))
    refs = {}
    for it in json.load(open(ROOT / "inputs/Reference_Storyboard.json")):
        sb = it["storyboard"]
        segs = sorted(sb["segments"], key=lambda s: s.get("order", 0))
        refs.setdefault(it["query_id"], []).append([canonical_role(s["role"]) for s in segs])
    qs = [q for q in qs if q["query_id"] in refs]
    return qs, [refs[q["query_id"]] for q in qs]


def scores(preds, refs_list):
    f1s, rss, pairs = [], [], []
    for p, refs in zip(preds, refs_list):
        m = best_match(p, refs)
        f1s.append(m["rs_f1"]); rss.append(m["rss"]); pairs.append((p, m["rss_ref"]))
    mac = role_presence_stats(pairs)["macro_f1"]
    return (float(np.mean(f1s)) + float(np.mean(rss)) + mac) / 3.0


def counts(R):
    c = Counter()
    for refs in R:
        for s in refs:
            c.update(s)
    return c


def train(cfg, Xtr, Rtr, Xva=None, Rva=None, seed=0, epochs=None):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    w = class_balanced_weights(counts(Rtr), ROLES, cfg["beta"])
    m = RoleSequencePredictor(ROLES, d=Xtr.shape[1], hidden=cfg["hidden"],
                              role_emb_dim=FIXED["role_emb_dim"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
    X = torch.tensor(Xtr)
    best, bad = (-1, None, 0), 0
    for ep in range(epochs or FIXED["max_epochs"]):
        m.train()
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(perm), FIXED["batch_size"]):
            idx = perm[s:s + FIXED["batch_size"]]
            opt.zero_grad()
            best_match_loss(m, X[idx], [Rtr[i] for i in idx], w).backward()
            opt.step()
        if Xva is not None:
            m.eval()
            with torch.no_grad():
                preds = [m.decode_greedy(z) for z in m.encode_query(torch.tensor(Xva))]
            sc = scores(preds, Rva)
            if sc > best[0] + 1e-6:
                best, bad = (sc, None, ep + 1), 0
            else:
                bad += 1
                if bad >= FIXED["patience"]:
                    break
    return m, best


def main():
    t0 = time.time()
    torch.set_num_threads(8)
    qs, R = load()
    enc = SentenceTransformerEncoder(str(MODELS / "all-roberta-large-v1"), device="cpu")
    X = enc.encode([q["query"] for q in qs]).astype(np.float32)
    print(f"{len(qs)} queries, {sum(map(len, R))} refs, X={X.shape} [{time.time()-t0:.0f}s]", flush=True)
    qtype = np.array([q.get("query_type", "na") for q in qs])
    tr, va = train_test_split(np.arange(len(qs)), test_size=FIXED["val_frac"], stratify=qtype, random_state=0)
    best = (-1, None, 0)
    for v in itertools.product(*GRID.values()):
        for b in CB_BETAS:
            cfg = {**dict(zip(GRID, v)), "beta": b}
            _, (sc, _, ep) = train(cfg, X[tr], [R[i] for i in tr], X[va], [R[i] for i in va])
            print(f"  {cfg} val={sc:.4f} epochs={ep} [{time.time()-t0:.0f}s]", flush=True)
            if sc > best[0]:
                best = (sc, cfg, ep)
    sc, cfg, ep = best
    print("best", cfg, sc, "epochs", ep, flush=True)
    model, _ = train(cfg, X, R, seed=0, epochs=max(ep, 5))
    model.eval()
    torch.save({"state_dict": model.state_dict(), "cfg": {**cfg, **FIXED, "d": int(X.shape[1])},
                "roles": list(ROLES), "val_score": sc}, MODELS / "role_predictor_final.pt")
    freq = Counter(tuple(s) for refs in R for s in refs)
    tmpl = list(freq.most_common(1)[0][0])
    json.dump({"role_template": tmpl, "count": freq.most_common(1)[0][1]}, open(MODELS / "role_fallback.json", "w"))
    print("fallback template", tmpl)
    with torch.no_grad():
        for i in (0, 100, 200):
            print(qs[i]["query"], "->", model.decode_greedy(model.encode_query(torch.tensor(X[i]))),
                  "| ref:", R[i][0])
    print(f"TRAIN_TIME {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
