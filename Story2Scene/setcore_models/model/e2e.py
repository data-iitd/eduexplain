"""End-to-end variant: a pointer-style set predictor trained with the
best-match (min over gold sets) bipartite loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

TYPE_EXPLANATION, TYPE_FIGURE, TYPE_NULL = 0, 1, 2
N_TYPE_CLASSES = 3


# --------------------------------------------------------------------- model
class SetPredictor(nn.Module):
    """Slot decoder + pointer over the candidate pool."""

    def __init__(self, d_in: int, d_model: int = 256, n_slots: int = 8,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2,
                 n_extra: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
                   
        self.cand_proj = nn.Sequential(
            nn.Linear(d_in + n_extra, d_model), nn.LayerNorm(d_model))
        self.q_proj = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LayerNorm(d_model))
        self.slot_emb = nn.Parameter(torch.randn(n_slots, d_model))
        self.slot_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.ptr = nn.Linear(d_model, d_model)
        self.null = nn.Parameter(torch.randn(d_model) * 0.02)
        self.type_head = nn.Linear(d_model, N_TYPE_CLASSES)
        self.scale = d_model ** -0.5

    def forward(self, cand: torch.Tensor, q: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cand [B,K,d_in+n_extra]; q [B,d_in]; mask [B,K] True = real candidate.

        Returns pointing logits [B, n_slots, K+1] (last column is NULL) and
        slot type logits [B, n_slots, 3].
        """
        C = self.cand_proj(cand)
        slots = self.slot_norm(self.slot_emb.unsqueeze(0) + self.q_proj(q).unsqueeze(1))
        out = self.dec(slots, C, memory_key_padding_mask=~mask)
        p = self.ptr(out)
        logits = torch.einsum("bnd,bkd->bnk", p, C) * self.scale
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)
        null_logit = (p @ self.null) * self.scale
        return torch.cat([logits, null_logit.unsqueeze(-1)], dim=-1), self.type_head(out)


# ----------------------------------------------------------------- batching
@dataclass
class QueryExample:
    q_emb: np.ndarray                 # [d_in]
    cand_emb: np.ndarray              # [K, d_in]
    cand_extra: np.ndarray            # [K, 4]
    pool_ids: List[str]
    gold_pos: List[List[int]]         # per gold set: positions within the pool
    gold_type: List[List[int]]        # per gold set: TYPE_* of each gold element
    n_gold_total: List[int]           # per gold set: size BEFORE pool filtering


def build_example(ix, q, q_emb: np.ndarray, pool: np.ndarray,
                  s: np.ndarray) -> QueryExample:
    """Assemble one training/inference example from an Index and a query."""
    rank = np.arange(len(pool), dtype=np.float32) / max(1, len(pool) - 1)
    lo, hi = float(s.min()), float(s.max())
    s_n = (s - lo) / (hi - lo) if hi - lo > 1e-9 else np.full_like(s, 0.5)
    types = ix.types[pool]
    extra = np.stack([
        (types == "explanation").astype(np.float32),
        (types == "figure").astype(np.float32),
        s_n.astype(np.float32),
        rank,
    ], axis=1)

    pos_of = {rid: j for j, rid in enumerate(
        [ix.resources[i].resource_id for i in pool])}
    gold_pos, gold_type, n_tot = [], [], []
    for g in q.gold_sets:
        ps, ts = [], []
        for x in g:
            if x in pos_of:
                ps.append(pos_of[x])
                ts.append(TYPE_FIGURE if ix.resources[ix.rid2i[x]].type == "figure"
                          else TYPE_EXPLANATION)
        gold_pos.append(ps)
        gold_type.append(ts)
        n_tot.append(len(g))
    return QueryExample(q_emb, ix.R[pool], extra,
                        [ix.resources[i].resource_id for i in pool],
                        gold_pos, gold_type, n_tot)


def collate(batch: Sequence[QueryExample], device) -> Tuple[torch.Tensor, ...]:
    B = len(batch)
    K = max(len(b.pool_ids) for b in batch)
    d_in = batch[0].cand_emb.shape[1]
    n_ex = batch[0].cand_extra.shape[1]
    cand = np.zeros((B, K, d_in + n_ex), dtype=np.float32)
    mask = np.zeros((B, K), dtype=bool)
    qs = np.stack([b.q_emb for b in batch]).astype(np.float32)
    for i, b in enumerate(batch):
        k = len(b.pool_ids)
        cand[i, :k] = np.concatenate([b.cand_emb, b.cand_extra], axis=1)
        mask[i, :k] = True
    return (torch.from_numpy(cand).to(device),
            torch.from_numpy(qs).to(device),
            torch.from_numpy(mask).to(device))


# --------------------------------------------------------------------- loss
def best_match_loss(logits: torch.Tensor, type_logits: torch.Tensor,
                    ex: QueryExample, lam_type: float,
                    null_weight: float = 1.0,
                    normalize: bool = True,
                    single_gold: bool = False,
                    gold_match: str = "best",
                    rng: Optional["np.random.Generator"] = None
                    ) -> Optional[torch.Tensor]:
    """Eq. (best-match): min over gold sets of the Hungarian-matched loss.
    """
    n = logits.shape[0]
    K = logits.shape[-1] - 1
    logp = F.log_softmax(logits, dim=-1)
    logt = F.log_softmax(type_logits, dim=-1)
    best = None
    losses = []
    pairs = list(zip(ex.gold_pos, ex.gold_type))

    # How the m gold sets are aggregated into one scalar:
    #   best          min over gold sets (the paper's Eq. 9), the training
    #                 analogue of the max-over-gold-sets evaluation
    #   single_first  the first gold set only
    #   single_random a gold set drawn uniformly, re-drawn every epoch, so over
    #                 training the model sees all of them but never gets to
    #                 choose which one to satisfy
    #   average       the mean over gold sets -- the model is pushed toward
    #                 every acceptable set at once, which is a different
    #                 objective: the minimiser is a blend, not a member
    if gold_match == "single_random" and len(pairs) > 1:
        g = rng if rng is not None else np.random.default_rng(0)
        j = int(g.integers(len(pairs)))
        pairs = pairs[j:j + 1]
    elif gold_match == "single_first" or single_gold:
        pairs = pairs[:1]
    for ps, ts in pairs:
        if not ps:
            continue                                  
        cost = (-logp[:, ps]).detach().float().cpu().numpy()
        ri, ci = linear_sum_assignment(cost)
        pos, neg, matched = [], [], set()
        for r, c in zip(ri, ci):
            pos.append(-logp[r, ps[c]] - lam_type * logt[r, ts[c]])
            matched.add(int(r))
        for sl in range(n):
            if sl not in matched:
                neg.append(-logp[sl, K] - lam_type * logt[sl, TYPE_NULL])
        if normalize:
            li = torch.stack(pos).mean()
            if neg:
                li = li + null_weight * torch.stack(neg).mean()
        else:
            li = (torch.stack(pos).sum()
                  + (null_weight * torch.stack(neg).sum() if neg else 0.0)) / n
        losses.append(li)
        best = li if best is None else torch.minimum(best, li)
    if not losses:
        return None
    if gold_match == "average":
        return torch.stack(losses).mean()
    return best


# ---------------------------------------------------------------- inference
def decode(logits: torch.Tensor, n_max: int, type_logits: Optional[torch.Tensor] = None,
           mode: str = "type", fire_threshold: float = 0.5,
           fixed_size: int = 0, profile: Optional[Dict[str, int]] = None,
           pool_types: Optional[Sequence[str]] = None) -> List[int]:
    """Slot outputs -> a set of pool positions.

    mode="joint" takes the argmax of the merged (K+1)-way pointing softmax, so
    a slot emits nothing when NULL wins. 

    mode="type" (default) separates the two decisions, as DETR does: the 3-way
    type head decides whether the slot fires, and the pointer decides where it
    points, ranging over candidates only. P(fire) = 1 - P(NULL).
    """
    K = logits.shape[-1] - 1
    p = F.softmax(logits, dim=-1)

    if profile is not None and pool_types is not None:
        if mode == "type" and type_logits is not None:
            fire = 1.0 - F.softmax(type_logits, dim=-1)[:, TYPE_NULL]
            score = p[:, :K] * fire[:, None]
        else:
            score = p[:, :K]
        remaining = {t: int(n) for t, n in profile.items()}
        out, seen = [], set()
        for t in torch.argsort(score.reshape(-1), descending=True).tolist():
            j = int(t % K)
            if j in seen or j >= len(pool_types):
                continue
            ty = str(pool_types[j])
            if remaining.get(ty, 0) <= 0:
                continue
            seen.add(j)
            out.append(j)
            remaining[ty] -= 1
            if sum(remaining.values()) <= 0:
                break
        return out

    if fixed_size and fixed_size > 0:
        if mode == "type" and type_logits is not None:
            fire = 1.0 - F.softmax(type_logits, dim=-1)[:, TYPE_NULL]
            score = p[:, :K] * fire[:, None]
        else:
            score = p[:, :K]
        out, seen = [], set()
        for t in torch.argsort(score.reshape(-1), descending=True).tolist():
            j = int(t % K)
            if j in seen:
                continue
            seen.add(j)
            out.append(j)
            if len(out) >= min(fixed_size, K):
                break
        return out

    if mode == "type" and type_logits is not None:
        fire = 1.0 - F.softmax(type_logits, dim=-1)[:, TYPE_NULL]
        cand_p, cand_i = p[:, :K].max(dim=-1)
        picks = [(float(fire[s] * cand_p[s]), int(cand_i[s]))
                 for s in range(logits.shape[0]) if float(fire[s]) >= fire_threshold]
    else:
        conf, idx = p.max(dim=-1)
        picks = [(float(conf[s]), int(idx[s])) for s in range(logits.shape[0])
                 if int(idx[s]) != K]
    picks.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, j in picks:
        if j not in seen:
            seen.add(j)
            out.append(j)
        if len(out) >= n_max:
            break
    return out


# ----------------------------------------------------------------- training
class E2ETrainer:
    def __init__(self, cfg, d_in: int, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        torch.manual_seed(cfg.seed)
        self._rng = np.random.default_rng(cfg.seed)
        self._shuffle_rng = np.random.default_rng(cfg.seed + 10_000)
        self.model = SetPredictor(
            d_in, cfg.e2e_d_model, cfg.n_max, cfg.e2e_heads, cfg.e2e_layers,
            cfg.e2e_dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.e2e_lr,
                                     weight_decay=cfg.e2e_weight_decay)

    def _epoch(self, data: List[QueryExample], train: bool) -> float:
        self.model.train(train)
        order = (self._shuffle_rng.permutation(len(data)) if train
                 else np.arange(len(data)))
        bs, tot, nb = self.cfg.e2e_batch_size, 0.0, 0
        for i in range(0, len(order), bs):
            batch = [data[j] for j in order[i:i + bs]]
            cand, q, mask = collate(batch, self.device)
            with torch.set_grad_enabled(train):
                logits, tlog = self.model(cand, q, mask)
                losses = [l for k, b in enumerate(batch)
                          if (l := best_match_loss(logits[k], tlog[k], b,
                                                   self.cfg.e2e_lam_type,
                                                   self.cfg.e2e_null_weight,
                                                   self.cfg.e2e_loss_norm,
                                                   self.cfg.e2e_single_gold,
                                                   self.cfg.e2e_gold_match,
                                                   self._rng))
                          is not None]
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.cfg.e2e_grad_clip)
                    self.opt.step()
            tot += float(loss.detach())
            nb += 1
        return tot / max(1, nb)

    @torch.no_grad()
    def dev_f1(self, data: List[QueryExample]) -> float:
        """Best-match Dice F1 on a dev slice, computed from pool positions.
        """
        preds = self.predict(data)
        out = []
        for ex, pr in zip(data, preds):
            ps = set(pr)
            best = 0.0
            for gp, ntot in zip(ex.gold_pos, ex.n_gold_total):
                den = len(ps) + ntot
                if den:
                    best = max(best, 2 * len(ps & set(gp)) / den)
            out.append(best)
        return float(np.mean(out)) if out else 0.0

    def fit(self, train: List[QueryExample], dev: Optional[List[QueryExample]] = None,
            epochs: Optional[int] = None, tag: str = "", quiet: bool = False):
        """Train with early stopping on a dev slice carved from TRAINING data.

        """
        epochs = epochs or self.cfg.e2e_epochs
        best, best_state, bad = np.inf, None, 0
        self.best_dev, self.best_epoch = None, None
        for ep in range(epochs):
            tr_loss = self._epoch(train, True)
            if dev:
                if self.cfg.e2e_early_stop == "f1":
                    dv = -self.dev_f1(dev)
                else:
                    dv = self._epoch(dev, False)
                if dv < best - 1e-4:
                    best, bad = dv, 0
                    self.best_dev, self.best_epoch = abs(dv), ep
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
                if not quiet and ep % 5 == 0:
                    lbl = "dev F1" if self.cfg.e2e_early_stop == "f1" else "dev loss"
                    print(f"  [{tag}] ep {ep:3d} train {tr_loss:.4f} "
                          f"{lbl} {abs(dv):.4f}")
                if bad >= self.cfg.e2e_patience:
                    if not quiet:
                        print(f"  [{tag}] early stop at epoch {ep} "
                              f"(best dev {abs(best):.4f})")
                    break
            elif not quiet and ep % 5 == 0:
                print(f"  [{tag}] ep {ep:3d} train {tr_loss:.4f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, data: List[QueryExample],
                overrides: Optional[Sequence[Optional[dict]]] = None
                ) -> List[List[int]]:
        self.model.eval()
        out: List[List[int]] = []
        bs = self.cfg.e2e_batch_size
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            cand, q, mask = collate(batch, self.device)
            logits, tlog = self.model(cand, q, mask)
            for k, b in enumerate(batch):
                lg = logits[k].clone()
                kk = len(b.pool_ids)
                if kk < lg.shape[-1] - 1:               # mask padding columns
                    lg[:, kk:-1] = -1e9
                ov = (overrides[len(out)] if overrides is not None
                      and len(out) < len(overrides) else None) or {}
                out.append(decode(lg, self.cfg.n_max, tlog[k],
                                  self.cfg.e2e_decode, self.cfg.e2e_fire_threshold,
                                  int(ov.get("size", self.cfg.e2e_fixed_size)),
                                  ov.get("profile"), ov.get("types")))
        return out
