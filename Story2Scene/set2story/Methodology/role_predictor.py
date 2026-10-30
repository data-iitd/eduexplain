"""Stage 1: role-sequence prediction (paper §3).

h_q = E(q) (frozen, 1024-d) -> two-layer trunk -> z_q -> autoregressive GRU
decoder over R ∪ {stop}. Training objective (paper Eq., §3):

    L = 1/|D| Σ_q  min_{π ∈ Π_q}  [ - Σ_{t=1}^{|π|+1}  w_{π_t} · log softmax(z_t)_{π_t} ]

with w_r = (1-β)/(1-β^{n_r}) normalised to mean one over roles, and the stop
symbol unweighted (w_stop = 1). Note the SUM over positions (not a
weight-normalised mean, which is what F.cross_entropy(weight=...) computes).
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from Role_Metrics import best_match


class RoleSequencePredictor(nn.Module):
    def __init__(self, roles: Sequence[str], d: int = 1024, hidden: int = 128,
                 role_emb_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.roles = list(roles)
        self.role2idx: Dict[str, int] = {r: i for i, r in enumerate(self.roles)}
        self.STOP = len(self.roles)
        self.vocab_size = len(self.roles) + 1          # roles + stop
        self.BOS = self.vocab_size                     # decoder start token (input only)
        self.trunk = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.role_embed = nn.Embedding(self.vocab_size + 1, role_emb_dim)
        self.gru = nn.GRU(role_emb_dim, hidden, batch_first=True)
        self.out_head = nn.Linear(hidden, self.vocab_size)

    def encode_query(self, h_q: torch.Tensor) -> torch.Tensor:
        return self.trunk(h_q)

    # ------------------------------------------------------------------ training
    def sequence_losses(self, z: torch.Tensor, seqs: List[List[int]],
                        class_weights: torch.Tensor = None) -> torch.Tensor:
        """Per-sequence weighted NLL, summed over positions (incl. trailing stop).
        z: (N, hidden), one row per sequence. Returns (N,)."""
        device = z.device
        N = len(seqs)
        T = max(len(s) for s in seqs) + 1
        inp = torch.full((N, T), self.STOP, dtype=torch.long, device=device)
        tgt = torch.full((N, T), -100, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            full = list(s) + [self.STOP]
            inp[i, 0] = self.BOS
            if s:
                inp[i, 1:len(s) + 1] = torch.tensor(s, device=device)
            tgt[i, :len(full)] = torch.tensor(full, device=device)
        out, _ = self.gru(self.role_embed(inp), z.unsqueeze(0).contiguous())
        logp = F.log_softmax(self.out_head(out), dim=-1)             # (N, T, V)
        mask = tgt != -100
        safe_tgt = tgt.clamp(min=0)
        nll = -logp.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)   # (N, T)
        if class_weights is not None:
            w = class_weights.to(device=device, dtype=nll.dtype)[safe_tgt]
            nll = nll * w
        return (nll * mask).sum(dim=1)

    # ------------------------------------------------------------------ decoding
    @torch.no_grad()
    def decode_greedy(self, z_q: torch.Tensor, max_len: int = 12) -> List[str]:
        h = z_q.reshape(1, 1, -1)
        prev = torch.tensor([[self.BOS]], device=z_q.device)
        seq: List[str] = []
        for _ in range(max_len):
            out, h = self.gru(self.role_embed(prev), h)
            idx = int(self.out_head(out[:, -1]).argmax(-1))
            if idx == self.STOP:
                break
            seq.append(self.roles[idx])
            prev = torch.tensor([[idx]], device=z_q.device)
        return seq

    @torch.no_grad()
    def first_step_confidence(self, z_q: torch.Tensor, seq: List[str]) -> float:
        """Mean per-step max-probability along a decoded sequence (diagnostic)."""
        ids = [self.role2idx[r] for r in seq]
        inp = torch.tensor([[self.BOS] + ids], device=z_q.device)
        out, _ = self.gru(self.role_embed(inp), z_q.reshape(1, 1, -1))
        p = F.softmax(self.out_head(out), -1)[0]
        return float(p.max(-1).values.mean())


def best_match_loss(model: RoleSequencePredictor, h: torch.Tensor,
                    batch_refs: List[List[List[str]]],
                    class_weights: torch.Tensor = None) -> torch.Tensor:
    """Mean over queries of min-over-references sequence loss.
    h: (B, d); batch_refs[b] = reference role sequences of query b."""
    z = model.encode_query(h)
    owners, seqs = [], []
    for b, refs in enumerate(batch_refs):
        for seq in refs:
            owners.append(b)
            seqs.append([model.role2idx[r] for r in seq])
    owner = torch.tensor(owners, device=h.device)
    per_seq = model.sequence_losses(z[owner], seqs, class_weights)
    per_q = torch.full((len(batch_refs),), float("inf"), device=h.device)
    per_q = per_q.scatter_reduce(0, owner, per_seq, reduce="amin")
    return per_q.mean()


def class_balanced_weights(counts: Dict[str, int], roles: Sequence[str], beta: float) -> torch.Tensor:
    """w_r = (1-β)/(1-β^{n_r}) (Cui et al.), mean-normalised over roles; stop = 1.
    β = 0 gives uniform weights, i.e. plain cross-entropy."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0,1), got {beta}")
    w = []
    for r in roles:
        n = counts.get(r, 0)
        w.append(1.0 if (n <= 0 or beta == 0.0) else (1.0 - beta) / (1.0 - beta ** n))
    t = torch.tensor(w, dtype=torch.float32)
    t = t / t.mean().clamp_min(1e-12)
    return torch.cat([t, torch.ones(1)])


def best_match_metrics(pred, refs):
    """(normalised edit distance, RS-F1) against the best-matching reference."""
    m = best_match(pred, refs, multiset=True, norm="max")
    if m is None:
        raise ValueError("best_match_metrics: no reference role sequences")
    return 1.0 - m["rss"], m["rs_f1"]


class MultisetFallback(nn.Module):
    """Role-multiset head (paper §3: used "where clean structure is weak").
    Predicts a count in {0..max_count} per role; order is recovered later by
    the precedence rule (slotfill.order_by_precedence_fallback)."""

    def __init__(self, roles: Sequence[str], d: int = 1024, hidden: int = 128,
                 max_count: int = 3, dropout: float = 0.0):
        super().__init__()
        self.roles = list(roles)
        self.max_count = max_count
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, len(self.roles) * (max_count + 1))

    def forward(self, h_q: torch.Tensor):
        z = self.trunk(h_q)
        return self.head(z).view(*z.shape[:-1], len(self.roles), self.max_count + 1)

    def loss(self, h: torch.Tensor, batch_refs: List[List[List[str]]],
             class_weights: torch.Tensor = None) -> torch.Tensor:
        """Best-match (min over references) class-weighted count CE."""
        logp = F.log_softmax(self.forward(h), -1)                    # (B, R, C)
        losses = []
        w = None if class_weights is None else class_weights[:len(self.roles)].to(h.device)
        for b, refs in enumerate(batch_refs):
            best = None
            for seq in refs:
                counts = torch.tensor([min(seq.count(r), self.max_count) for r in self.roles],
                                      device=h.device)
                nll = -logp[b].gather(-1, counts.unsqueeze(-1)).squeeze(-1)
                l = (nll * w).sum() if w is not None else nll.sum()
                best = l if best is None else torch.minimum(best, l)
            losses.append(best)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict_multiset(self, h_q: torch.Tensor) -> List[str]:
        counts = self.forward(h_q).argmax(-1).tolist()
        out = []
        for role, c in zip(self.roles, counts):
            out.extend([role] * c)
        return out
