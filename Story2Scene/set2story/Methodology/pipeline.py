"""Algorithm 1: grounded storyboard construction, five stages end to end."""
from typing import Dict, List, Optional

import numpy as np
import torch

from data import Resource, Storyboard
from concept import ConceptIndex, concept_weights
from slotfill import allocate_time, fill_slots, fill_uncontrolled, order_by_precedence_fallback
from realization import NarrationVerifier, realize_segment
from resource_converter import resources_from_gold_set

# Defaults; fold-specific values come from training/ (validation-tuned).
DEFAULT_PARAMS = dict(beta=1.0, b_text=3, lam=0.15, reuse_lambda=0.25,
                      gamma=0.25, kappa=8, tau_min=6.0, max_regen=3)


class GSCContext:
    """Per-process shared state: frozen encoder E_φ, the concept index and a
    cache of resource embeddings E(desc(r))."""

    def __init__(self, encoder, concept_vocab):
        self.encoder = encoder
        self.concepts = ConceptIndex(concept_vocab, encoder)
        self._emb: Dict[str, np.ndarray] = {}

    def query_vec(self, query: str) -> np.ndarray:
        return np.asarray(self.encoder.encode(query), dtype=np.float32)

    def attach_relevance(self, Rq: List[Resource], h_q: np.ndarray) -> List[Resource]:
        """s(q, r) = cos(E(q), E(desc(r))). The benchmark ships no relevance
        scores, so any input `relevance` is overwritten."""
        todo = [r for r in Rq if r.rid not in self._emb]
        if todo:
            E = np.asarray(self.encoder.encode([r.content or r.rid for r in todo]), dtype=np.float32)
            for r, e in zip(todo, E):
                self._emb[r.rid] = e / (np.linalg.norm(e) + 1e-9)
        hq = h_q / (np.linalg.norm(h_q) + 1e-9)
        for r in Rq:
            r.relevance = float(self._emb[r.rid] @ hq)
        return Rq

    def prepare(self, query: str, gold_set: Dict, gamma: float, kappa: int):
        """-> (Rq with relevance, U_q weights, precedence on U_q)."""
        h_q = self.query_vec(query)
        Rq = self.attach_relevance(resources_from_gold_set(gold_set), h_q)
        Uq_w = concept_weights(self.concepts.signal(query, gamma, kappa, h_q=h_q))
        return Rq, Uq_w, self.concepts.precedence(list(Uq_w))


def predict_role_sequence(model, h_q, max_len: int = 12) -> List[str]:
    model.eval()
    with torch.no_grad():
        z = model.encode_query(torch.as_tensor(h_q, dtype=torch.float32))
        return model.decode_greedy(z, max_len=max_len)


def plan_storyboard(role_seq: List[str], Rq: List[Resource], Uq_w: Dict[str, float],
                    precedence: Dict, role_shares: Dict[str, float], T_target: float,
                    params: Dict, order: str = "sequence", fill: str = "controlled"):
    """Stages 3-4 (deterministic): slot-filling, (fallback) ordering, timing."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if fill == "uncontrolled":
        segs = fill_uncontrolled(role_seq, Rq)
    else:
        segs = fill_slots(role_seq, Rq, Uq_w, beta=p["beta"], b_text=int(p["b_text"]),
                          lam=p["lam"], reuse_lambda=p["reuse_lambda"])
    if order == "precedence":       # multiset fallback: order recovered by ≺
        segs = order_by_precedence_fallback(segs, Uq_w, precedence)
    return allocate_time(segs, role_shares, T_target, p["tau_min"])


def build_storyboard(query: str, role_seq: List[str], Rq: List[Resource], Uq_w: Dict[str, float],
                     precedence: Dict, role_shares: Dict[str, float], T_target: float,
                     segment_generator, verifier: NarrationVerifier, params: Optional[Dict] = None,
                     order: str = "sequence", fill: str = "controlled") -> Storyboard:
    """Stages 3-5 given the role sequence (stage 1) and concept signal (stage 2)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    role_seq = role_seq or ["hook", "summary"]      # degenerate decode guard
    segs = plan_storyboard(role_seq, Rq, Uq_w, precedence, role_shares, T_target, p, order, fill)
    prior: List[str] = []
    for seg in segs:
        seg.narration, seg.onscreen, seg.src = realize_segment(
            query, seg.role, seg.resources, verifier, segment_generator,
            max_regen=int(p["max_regen"]), prior_narrations=prior, trace=seg.trace)
        prior.append(seg.narration)
    return Storyboard(query=query, segments=segs)
