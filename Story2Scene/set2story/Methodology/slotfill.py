"""Stages 3-4: grounded slot-filling and time allocation (paper §3)."""
import heapq
from typing import Dict, List, Set

from data import ModalType, Resource, Segment


def compat(r: Resource, role: str, visual_role: str = "visual") -> bool:
    """Role-type compatibility: a `visual` slot must show a figure."""
    if role == visual_role:
        return r.modality == ModalType.IMAGE
    return True


def _marginal_gain(r: Resource, covered: Set[str], Uq_weights: Dict[str, float], beta: float,
                   lam: float, text_already_in_slot: bool, reuse_count: Dict[str, int],
                   reuse_lambda: float):
    """Δ(r|ρ) = Σ_{v ∈ U_q ∩ Con(r) \\ Covered} w_v + β·s_eff(q,r) − λ·1[mod(r)=text ∧ S^text ≠ ∅],
    with s_eff(q,r) = s(q,r) / (1 + λ_reuse · use(r))."""
    new_concepts = (r.concepts & Uq_weights.keys()) - covered
    cov = sum(Uq_weights[v] for v in new_concepts)
    s_eff = r.relevance / (1.0 + reuse_lambda * reuse_count.get(r.rid, 0))
    pen = lam if (r.modality == ModalType.TEXT and text_already_in_slot) else 0.0
    return cov + beta * s_eff - pen, new_concepts


def fill_slots(role_seq: List[str], Rq: List[Resource], Uq_weights: Dict[str, float],
               beta: float = 1.0, b_text: int = 3, lam: float = 0.15,
               visual_role: str = "visual", reuse_lambda: float = 0.25) -> List[Segment]:
    """Greedy partition-matroid fill: per slot, up to b_text texts and one figure.
    The first text of a slot is always taken (S_ℓ ≠ ∅); further texts and the
    figure only on positive marginal gain (a visual slot always takes its figure).
    If no resource is compatible with a role (e.g. `visual` with no figure in
    R_q), the compatibility rule is relaxed for that slot so S_ℓ is never empty."""
    covered: Set[str] = set()
    reuse: Dict[str, int] = {}
    segments: List[Segment] = []

    for role in role_seq:
        cands = [r for r in Rq if compat(r, role, visual_role)] or list(Rq)
        must_take_figure = role == visual_role and any(r.modality == ModalType.IMAGE for r in cands)
        chosen: List[Resource] = []

        texts = [r for r in cands if r.modality == ModalType.TEXT]
        while sum(c.modality == ModalType.TEXT for c in chosen) < b_text:
            pool = [r for r in texts if r not in chosen]
            if not pool:
                break
            has_text = any(c.modality == ModalType.TEXT for c in chosen)
            (gain, new), best = max(
                ((_marginal_gain(r, covered, Uq_weights, beta, lam, has_text, reuse, reuse_lambda), r)
                 for r in pool),
                key=lambda x: x[0][0],
            )
            if has_text and gain <= 0:
                break
            chosen.append(best)
            covered |= new

        figs = [r for r in cands if r.modality == ModalType.IMAGE]
        if figs:
            (gain, new), best = max(
                ((_marginal_gain(r, covered, Uq_weights, beta, lam, False, reuse, reuse_lambda), r)
                 for r in figs),
                key=lambda x: x[0][0],
            )
            if must_take_figure or gain > 0 or not chosen:
                chosen.append(best)
                covered |= new

        for r in chosen:
            reuse[r.rid] = reuse.get(r.rid, 0) + 1
        segments.append(Segment(role=role, resources=chosen))
    return segments


def fill_uncontrolled(role_seq: List[str], Rq: List[Resource],
                      visual_role: str = "visual") -> List[Segment]:
    """Ablation "− compositing control": no per-modality caps, no compositing
    penalty, no marginal-gain rule. Every compatible resource goes into every
    slot (a visual slot still only takes figures — that is typing, not
    compositing), ordered by relevance."""
    segs = []
    for role in role_seq:
        chosen = [r for r in Rq if compat(r, role, visual_role)] or list(Rq)
        segs.append(Segment(role=role, resources=sorted(chosen, key=lambda r: -r.relevance)))
    return segs


def order_by_precedence_fallback(segments: List[Segment], Uq_weights: Dict[str, float],
                                 precedence: Dict[str, Set[str]]) -> List[Segment]:
    """Multiset fallback ordering (paper §3, time allocation): topological sort
    of segments consistent with ≺ over the U_q concepts each introduces, ties
    broken by relevance. Cycles (two segments each holding a prerequisite of
    the other) are broken by relevance."""
    n = len(segments)
    uq = set(Uq_weights)
    seg_c = [set().union(*[r.concepts & uq for r in s.resources]) if s.resources else set()
             for s in segments]
    edges = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and any(precedence.get(u, set()) & seg_c[j] for u in seg_c[i] - seg_c[j]):
                edges[i].append(j)
                indeg[j] += 1

    def key(i):
        return (-max((r.relevance for r in segments[i].resources), default=0.0), i)

    heap = [(key(i), i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order, seen = [], set()
    while len(order) < n:
        if not heap:  # cycle: release the most relevant remaining segment
            rest = [i for i in range(n) if i not in seen]
            i = min(rest, key=key)
            indeg[i] = 0
            heap = [(key(i), i)]
        _, i = heapq.heappop(heap)
        if i in seen:
            continue
        seen.add(i)
        order.append(i)
        for j in edges[i]:
            indeg[j] -= 1
            if indeg[j] == 0 and j not in seen:
                heapq.heappush(heap, (key(j), j))
    return [segments[i] for i in order]


def allocate_time(segments: List[Segment], role_shares: Dict[str, float], T_target: float,
                  tau_min: float = 6.0, delta: float = None) -> List[Segment]:
    """τ_ℓ = T_tgt · τ̄_ρℓ / Σ τ̄, clipped to τ_min and renormalised so that
    Σ τ_ℓ = T_tgt (hence |T(B) − T_tgt| ≤ δ by construction). Roles without a
    prior get the mean prior."""
    if not segments:
        return segments
    n = len(segments)
    if T_target < n * tau_min:
        tau_min = T_target / n
    mean_share = (sum(role_shares.values()) / len(role_shares)) if role_shares else 1.0
    shares = [max(role_shares.get(s.role, mean_share), 1e-9) for s in segments]
    floored: Set[int] = set()
    durations = [0.0] * n
    # iterate: fix segments below the floor, redistribute the rest
    while True:
        free = [i for i in range(n) if i not in floored]
        budget = T_target - tau_min * len(floored)
        tot = sum(shares[i] for i in free)
        for i in free:
            durations[i] = budget * shares[i] / tot
        for i in floored:
            durations[i] = tau_min
        newly = {i for i in free if durations[i] < tau_min}
        if not newly:
            break
        floored |= newly
    for s, d in zip(segments, durations):
        s.duration = float(d)
    if delta is not None:
        assert abs(sum(durations) - T_target) <= delta + 1e-6
    return segments


def role_duration_shares(reference_boards: List[dict]) -> Dict[str, float]:
    """τ̄_ρ: mean duration share of role ρ in the (training-fold) references."""
    acc: Dict[str, List[float]] = {}
    for sb in reference_boards:
        segs = sb["segments"]
        total = sum(float(s["duration_sec"]) for s in segs) or 1.0
        for s in segs:
            acc.setdefault(s["role"], []).append(float(s["duration_sec"]) / total)
    return {r: sum(v) / len(v) for r, v in acc.items()}
