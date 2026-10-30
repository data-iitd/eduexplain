from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

# A vocab entry is (name, description) or (name, description, curriculum_position).
# Use inputs/concept_vocab_curriculum.json (tools/build_curriculum_vocab.py):
# the raw concept_vocab.json is alphabetical and carries no curriculum order.


class ConceptIndex:
    """Caches E(v) for the whole vocabulary, so U_q costs one query encode."""

    def __init__(self, vocab: Sequence[Sequence], encoder):
        self.names = [v[0] for v in vocab]
        self.position = {
            v[0]: (float(v[2]) if len(v) > 2 else float(i)) for i, v in enumerate(vocab)
        }
        descs = [f"{v[0]}: {v[1]}" for v in vocab]
        self.H = np.asarray(encoder.encode(descs), dtype=np.float32)
        self.H /= np.linalg.norm(self.H, axis=1, keepdims=True) + 1e-9
        self.encoder = encoder

    def signal(self, query: str, gamma: float, kappa: int, h_q=None) -> List[Tuple[str, float]]:
        """U_q = {v : cos(E(q), E(v)) >= gamma}, top-kappa (paper §3, concept signal)."""
        if h_q is None:
            h_q = self.encoder.encode(query)
        h_q = np.asarray(h_q, dtype=np.float32)
        scores = self.H @ (h_q / (np.linalg.norm(h_q) + 1e-9))
        order = np.argsort(-scores)
        Uq = [(self.names[i], float(scores[i])) for i in order if scores[i] >= gamma]
        return Uq[:kappa]

    def precedence(self, Uq_names: List[str]) -> Dict[str, Set[str]]:
        return precedence_from_curriculum(Uq_names, self.position)


def concept_signal(query: str, vocab, encoder, gamma: float = 0.25, kappa: int = 8):
    """Uncached convenience wrapper around ConceptIndex.signal."""
    return ConceptIndex(vocab, encoder).signal(query, gamma, kappa)


def concept_weights(Uq: List[Tuple[str, float]]) -> Dict[str, float]:
    """w_v = cos(E(q), E(v)) (floored to stay positive)."""
    return {v: max(score, 1e-3) for v, score in Uq}


def precedence_from_curriculum(Uq_names: List[str], position: Dict[str, float]) -> Dict[str, Set[str]]:
    """u < v (u prerequisite of v) iff u appears strictly earlier in the
    curriculum. Returned as {u: {v, ...}}. Concepts with no curriculum position,
    or tied positions, produce no pair."""
    prec: Dict[str, Set[str]] = {v: set() for v in Uq_names}
    placed = [v for v in Uq_names if v in position]
    for u in placed:
        for v in placed:
            if position[u] < position[v]:
                prec[u].add(v)
    return prec
