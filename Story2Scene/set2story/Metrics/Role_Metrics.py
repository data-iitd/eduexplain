"""
Role-sequence metrics -- ONE implementation shared by
  - the role-predictor ,
  - metrics.py (RS-F1 / RSS in the method comparison),

Both metrics are BEST-MATCH over the reference role sequences: a query may
have several acceptable references and the prediction only has to match ONE
of them. It is never averaged over (or penalised for) the other references.

Definitions (pred = predicted role list, ref = one reference role list):

  RSS(pred, ref)   = 1 - Levenshtein(pred, ref) / max(len(pred), len(ref))
                     bounded in [0, 1]; 1 = identical sequence.
  RS-F1(pred, ref) = order-free F1 between the role MULTISETS
                     (a role used twice must appear twice in the reference
                     to count twice; Eq. 17 "role-multiset F1").

  RSS   = max over refs of RSS(pred, ref)
  RS-F1 = max over refs of RS-F1(pred, ref)         (each metric picks its
                                                      own best reference)

`multiset=False` gives set-F1 (duplicates ignored) and `norm="ref"` divides
the edit distance by len(ref) -- both only for reproducing the older
predictor-test numbers. Do not mix settings between scripts.
"""

from collections import Counter
from typing import Dict, List, Optional, Sequence

ROLES = ["hook", "context", "definition", "mechanism", "comparison", "example",
         "visual", "activity", "check", "misconception", "summary"]

# non-canonical labels that occur in the reference storyboards
ROLE_ALIASES = {
    "misconception_correction": "misconception",
    "concept_definition": "definition",
    "grounded_example": "example",
}


def canonical_role(role: str) -> str:
    r = (role or "").strip().lower()
    return ROLE_ALIASES.get(r, r)


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:

    n, m = len(a), len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m]


def rss_single(
    pred: Sequence[str],
    ref: Sequence[str],
    norm: str = "max",
) -> float:
    """1 - normalised edit distance for ONE reference."""

    if norm == "max":
        denom = max(len(pred), len(ref))

    elif norm == "ref":
        denom = len(ref)

    else:
        raise ValueError("norm must be 'max' or 'ref'")

    if denom == 0:
        return 1.0 if levenshtein(pred, ref) == 0 else 0.0

    return 1.0 - levenshtein(pred, ref) / denom


def f1_single(
    pred: Sequence[str],
    ref: Sequence[str],
    multiset: bool = True,
) -> float:
    """Order-free role F1 for ONE reference."""

    if multiset:

        pc, rc = Counter(pred), Counter(ref)

        overlap = sum((pc & rc).values())

        n_pred, n_ref = sum(pc.values()), sum(rc.values())

    else:

        ps, rs = set(pred), set(ref)

        overlap = len(ps & rs)

        n_pred, n_ref = len(ps), len(rs)

    if n_pred == 0 and n_ref == 0:
        return 1.0

    if n_pred == 0 or n_ref == 0:
        return 0.0

    p = overlap / n_pred

    r = overlap / n_ref

    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def best_match(
    pred: Sequence[str],
    refs: Optional[List[Sequence[str]]],
    multiset: bool = True,
    norm: str = "max",
) -> Optional[Dict]:
    """
    Best-match RSS and RS-F1 against every acceptable reference.

    Returns None if there are no references. Otherwise:
        rss       best RSS over references
        rs_f1     best RS-F1 over references (independent of the RSS pick)
        rss_ref   the reference that gave `rss`
        f1_ref    the reference that gave `rs_f1`
        joint_f1  RS-F1 against `rss_ref` (the older "pick one reference by
                  edit distance, then score F1 on that same one" rule)
    """

    if not refs:
        return None

    best_rss, rss_ref = -1.0, None

    best_f1, f1_ref = -1.0, None

    for ref in refs:

        s = rss_single(pred, ref, norm)

        f = f1_single(pred, ref, multiset)

        if s > best_rss:
            best_rss, rss_ref = s, list(ref)

        if f > best_f1:
            best_f1, f1_ref = f, list(ref)

    return {
        "rss": best_rss,
        "rs_f1": best_f1,
        "rss_ref": rss_ref,
        "f1_ref": f1_ref,
        "joint_f1": f1_single(pred, rss_ref, multiset),
    }

def role_presence_stats(pairs, roles: Sequence[str] = ROLES) -> Dict:
    """Per-role binary presence stats over (predicted roles, reference roles)
    pairs. The reference is the RSS-best one. Returns
    {role: {support, tp, fp, fn, recall, f1}, "macro_f1": mean F1 over roles}."""
    st = {r: {"support": 0, "tp": 0, "fp": 0, "fn": 0} for r in roles}
    for pred, ref in pairs:
        P, R = set(pred), set(ref)
        for r in roles:
            st[r]["support"] += r in R
            if r in P and r in R:
                st[r]["tp"] += 1
            elif r in P:
                st[r]["fp"] += 1
            elif r in R:
                st[r]["fn"] += 1
    for r in roles:
        tp, fp, fn = st[r]["tp"], st[r]["fp"], st[r]["fn"]
        st[r]["recall"] = tp / (tp + fn) if tp + fn else 0.0
        st[r]["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    st["macro_f1"] = sum(st[r]["f1"] for r in roles) / len(roles)
    return st
