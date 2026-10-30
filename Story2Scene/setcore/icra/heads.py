"""Trained components: the role head g_theta and the optional concept head g_psi.

Both are 2-layer MLPs on the frozen embedding h_q (a few thousand parameters),
implemented in numpy with Adam so the repo has no deep-learning dependency.
To swap in PyTorch, replace RoleHead.fit/predict -- the interfaces are
`fit(H, targets)` and `predict(H) -> (counts, fig_probs)`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------- utils
def _he(shape, rng):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / shape[0])).astype(np.float64)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class _Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=1e-3, wd=0.0,
                 b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.wd = params, lr, wd
        self.b1, self.b2, self.eps, self.t = b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]):
        self.t += 1
        for k, g in grads.items():
            if self.wd and k.startswith("W"):
                g = g + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------- role head
class RoleHead:
    """Eqs. (2)-(5): shared ReLU trunk, softmax count head + sigmoid figure head.

    role_loss="mean"       -> targets are the rounded-mean count and majority
                              figure decision across the m gold sets (Eq. 4).
    role_loss="single_first"  -> train against the first gold profile only.
    role_loss="single_random" -> one gold profile drawn uniformly per epoch.
    role_loss="average"       -> the loss is averaged over all gold profiles,
                                 i.e. a soft target over the admissible counts.
    role_loss="best_match" -> the loss is the minimum over the m gold profiles,
                              which is consistent with the best-match set metric
                              the system is actually scored by. The rounded mean
                              can be a profile that no gold set instantiates.
    """

    def __init__(self, in_dim: int, n_max: int, hidden: int = 256, lr: float = 1e-3,
                 wd: float = 1e-4, epochs: int = 300, lam: float = 1.0,
                 role_loss: str = "mean", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_max, self.epochs, self.lam, self.role_loss = n_max, epochs, lam, role_loss
        self.rng = np.random.default_rng(seed)
        self.P = {
            "W1": _he((in_dim, hidden), rng), "b1": np.zeros(hidden),
            "W2": _he((hidden, hidden), rng), "b2": np.zeros(hidden),
            "Wc": _he((hidden, n_max + 1), rng), "bc": np.zeros(n_max + 1),
            "wf": _he((hidden, 1), rng), "bf": np.zeros(1),
        }
        self.opt = _Adam(self.P, lr=lr, wd=wd)

    def _forward(self, X):
        P = self.P
        a1 = X @ P["W1"] + P["b1"]; z1 = np.maximum(a1, 0)
        a2 = z1 @ P["W2"] + P["b2"]; z2 = np.maximum(a2, 0)
        pc = _softmax(z2 @ P["Wc"] + P["bc"])
        pf = _sigmoid(z2 @ P["wf"] + P["bf"]).ravel()
        return dict(a1=a1, z1=z1, a2=a2, z2=z2, pc=pc, pf=pf)

    def _backward(self, X, c, dpc, dpf):
        """dpc/dpf are dL/dlogits, already averaged over the batch."""
        P, g = self.P, {}
        g["Wc"] = c["z2"].T @ dpc; g["bc"] = dpc.sum(0)
        g["wf"] = c["z2"].T @ dpf[:, None]; g["bf"] = np.array([dpf.sum()])
        dz2 = dpc @ P["Wc"].T + dpf[:, None] * P["wf"].T
        da2 = dz2 * (c["a2"] > 0)
        g["W2"] = c["z1"].T @ da2; g["b2"] = da2.sum(0)
        dz1 = da2 @ P["W2"].T
        da1 = dz1 * (c["a1"] > 0)
        g["W1"] = X.T @ da1; g["b1"] = da1.sum(0)
        return g

    def fit(self, X: np.ndarray, count_targets: List[List[int]],
            fig_targets: List[List[int]], verbose: bool = False):
        """count_targets[i] / fig_targets[i]: the admissible values for query i.

        Under role_loss="mean" these lists hold exactly one value (Eq. 4);
        under "best_match" they hold one entry per gold set.
        """
        X = np.asarray(X, dtype=np.float64)
        N = X.shape[0]
        for ep in range(self.epochs):
            c = self._forward(X)
            pc, pf = c["pc"], c["pf"]

            # --- count head
            mode = self.role_loss
            if mode == "average":
                # Push toward EVERY admissible count at once. The minimiser is
                # the mean of the gold profiles, not any one of them, so this is
                # a materially different objective from best-match -- it is the
                # right comparison for "does set-of-acceptable-sets matter".
                soft = np.zeros_like(pc)
                for i, ts in enumerate(count_targets):
                    for k in ts:
                        soft[i, k] += 1.0 / len(ts)
                dpc = (pc - soft) / N
                tgt = soft.argmax(axis=1)
            else:
                if mode == "single_first":
                    tgt = np.array([ts[0] for ts in count_targets], dtype=int)
                elif mode == "single_random":
                    # Re-drawn every epoch: over training the model sees all the
                    # gold profiles, but on any step it cannot pick the one it
                    # already agrees with, which is what best-match allows.
                    tgt = np.array([ts[self.rng.integers(len(ts))]
                                    for ts in count_targets], dtype=int)
                else:   # "mean" (single value per query) or "best_match"
                    tgt = np.array([max(ts, key=lambda k: pc[i, k])
                                    for i, ts in enumerate(count_targets)],
                                   dtype=int)
                onehot = np.zeros_like(pc); onehot[np.arange(N), tgt] = 1.0
                dpc = (pc - onehot) / N

            # --- figure head
            yf = np.zeros(N); free = np.zeros(N, dtype=bool)
            for i, fs in enumerate(fig_targets):
                if mode == "average":
                    yf[i] = float(np.mean(fs))          # fractional target
                elif mode == "single_first":
                    yf[i] = float(fs[0])
                elif mode == "single_random":
                    yf[i] = float(fs[self.rng.integers(len(fs))])
                else:
                    sset = set(fs)
                    if len(sset) == 2:
                        free[i] = True                  # both admissible -> free
                    else:
                        yf[i] = float(next(iter(sset)))
            dpf = self.lam * (pf - yf) / N
            dpf[free] = 0.0

            self.opt.step(self._backward(X, c, dpc, dpf))
            if verbose and ep % 50 == 0:
                ll = -np.log(pc[np.arange(N), tgt] + 1e-12).mean()
                print(f"  ep {ep:4d} count-CE {ll:.4f}")
        return self

    def predict(self, X: np.ndarray, decode: str = "argmax",
                eta: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self._forward(np.asarray(X, dtype=np.float64))
        pc, pf = c["pc"], c["pf"]
        if decode == "expectation":
            # Cross-entropy is blind to ordinality; decoding by rounded
            # expectation at least respects the ordering of the count classes.
            counts = np.rint(pc @ np.arange(self.n_max + 1)).astype(int)
        else:
            counts = pc.argmax(axis=1)
        return counts, (pf >= eta).astype(int), pf


class MajorityRoleBaseline:
    """Constant predictor. If RoleHead does not beat this, say so in the paper."""

    def fit(self, X, count_targets, fig_targets):
        cs = [ts[0] for ts in count_targets]
        fs = [ts[0] for ts in fig_targets]
        self.c = int(np.bincount(cs).argmax())
        self.f = int(round(float(np.mean(fs))))
        return self

    def predict(self, X, decode="argmax", eta=0.5):
        n = len(X)
        return (np.full(n, self.c), np.full(n, self.f), np.full(n, float(self.f)))


# -------------------------------------------------------------- concept head
class ConceptHead:
    """g_psi: query embedding -> distribution over V, multi-label trained.

    activation="sigmoid" (default) is the correct choice for a multi-label
    objective. activation="softmax" reproduces the write-up literally, but a
    simplex over |V| concepts puts every entry near 1/|V|, so a fixed
    gamma' = 0.5 yields U_q = {} for every query; only use it to demonstrate
    that degeneracy.
    """

    def __init__(self, in_dim: int, n_vocab: int, hidden: int = 256,
                 lr: float = 1e-3, epochs: int = 300, activation: str = "sigmoid",
                 seed: int = 0, batch_size: int = 0, wd: float = 0.0):
        rng = np.random.default_rng(seed)
        self.activation, self.epochs = activation, epochs
        self.lr, self.wd, self.batch_size, self.seed = lr, wd, batch_size, seed
        self.P = {
            "W1": _he((in_dim, hidden), rng), "b1": np.zeros(hidden),
            "W2": _he((hidden, n_vocab), rng), "b2": np.zeros(n_vocab),
        }
        self.opt = _Adam(self.P, lr=lr, wd=wd)
        # One generator for the whole run, NOT re-created per fit() call, so
        # fit(10) six times draws the same minibatch sequence as fit(60). The
        # epoch tuner trains once and checkpoints, and that is only equivalent
        # to training to each epoch budget separately if this persists.
        self._shuffle = np.random.default_rng(seed + 7919)

    # -- warm starting, for the silver -> gold protocol -----------------
    def state(self) -> Dict[str, np.ndarray]:
        """Parameters only. The Adam moments are deliberately NOT carried over:
        fine-tuning on a differently distributed, much smaller set should not
        inherit second-moment estimates from the silver phase."""
        return {k: v.copy() for k, v in self.P.items()}

    def load_state(self, st: Dict[str, np.ndarray], lr: Optional[float] = None,
                   epochs: Optional[int] = None) -> "ConceptHead":
        for k, v in st.items():
            self.P[k] = v.copy()
        self.opt = _Adam(self.P, lr=self.lr if lr is None else lr, wd=self.wd)
        if epochs is not None:
            self.epochs = epochs
        return self

    def _forward(self, X):
        a1 = X @ self.P["W1"] + self.P["b1"]; z1 = np.maximum(a1, 0)
        logits = z1 @ self.P["W2"] + self.P["b2"]
        out = _softmax(logits) if self.activation == "softmax" else _sigmoid(logits)
        return a1, z1, logits, out

    def _step(self, Xb, Yb, pw):
        a1, z1, logits, out = self._forward(Xb)
        n = Xb.shape[0]
        if self.activation == "softmax":
            dlog = (out - Yb / np.maximum(Yb.sum(1, keepdims=True), 1)) / n
        else:
            wmat = np.where(Yb > 0, pw, 1.0)
            dlog = wmat * (out - Yb) / (n * Yb.shape[1])
        g = {"W2": z1.T @ dlog, "b2": dlog.sum(0)}
        da1 = (dlog @ self.P["W2"].T) * (a1 > 0)
        g["W1"] = Xb.T @ da1; g["b1"] = da1.sum(0)
        self.opt.step(g)

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: Optional[int] = None):
        """Full batch by default (batch_size=0), which is what the few-hundred
        query head wants. batch_size > 0 switches to shuffled minibatches, which
        is what makes the resource-side head (tens of thousands of rows)
        tractable in numpy."""
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
        N = X.shape[0]
        pos = Y.sum()
        pw = float((Y.size - pos) / max(pos, 1.0)) if pos > 0 else 1.0
        pw = min(pw, 50.0)  # class-imbalance reweighting; V is large and Y sparse
        n_ep = self.epochs if epochs is None else epochs
        bs = self.batch_size if self.batch_size and self.batch_size < N else 0
        rng = self._shuffle
        for _ in range(n_ep):
            if bs == 0:
                self._step(X, Y, pw)
            else:
                order = rng.permutation(N)
                for s in range(0, N, bs):
                    b = order[s:s + bs]
                    self._step(X[b], Y[b], pw)
        return self

    def predict(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] <= chunk:
            return self._forward(X)[3]
        return np.concatenate([self._forward(X[i:i + chunk])[3]
                               for i in range(0, X.shape[0], chunk)], axis=0)


# ------------------------------------------------------------------- k-folds
def group_kfold_indices(groups: Sequence, k: int, seed: int = 0
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """k-fold that keeps every member of a group in the same fold.

    Needed because surface-form variants of one underlying query share gold
    sets and near-identical embeddings; splitting them across folds leaks.
    Groups are shuffled then greedily packed into the currently smallest fold
    so the folds stay balanced when group sizes are uneven.
    """
    order: Dict = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    keys = list(order)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -len(order[g]))
    folds: List[List[int]] = [[] for _ in range(k)]
    for g in keys:
        folds[int(np.argmin([len(f) for f in folds]))].extend(order[g])
    out = []
    for i in range(k):
        te = np.array(sorted(folds[i]), dtype=int)
        tr = np.array(sorted(j for f in range(k) if f != i for j in folds[f]), dtype=int)
        out.append((tr, te))
    return out


def kfold_indices(n: int, k: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((tr, te))
    return out


class IntentHead:
    """Query -> intent classifier, for the "predicted intent" ablation.

    Intent labels supervise this head but are never an inference input, so the
    row stays inside the leakage boundary. Comparing it against the gold-intent
    row separates how much is lost to intent-prediction error from how much the
    intent -> profile mapping was ever worth.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 256,
                 lr: float = 1e-3, epochs: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.epochs = epochs
        self.P = {"W1": _he((in_dim, hidden), rng), "b1": np.zeros(hidden),
                  "W2": _he((hidden, n_classes), rng), "b2": np.zeros(n_classes)}
        self.opt = _Adam(self.P, lr=lr)

    def _forward(self, X):
        a1 = X @ self.P["W1"] + self.P["b1"]
        z1 = np.maximum(a1, 0)
        return a1, z1, _softmax(z1 @ self.P["W2"] + self.P["b2"])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        N, C = X.shape[0], self.P["W2"].shape[1]
        Y = np.zeros((N, C)); Y[np.arange(N), y] = 1.0
        for _ in range(self.epochs):
            a1, z1, p = self._forward(X)
            dlog = (p - Y) / N
            g = {"W2": z1.T @ dlog, "b2": dlog.sum(0)}
            da1 = (dlog @ self.P["W2"].T) * (a1 > 0)
            g["W1"] = X.T @ da1; g["b1"] = da1.sum(0)
            self.opt.step(g)
        return self

    def predict(self, X):
        return self._forward(np.asarray(X, dtype=np.float64))[2].argmax(axis=1)
