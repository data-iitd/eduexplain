"""Central configuration.

Every knob in the method lives here. 
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    resources_path: str = "resources.json"
    queries_path: str = "queries.json"         
    train_queries_path: Optional[str] = None
    candidates_path: Optional[str] = None   
    train_queries_path: Optional[str] = None
    candidates_path: Optional[str] = None
    out_dir: str = "runs/main"
    cache_dir: str = ".cache"

    # -------------------------------------------------------------- encoder
    encoder_name: str = "BAAI/bge-large-en-v1.5"
    encoder_backend: str = "sentence_transformers"
    batch_size: int = 32
    max_seq_len: int = 512
    hash_dim: int = 256  

    # ---------------------------------------------------------- first stage
    K: int = 100
    stratified_first_stage: bool = False
    K_figure: int = 20
    restrict_to_subject: bool = False

    # ---------------------------------------------------------- vocabulary V
    # "structure" = chapter/section/subsection titles (as the paper describes)
    # "gold_concepts" = the corpus `concepts` field (CIRCULAR - eval only, see README)
    vocab_source: str = "gold_concepts"   # "gold_concepts" | "structure"
    vocab_min_df: int = 1              # drop concepts on fewer than this many resources
    vocab_max_df_frac: float = 0.25    # drop hub concepts on >25% of the corpus
    vocab_drop_structural: bool = False  # drop concepts that merely name their own heading
    # A concept counts as structural only if >= this fraction of the resources
    # carrying it have it as their own chapter/section/subsection heading.
    vocab_structural_frac: float = 0.5
    vocab_drop_generic: bool = True      # drop "example", "figure", "activity", ...
    vocab_context_descriptors: bool = True  # enrich descriptor with textbook location

    # ------------------------------------------------------- Con(r) tagging
    tag_threshold: float = 0.45   # used only when tag_zscore=False
    tag_z_threshold: float = 0.0   # absolute floor in sd; 0 disables
    # PRIMARY control on |Con(r)|: keep concepts scoring >= tag_rel x this
    # resource's best match. Tune so mean_Con_size matches the corpus's own
    # concepts-per-resource (reported by `vocab`).
    tag_rel: float = 0.85
    tag_topm: int = 8
    # Source of the covering relation Con(r):
    #   "embed"     -- predicted by embedding similarity against V (default)
    #   "annotated" -- the corpus `concepts` field, which is what the
    #                  benchmark's own Concept Coverage metric uses
    #   "both"      -- annotation, with embedding tags added on top
    #   "learned"   -- predicted by a classifier trained on the annotation,
    #                  out of fold over resources (see res_concept_* below)
    con_source: str = "embed"
    # uqk only: what U_q is scored against -- "union" (all concepts of the gold
    # resources), "core" (Core(q) at tau, what the benchmark metric uses), or
    # "query" (the query-level annotation). Sizes differ by ~7x, so a k tuned
    # on one target does not transfer to another.
    uq_target: str = "union"
    tag_zscore: bool = True

    # ----------------------------------------------------------------- U_q
    # query-anchored baseline: a candidate covers a phrase when its similarity
    # is within anchored_rel of the best candidate for that phrase.
    anchored_rel: float = 0.9
    anchored_cos: float = 0.0
    uq_mode: str = "match"  # "match" | "anchored" | "learned" | "prf"
    prf_docs: int = 5      # top-n first-stage hits used by the prf mode
    prf_alpha: float = 0.5  # weight on the pool centroid vs the query
    gamma: float = 0.80
    # [FIX] relative threshold gamma * max_v a_q(v); absolute cosine cutoffs are brittle
    gamma_relative: bool = True
    kappa: int = 12
    # Absolute cosine floor on a_q(v). When > 0 it REPLACES the relative gamma
    # gate, giving "the top-kappa concepts, provided similarity >= tau".
    uq_abs_tau: float = 0.0
    uq_min: int = 3  # fallback so U_q is never empty;
    w_mode: str = "aq_idf"  # "uniform" | "aq" | "aq_idf"

    # learned concept head
    concept_hidden: int = 256
    concept_lr: float = 1e-3
    concept_epochs: int = 300
    concept_activation: str = "sigmoid"
    # Target for the concept head g_psi: "query" (query-level `concepts`
    # annotation -- a direct gold U_q), "gold_con" (Con(r) of gold resources),
    # or "both". Training target only; never an inference input.
    concept_target: str = "query"
    gamma_learned: float = 0.5
    # How the query-side concept head is fitted:
    #   "auto"           = legacy behaviour: the separate --train-queries file if
    #                      one is given (and --role-train is not cv), else gold CV
    #   "cv"             = grouped k-fold over the EVALUATION queries only, so
    #                      every rho_q comes from a head that never saw that
    #                      query or any paraphrase of it
    #   "silver"         = fit once on --train-queries, predict every eval query
    #   "silver_then_cv" = pre-train once on --train-queries, then warm-start and
    #                      fine-tune per gold fold. The silver and gold sets have
    #                      different distributions, so this and "cv" are the only
    #                      two protocols that are defensible.
    concept_train: str = "auto"
    concept_ft_lr: float = 3e-4      
    concept_ft_epochs: int = 150     
    concept_topk: int = 0

    # ------------------------------------- learned Con(r) (resource side)
    res_concept_hidden: int = 256
    res_concept_lr: float = 1e-3
    res_concept_epochs: int = 60
    res_concept_batch: int = 512
    res_concept_wd: float = 1e-5
    con_folds: int = 5
    con_group: str = "chapter"    
    con_learned_topm: int = 0
    con_learned_tau: float = 0.30
    con_learned_size: str = "topm"   # "topm" | "match"

    # ------------------------------------------------------ role predictor
    n_max: int = 6
    role_hidden: int = 256
    role_lr: float = 1e-3
    role_epochs: int = 300
    role_weight_decay: float = 1e-4
    lam: float = 1.0
    eta: float = 0.5
    role_loss: str = "mean"       
    count_decode: str = "argmax"  
    sizes_composition: str = "role" 
    folds: int = 5
    seed: int = 0
    cv_group: str = "auto"   # "auto"|"bundle_id"|"gold_signature"|"id_prefix"|"none"
    cv_group_prefix_len: int = 4

    # ------------------------------------------- end-to-end variant (E2E)
    # "greedy" = lightweight variant (role head + constrained selection).
    # "e2e"    = learned set predictor with the best-match bipartite loss.
    # "greedy" = lightweight method | "e2e" = learned set predictor
    selector: str = "greedy"
    mmr_lambda: float = 0.5
    e2e_d_model: int = 128
    e2e_heads: int = 4
    e2e_layers: int = 1
    e2e_dropout: float = 0.3
    e2e_lr: float = 3e-4
    e2e_weight_decay: float = 1e-2
    e2e_epochs: int = 120
    e2e_batch_size: int = 16
    e2e_patience: int = 12
    e2e_lam_type: float = 0.5      
    e2e_null_weight: float = 1.0
    e2e_grad_clip: float = 1.0
    e2e_loss_norm: bool = True
    e2e_single_gold: bool = False
    # How the m gold sets are aggregated in the e2e loss:
    #   "best" (Eq. 9) | "single_first" | "single_random" | "average".
    # e2e_single_gold=True is kept as an alias for "single_first".
    e2e_gold_match: str = "best"
    # True: fine-tune the e2e model on the EVALUATION (gold) queries under
    # grouped k-fold CV. Combine with --silver-queries to pre-train on silver first.
    e2e_cv: bool = False
    # External per-query set sizes, {query_id: n}, as written by
    # tools_target_sizes.py. 
    sizes_file: str = ""
    # How Core Concept Coverage is averaged.
    #   "defined"  (default) only queries with >= 2 gold sets AND a non-empty
    #              core. 
    #   "official" every query, scoring 1.0 where the core is empty.
    cc_population: str = "defined"
    # "type": the 3-way type head decides whether a slot fires, the pointer
    # decides where (DETR-style, and the default). "joint": argmax of the merged
    # (K+1)-way softmax, where NULL is one column against K diluted candidate
    # columns and the decode is biased toward the empty set.
    e2e_decode: str = "type"
    e2e_fire_threshold: float = 0.5
    e2e_fixed_size: int = 0
    # Where the end-to-end variant's SET SIZE comes from:
    #   "model"        -- the NULL/fire head decides (default)
    #   "role"         -- the lightweight role head's c_q + f_q
    #   "role_profile" -- the role head's full profile, with each type block
    #                     filled to capacity by pointer confidence
    e2e_size_source: str = "model"
    # Where the lightweight heads (role/concept/intent) are fitted:
    #   "auto" -- the training file when one is given, else k-fold CV (default)
    #   "cv"   -- always grouped k-fold CV over the evaluation queries, even
    #             when --train-queries is supplied for the e2e model
    role_train: str = "auto"
    type_constraint: bool = True
    e2e_dev_frac: float = 0.15
    e2e_early_stop: str = "f1"
    e2e_device: str = "cpu"
    # Optional silver pre-training: Pre-train on it,
    # then fine-tune on the human-validated training queries.
    silver_queries_path: Optional[str] = None
    e2e_silver_epochs: int = 40

    # ------------------------------------------------------------ selection
    greedy: str = "global"
    coverage: str = "graded_max"
    relevance: str = "linear" 
    beta: float = 1.0
    normalize_relevance: bool = True   
    normalize_weights: bool = True     
    stop_on_nonpositive: bool = False  
    backfill: str = "none"             
    strict_inputs: bool = True
    allow_oracles: bool = False

    # ------------------------------------------- official benchmark metrics
    tau: float = 0.8                     # core-concept frequency threshold
    k_values: tuple = (25, 50, 100, 200)  # Recall@K

    # ----------------------------------------------------------------- eval
    profile_mode: str = "predicted"  # "predicted" | "intent" | "oracle" | "fixed"
    fixed_profile: str = "3,1"
    uq_oracle: bool = False          # oracle U_q from gold concepts (diagnostic)
    uq_oracle_mix: float = 0.0
    uq_file: Optional[str] = None
    uq_file_per_text: int = 2  
    exact_gap_check: bool = False
    exact_gap_topn: int = 25
    exact_gap_queries: int = 25

    # -------------------------------------------------------------- helpers
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_args(cls, ns) -> "Config":
        """Build from an argparse namespace, ignoring unset (None) values."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in vars(ns).items() if k in known and v is not None}
        return cls(**kw)

    def parsed_fixed_profile(self):
        c, f = self.fixed_profile.split(",")
        return int(c), int(f)
