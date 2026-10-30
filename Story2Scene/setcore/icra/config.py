"""Central configuration.

Every knob in the method lives here. Anything marked [PAPER] reproduces the
write-up literally; anything marked [FIX] is one of the corrections discussed
in review and is switchable so you can ablate faithful-vs-corrected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    resources_path: str = "resources.json"
    queries_path: str = "queries.json"          # EVALUATION queries (held out)
    # Separate training queries. When set, heads are fit ONCE on this file and
    # the evaluation file is never trained on -- no k-fold needed. This is the
    # clean setup when the gold benchmark queries are your test set.
    train_queries_path: Optional[str] = None
    # Precomputed first-stage pools (the benchmark's candidate files). When set,
    # retrieval is read from this file instead of being recomputed, which is how
    # the BM25 / OpenCLIP / dense+reranker rows of the first-stage table are
    # produced without reimplementing those retrievers.
    candidates_path: Optional[str] = None          # EVALUATION queries (held out)
    # Separate training queries. When set, heads are fit ONCE on this file and
    # the evaluation file is never trained on -- no k-fold needed. This is the
    # clean setup when the gold benchmark queries are your test set.
    train_queries_path: Optional[str] = None
    # Precomputed first-stage pools (the benchmark's candidate files). When set,
    # retrieval is read from this file instead of being recomputed, which is how
    # the BM25 / OpenCLIP / dense+reranker rows of the first-stage table are
    # produced without reimplementing those retrievers.
    candidates_path: Optional[str] = None
    out_dir: str = "runs/main"
    cache_dir: str = ".cache"

    # -------------------------------------------------------------- encoder
    encoder_name: str = "BAAI/bge-m3"
    # "sentence_transformers" = real BGE-M3; "hash" = deterministic offline stub
    encoder_backend: str = "sentence_transformers"
    batch_size: int = 32
    max_seq_len: int = 512
    hash_dim: int = 256  # only used by the hash backend

    # ---------------------------------------------------------- first stage
    K: int = 100
    # [FIX] type-stratified pool so the figure slot is fillable
    stratified_first_stage: bool = False
    K_figure: int = 20
    restrict_to_subject: bool = False  # filter pool by class_no/subject if query has them

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
    # Cutoff in standard deviations, applied when tag_zscore=True. This is the
    # knob that actually controls |Con(r)|; tag_threshold cannot, because after
    # z-scoring every top-ranked concept has sigmoid(z) > 0.9.
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
    # [FIX] column-wise standardisation of the concept-similarity matrix (hubness)
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
    # Motivation: gamma is applied to raw cosine while Con(r) is gated on
    # z-scored similarity, so the two halves of the concept layer use
    # incompatible scales. Measured on EduExplain, the relative gate binds for
    # only ~11% of queries and kappa truncates the other 89% -- U_q is "the 12
    # nearest concepts" regardless of how specific the query is. An absolute
    # floor lets |U_q| shrink for narrow queries instead.
    uq_abs_tau: float = 0.0
    uq_min: int = 3  # fallback so U_q is never empty; set 0 to allow empty
    # aq_idf is the sensible default here: structural concepts have very high
    # document frequency and should not carry the same weight as leaf concepts.
    w_mode: str = "aq_idf"  # "uniform" | "aq" | "aq_idf"

    # learned concept head
    concept_hidden: int = 256
    concept_lr: float = 1e-3
    concept_epochs: int = 300
    # [FIX] sigmoid is the correct activation for a multi-label objective.
    # "softmax" reproduces the paper literally (see README - it degenerates).
    concept_activation: str = "sigmoid"
    # Target for the concept head / oracle U_q:
    #   "query"     = the query-level `concepts` annotation (a gold U_q)
    #   "gold_con"  = Con(r) of the gold resources (fallback when absent)
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
    concept_ft_lr: float = 3e-4      # fine-tuning lr for silver_then_cv
    concept_ft_epochs: int = 150     # fine-tuning epochs for silver_then_cv
    # Top-|U_q| decode for the learned head. gamma_learned is a probability
    # threshold, which is hard to calibrate across folds; with
    # concept_topk > 0 the head's top-k concepts are taken instead, which makes
    # |U_q| directly comparable to the matcher at the same kappa.
    concept_topk: int = 0

    # ------------------------------------- learned Con(r) (resource side)
    # con_source="learned" replaces the embedding tagger with a multi-label
    # classifier resource embedding -> V, supervised by the corpus `concepts`
    # annotation. Predictions are OUT OF FOLD over resources, so no resource's
    # own annotation is ever used to tag it; grouping by chapter keeps
    # near-duplicate resources from the same chapter out of each other's
    # training folds.
    res_concept_hidden: int = 256
    res_concept_lr: float = 1e-3
    res_concept_epochs: int = 60
    res_concept_batch: int = 512
    res_concept_wd: float = 1e-5
    con_folds: int = 5
    con_group: str = "chapter"       # "chapter" | "document" | "none"
    # Con(r) decode: top-m per resource, intersected with a probability floor.
    # 0 falls back to tag_topm so the learned and embedding taggers produce
    # comparably sized Con(r) and the comparison is not a sparsity confound.
    con_learned_topm: int = 0
    con_learned_tau: float = 0.30
    # "match" sizes Con(r) to the mean |annotation| instead of a fixed top-m,
    # by taking each resource's top-ceil(mean) concepts above the floor.
    con_learned_size: str = "topm"   # "topm" | "match"

    # ------------------------------------------------------ role predictor
    n_max: int = 6
    role_hidden: int = 256
    role_lr: float = 1e-3
    role_epochs: int = 300
    role_weight_decay: float = 1e-4
    lam: float = 1.0
    eta: float = 0.5
    role_loss: str = "mean"       # "mean" = Eq.(4)/(5) | [FIX] "best_match"
    count_decode: str = "argmax"  # "argmax" | [FIX] "expectation"
    # How a --sizes TOTAL is turned into a profile for the lightweight
    # selector. "role" keeps the role head's figure decision and spends the
    # rest on explanations, so only |R_q| is pinned and the method is otherwise
    # intact. "free" drops the type constraint for budgeted queries, matching
    # how the LLM baselines consume the same file.
    sizes_composition: str = "role"   # "role" | "free"
    folds: int = 5
    seed: int = 0
    # [FIX] grouped CV so surface-form variants of one item cannot straddle a split
    cv_group: str = "auto"   # "auto"|"bundle_id"|"gold_signature"|"id_prefix"|"none"
    cv_group_prefix_len: int = 4

    # ------------------------------------------- end-to-end variant (E2E)
    # "greedy" = lightweight variant (role head + constrained selection).
    # "e2e"    = learned set predictor with the best-match bipartite loss.
    # "greedy" = lightweight method | "e2e" = learned set predictor
    # Non-learned controls for Table 1: "topn", "modality_capped", "mmr",
    # "mmr_modality".
    selector: str = "greedy"
    mmr_lambda: float = 0.5
    # Sized for a few hundred to a couple of thousand training queries. At
    # d_model=256 / 2 layers (~2.7M params on a 1024-d encoder) this model
    # memorises: measured train F1 0.87 against eval F1 0.34 on 841 training
    # queries. d_model=128 / 1 layer is ~6x smaller. Raise them only if
    # train_f1 and the eval F1 in the `e2e` block are close, which means
    # capacity is not the binding constraint.
    e2e_d_model: int = 128
    e2e_heads: int = 4
    e2e_layers: int = 1
    e2e_dropout: float = 0.3
    e2e_lr: float = 3e-4
    e2e_weight_decay: float = 1e-2
    e2e_epochs: int = 120
    e2e_batch_size: int = 16
    e2e_patience: int = 12
    e2e_lam_type: float = 0.5      # weight on the role-consistency term
    # Relative weight of the NULL term. With e2e_loss_norm=True both the
    # pointing and NULL terms are already means, so the count imbalance DETR's
    # eos_coef=0.1 compensates for is gone and 1.0 is the balanced choice.
    # Lower it to make the model emit larger sets, raise it for smaller ones:
    # measured on synthetic data, 0.1 -> 4.1 resources, 1.0 -> 2.9.
    # If you set e2e_loss_norm=False, use ~0.1 instead.
    e2e_null_weight: float = 1.0
    e2e_grad_clip: float = 1.0
    # Average the pointing term over matched pairs and the NULL term over
    # unmatched slots, rather than summing both over the slot count. Without
    # this the per-gold-set loss grows with gold set size, so the min over gold
    # sets selects the SMALLEST acceptable set (measured: 100% of the time) and
    # the model collapses to ~1 resource per query. False reproduces that.
    e2e_loss_norm: bool = True
    # Table 6 ablation: train against one gold set instead of the min over all
    # of them, isolating what the set-of-acceptable-sets objective contributes.
    e2e_single_gold: bool = False
    # How the m gold sets are aggregated in the e2e loss:
    #   "best" (Eq. 9) | "single_first" | "single_random" | "average".
    # e2e_single_gold=True is kept as an alias for "single_first".
    e2e_gold_match: str = "best"
    # True: fine-tune the e2e model on the EVALUATION (gold) queries under
    # grouped k-fold CV, so every prediction comes from a model that never saw
    # that query. Combine with --silver-queries to pre-train on silver first.
    # The silver and gold sets differ in distribution (silver gold sets are
    # smaller), so training on silver and testing on gold is a transfer setting,
    # not the same experiment.
    e2e_cv: bool = False
    # External per-query set sizes, {query_id: n}, as written by
    # tools_target_sizes.py. When given, EVERY system reads the same numbers:
    # the lightweight selector fills exactly n slots (type constraint dropped,
    # since the file carries a total and not a split) and the e2e decoder
    # returns exactly n. This is the only way to be certain the comparison is
    # size-matched -- icra's own oracle profile rounds the explanation count and
    # the figure flag separately, which is a different function of the gold
    # sets than "round the total".
    sizes_file: str = ""
    # How Core Concept Coverage is averaged.
    #   "defined"  (default) only queries with >= 2 gold sets AND a non-empty
    #              core. With one gold set every concept clears tau trivially,
    #              so the notion is vacuous there.
    #   "official" every query, scoring 1.0 where the core is empty. This is
    #              what the benchmark's own evaluation/evaluator.py does, so use
    #              it when a number must line up with numbers produced by that
    #              script. The two average over different populations (272 vs
    #              321 queries here) and are NOT interchangeable.
    cc_population: str = "defined"
    # "type": the 3-way type head decides whether a slot fires, the pointer
    # decides where (DETR-style, and the default). "joint": argmax of the merged
    # (K+1)-way softmax, where NULL is one column against K diluted candidate
    # columns and the decode is biased toward the empty set.
    e2e_decode: str = "type"
    e2e_fire_threshold: float = 0.5
    # > 0 forces the decode to emit exactly this many resources, overriding the
    # NULL/fire decision. For size-matched comparison against the lightweight
    # variant and the controls; 0 leaves the model free to choose its own size.
    e2e_fixed_size: int = 0
    # Where the end-to-end variant's SET SIZE comes from:
    #   "model"        -- the NULL/fire head decides (default)
    #   "role"         -- the lightweight role head's c_q + f_q
    #   "role_profile" -- the role head's full profile, with each type block
    #                     filled to capacity by pointer confidence
    # The hybrid separates the two things the e2e model has to learn jointly
    # from 992 examples: how many resources, and which ones.
    e2e_size_source: str = "model"
    # Where the lightweight heads (role/concept/intent) are fitted:
    #   "auto" -- the training file when one is given, else k-fold CV (default)
    #   "cv"   -- always grouped k-fold CV over the evaluation queries, even
    #             when --train-queries is supplied for the e2e model
    role_train: str = "auto"
    # False drops the per-type capacity constraint from the selector, keeping
    # only the total size c_q + f_q. The e2e pointer scores HIGHER without the
    # type blocks (0.4063 vs 0.3972), so it is worth asking the same question
    # of the greedy selector.
    type_constraint: bool = True
    # Dev slice carved out of the TRAINING queries for early stopping. Never
    # touches the evaluation file. Set to 0 to disable (expect overfitting).
    e2e_dev_frac: float = 0.15
    # "f1" selects the checkpoint by dev best-match F1; "loss" by dev loss.
    e2e_early_stop: str = "f1"
    e2e_device: str = "cpu"
    # Optional silver pre-training: a queries file in the same schema, produced
    # by the benchmark's proposer WITHOUT human validation. Pre-train on it,
    # then fine-tune on the human-validated training queries.
    silver_queries_path: Optional[str] = None
    e2e_silver_epochs: int = 40

    # ------------------------------------------------------------ selection
    # [FIX] "global" = one argmax over all feasible candidates (matroid greedy, 1/2)
    # "block"  = Algorithm 1 exactly as written (type-by-type, no guarantee)
    greedy: str = "global"
    # [PAPER] "binary" | [FIX] "graded_max" | [FIX] "noisy_or"
    coverage: str = "graded_max"
    relevance: str = "linear"  # "linear" | [FIX] "sqrt" (saturating)
    beta: float = 1.0
    normalize_relevance: bool = True   # min-max s(q,.) into [0,1]; needed for monotonicity
    normalize_weights: bool = True     # sum_v w_v = 1 over U_q; makes beta interpretable
    stop_on_nonpositive: bool = False  # [FIX] allow returning fewer than pi_q[t]
    backfill: str = "none"             # "none" | "explanation" when figure slot unfillable

    # --------------------------------------------------------- input hygiene
    # When True the model may read ONLY: query text, resource_type, subject,
    # chapter_title, summary, description, image_caption, text, and the
    # resource `concepts` field (via the vocabulary + Con(r) tagging).
    # Construction-time query fields -- bundle_id, bundle_type, variant_style,
    # query_type, grade, and query-level `concepts` -- are excluded from every
    # model path. They remain loaded for the `audit` command and for reporting
    # breakdowns, which do not feed the selector.
    strict_inputs: bool = True
    # Opt-in escape hatch for the oracle ablations the write-up promises
    # (profile_mode="intent", uq_oracle). Refused under strict_inputs unless set.
    allow_oracles: bool = False

    # ------------------------------------------- official benchmark metrics
    tau: float = 0.8                     # core-concept frequency threshold
    k_values: tuple = (25, 50, 100, 200)  # Recall@K

    # ----------------------------------------------------------------- eval
    profile_mode: str = "predicted"  # "predicted" | "intent" | "oracle" | "fixed"
    fixed_profile: str = "3,1"
    uq_oracle: bool = False          # oracle U_q from gold concepts (diagnostic)
    # Fraction of U_q replaced by gold concepts, in [0,1]. 0 = the deployable
    # predicted signal, 1 = the oracle. Sweeping it measures how accurate
    # concept prediction must be before the coverage term pays for itself.
    uq_oracle_mix: float = 0.0
    # External U_q source: JSON of {query_id: [...]}. Entries may be vocabulary
    # concept ids, or free text (an LLM's sub-queries or stated information
    # requirements), which is embedded and mapped onto the vocabulary. Lets an
    # expensive concept predictor be plugged in while selection stays cheap.
    uq_file: Optional[str] = None
    uq_file_per_text: int = 2   # vocabulary concepts kept per free-text entry
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
