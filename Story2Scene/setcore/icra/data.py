"""Loading resources / queries, pedagogical-type folding, vocabulary V."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# T = {explanation, figure}. Examples and activities fold into `explanation`
# (Section "Inputs and Assumptions"); their concepts still enter coverage.
EXPLANATION_RAW = {"explanation", "explanatory", "example", "activity", "text", "passage"}
FIGURE_RAW = {"figure", "image", "diagram", "illustration", "photo", "table"}
TYPES = ("explanation", "figure")


def fold_type(raw: Optional[str]) -> str:
    r = (raw or "explanation").strip().lower()
    if r in FIGURE_RAW:
        return "figure"
    if r in EXPLANATION_RAW:
        return "explanation"
    # Unknown types default to explanation rather than being silently dropped.
    return "explanation"


@dataclass
class Resource:
    resource_id: str
    type: str                      # folded, in TYPES
    raw_type: str
    document_id: Optional[str] = None
    class_no: Optional[int] = None
    subject: Optional[str] = None
    chapter_no: Optional[int] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    text: str = ""
    summary: Optional[str] = None
    description: Optional[str] = None
    image_caption: Optional[str] = None
    gold_concepts: List[str] = field(default_factory=list)  # EVAL ONLY
    raw: Dict[str, Any] = field(default_factory=dict)

    def index_text(self) -> str:
        """The stored summary / generated description.

        This is the *only* resource representation the selector may read --
        the same one the first-stage retriever indexes. `gold_concepts` is
        never touched here.
        """
        for cand in (self.summary, self.description, self.image_caption):
            if cand and cand.strip():
                return cand.strip()
        return (self.text or "").strip()[:2000] or self.resource_id


@dataclass
class Query:
    query_id: str
    text: str
    gold_sets: List[List[str]] = field(default_factory=list)  # m acceptable sets
    intent: Optional[str] = None      # = query_type; construction-time, oracle only
    class_no: Optional[int] = None
    subject: Optional[str] = None
    # Construction-time annotations. NONE of these may be an inference input;
    # bundle_type in particular is close to a direct label for pi_q.
    bundle_type: Optional[str] = None
    variant_style: Optional[str] = None
    bundle_id: Optional[str] = None   # identifies the underlying item
    # Query-level concept annotation = a gold U_q. Construction-time, so it may
    # be a concept-head TARGET or an oracle, never an inference input.
    gold_concepts: List[str] = field(default_factory=list)
    # Surface-form grouping key. Paraphrase variants of one underlying item must
    # not straddle a CV split (see assign_groups).
    group_id: Optional[str] = None
    # Resource metadata embedded in the gold sets, kept for cross-validation
    # against resources.json only.
    gold_meta: List[List[Dict[str, Any]]] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    rationales: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


DOCID_RE = re.compile(r"CLASS[_-]?(\d+)[_-]([A-Z]+)[_-]?CH[_-]?(\d+)", re.I)
_SUBJ = {"BIO": "Biology", "SCI": "Science", "PHY": "Physics", "CHEM": "Chemistry",
         "MATH": "Mathematics", "CH": "Chemistry"}


def parse_document_id(doc_id: Optional[str]):
    """CLASS_11_BIO_CH_03 -> (11, 'Biology', 3). Returns (None, None, None) on miss."""
    if not doc_id:
        return None, None, None
    m = DOCID_RE.search(str(doc_id))
    if not m:
        return None, None, None
    return int(m.group(1)), _SUBJ.get(m.group(2).upper(), m.group(2).title()), int(m.group(3))


def _first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_resources(path: str) -> List[Resource]:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # tolerate {"resources": [...]}
        raw = _first(raw, "resources", "data", default=[])
    out = []
    for d in raw:
        rt = _first(d, "resource_type", "type", default="explanation")
        out.append(
            Resource(
                resource_id=str(_first(d, "resource_id", "id")),
                type=fold_type(rt),
                raw_type=str(rt),
                document_id=_first(d, "document_id", "doc_id"),
                class_no=_first(d, "class_no", "grade"),
                subject=_first(d, "subject"),
                chapter_no=_first(d, "chapter_no"),
                chapter_title=_first(d, "chapter_title"),
                section=_first(d, "section"),
                subsection=_first(d, "subsection"),
                text=_first(d, "text", default="") or "",
                summary=_first(d, "summary"),
                description=_first(d, "description"),
                image_caption=_first(d, "image_caption", "caption"),
                gold_concepts=list(_first(d, "concepts", default=[]) or []),
                raw=d,
            )
        )
    ids = [r.resource_id for r in out]
    assert len(ids) == len(set(ids)), "duplicate resource_id in resources.json"
    return out


def _extract_gold_set(gs: Any) -> Tuple[List[str], List[Dict[str, Any]], str, str]:
    """Accept either a flat id list or the {resources_used, answer, rationale} form."""
    if isinstance(gs, dict):
        ru = _first(gs, "resources_used", "resources", "items", default=[]) or []
        if ru and isinstance(ru[0], dict):
            ids = [str(_first(r, "resource_id", "id")) for r in ru]
            meta = list(ru)
        else:
            ids, meta = [str(x) for x in ru], []
        return ids, meta, str(gs.get("answer") or ""), str(gs.get("rationale") or "")
    if isinstance(gs, (list, tuple)):
        if gs and isinstance(gs[0], dict):
            return ([str(_first(r, "resource_id", "id")) for r in gs], list(gs), "", "")
        return ([str(x) for x in gs], [], "", "")
    return [str(gs)], [], "", ""


def load_queries(path: str) -> List[Query]:
    """Tolerant loader for the EduExplain query schema:

        {"query_id": "Q020058",
         "query": "...",
         "gold_sets": [{"resources_used": [{"resource_id": "FIG_000230", ...}, ...],
                        "answer": "...", "rationale": "..."}],
         "query_type": "visual", "bundle_type": "triple",
         "variant_style": "learning_context", "grade": 11}

    Flat `gold_sets: [["TXT_1","FIG_2"]]` and `gold_set: [...]` are also accepted.
    class_no/subject are taken from `grade`/`subject` when present, otherwise
    inferred from the gold resources' document_id.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = _first(raw, "queries", "data", default=[])
    out = []
    for d in raw:
        gs_raw = _first(d, "gold_sets", "acceptable_sets", "gold")
        if gs_raw is None:
            single = _first(d, "gold_set", "relevant_resources", default=[])
            gs_raw = [single] if single else []
        elif isinstance(gs_raw, list) and gs_raw and isinstance(gs_raw[0], str):
            gs_raw = [gs_raw]  # a single flat list given under a plural key

        ids, meta, answers, rats = [], [], [], []
        for gs in gs_raw:
            i, m, a, r = _extract_gold_set(gs)
            ids.append(i); meta.append(m); answers.append(a); rats.append(r)

        cls = _first(d, "class_no", "grade")
        subj = _first(d, "subject")
        if cls is None or subj is None:
            for mset in meta:
                for r in mset:
                    c, s, _ = parse_document_id(_first(r, "document_id"))
                    cls = cls if cls is not None else c
                    subj = subj or s
                if cls is not None and subj:
                    break

        out.append(
            Query(
                query_id=str(_first(d, "query_id", "id")),
                text=str(_first(d, "query_text", "query", "text", default="")),
                gold_sets=ids, gold_meta=meta, answers=answers, rationales=rats,
                intent=_first(d, "query_type", "intent", "intent_category"),
                bundle_type=_first(d, "bundle_type"),
                variant_style=_first(d, "variant_style"),
                bundle_id=_first(d, "bundle_id"),
                gold_concepts=[slugify(c) for c in
                               (_first(d, "concepts", default=[]) or [])],
                class_no=cls, subject=subj, raw=d,
            )
        )
    return out


def assign_groups(queries: Sequence[Query], mode: str = "auto",
                  prefix_len: int = 4) -> None:
    """Set `group_id` so paraphrase variants stay on one side of a CV split.

    `variant_style` says each underlying item appears in several surface forms.
    Those variants share gold sets and near-identical embeddings, so a plain
    k-fold over queries puts near-duplicates in both train and test and inflates
    the role predictor's apparent accuracy. Grouped CV is what the write-up
    should describe.

      auto           : `bundle_id` when present (the corpus's own item key, and
                       the cleanest option since it touches no gold data),
                       else gold_signature.
      bundle_id      : force the corpus item key.
      gold_signature : group by the exact multiset of gold resource ids. Works,
                       but builds the split from gold data.
      id_prefix      : group by the first `prefix_len` chars of query_id, for
                       corpora where the id encodes item + variant.
      none           : plain k-fold over queries (leaky if variants exist).
    """
    if mode == "auto":
        mode = "bundle_id" if any(q.bundle_id for q in queries) else "gold_signature"
    for q in queries:
        if mode == "none":
            q.group_id = q.query_id
        elif mode == "id_prefix":
            q.group_id = q.query_id[:prefix_len]
        elif mode == "bundle_id" and q.bundle_id:
            q.group_id = str(q.bundle_id)
        else:
            sig = tuple(sorted(tuple(sorted(s)) for s in q.gold_sets))
            # sha1, NOT hash(): Python salts string hashing per process, so
            # hash() gives a different group_id every run. The ids are then
            # sorted lexicographically when the e2e dev split is drawn, which
            # made two identical commands train on different dev sets. Group
            # MEMBERSHIP was always correct; only the order was unstable.
            q.group_id = hashlib.sha1(
                repr(sig).encode("utf-8")).hexdigest()[:16]


def validate(resources: Sequence[Resource], queries: Sequence[Query]) -> Dict[str, Any]:
    """Sanity report; run this before anything else.

    Also cross-checks the resource metadata embedded in gold sets against
    resources.json. Disagreement there is a corpus bug that would silently
    corrupt the role targets (which are read off gold-set types).
    """
    by_id = {r.resource_id: r for r in resources}
    n_types: Dict[str, int] = {}
    for r in resources:
        n_types[r.type] = n_types.get(r.type, 0) + 1

    dangling, empty_gold, sizes, m_sizes = 0, 0, [], []
    type_mismatch, concept_mismatch, checked = 0, 0, 0
    n_exp, n_fig, has_fig = [], [], []
    by_intent: Dict[str, int] = {}
    by_bundle: Dict[str, Dict[str, Any]] = {}
    by_variant: Dict[str, int] = {}
    cross_chapter = 0

    for q in queries:
        if not q.gold_sets:
            empty_gold += 1
        m_sizes.append(len(q.gold_sets))
        by_intent[str(q.intent)] = by_intent.get(str(q.intent), 0) + 1
        by_variant[str(q.variant_style)] = by_variant.get(str(q.variant_style), 0) + 1

        for si, s in enumerate(q.gold_sets):
            sizes.append(len(s))
            dangling += sum(1 for x in s if x not in by_id)
            ex = sum(1 for x in s if x in by_id and by_id[x].type == "explanation")
            fg = sum(1 for x in s if x in by_id and by_id[x].type == "figure")
            n_exp.append(ex); n_fig.append(fg); has_fig.append(int(fg > 0))
            docs = {by_id[x].document_id for x in s if x in by_id}
            if len(docs) > 1:
                cross_chapter += 1
            if q.bundle_type:
                b = by_bundle.setdefault(str(q.bundle_type),
                                         {"n": 0, "sizes": [], "with_figure": 0})
                b["n"] += 1; b["sizes"].append(len(s)); b["with_figure"] += int(fg > 0)

            for meta in (q.gold_meta[si] if si < len(q.gold_meta) else []):
                rid = str(meta.get("resource_id"))
                if rid not in by_id:
                    continue
                checked += 1
                if meta.get("resource_type") and \
                        fold_type(meta["resource_type"]) != by_id[rid].type:
                    type_mismatch += 1
                if meta.get("concepts") is not None and \
                        set(meta["concepts"]) != set(by_id[rid].gold_concepts):
                    concept_mismatch += 1

    for b in by_bundle.values():
        b["mean_set_size"] = float(sum(b["sizes"]) / max(1, len(b["sizes"])))
        b["frac_with_figure"] = b["with_figure"] / max(1, b["n"])
        b.pop("sizes")

    ng = len({q.group_id for q in queries if q.group_id is not None})
    return {
        "n_resources": len(resources),
        "resources_by_folded_type": n_types,
        "resources_with_gold_concepts": sum(1 for r in resources if r.gold_concepts),
        "n_queries": len(queries),
        "queries_without_gold_sets": empty_gold,
        "mean_m_gold_sets": float(np_mean(m_sizes)),
        "frac_queries_with_m_eq_1": float(np_mean([1.0 if m == 1 else 0.0 for m in m_sizes])),
        "mean_gold_set_size": float(np_mean(sizes)),
        "mean_explanations_per_gold_set": float(np_mean(n_exp)),
        "mean_figures_per_gold_set": float(np_mean(n_fig)),
        "frac_gold_sets_with_figure": float(np_mean(has_fig)),
        "frac_gold_sets_cross_chapter": float(cross_chapter / max(1, len(sizes))),
        "dangling_gold_resource_ids": dangling,
        "embedded_metadata_checked": checked,
        "embedded_type_mismatches": type_mismatch,
        "embedded_concept_mismatches": concept_mismatch,
        "queries_by_intent": by_intent,
        "queries_by_variant_style": by_variant,
        "bundle_type_vs_actual": by_bundle,
        "n_cv_groups": ng,
        "mean_queries_per_group": float(len(queries) / ng) if ng else 0.0,
    }


def np_mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------- vocab
@dataclass
class Vocabulary:
    ids: List[str]
    names: List[str]
    descriptors: List[str]
    df: List[int] = field(default_factory=list)          # resources carrying it
    is_structural: List[bool] = field(default_factory=list)
    structural_frac: List[float] = field(default_factory=list)
    dropped: Dict[str, List[str]] = field(default_factory=dict)  # reason -> concepts

    def __len__(self):
        return len(self.ids)


def _norm_topic(s: str) -> str:
    s = re.sub(r"^\s*[\d.]+\s*", "", str(s)).strip()
    return re.sub(r"\s+", " ", s)


def slugify(s: Optional[str]) -> str:
    s = _norm_topic(str(s or "")).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# Concepts that describe the *resource*, not its subject matter. `example` and
# `activity` are pedagogical types you fold into `explanation`, so leaving them
# in the concept space lets coverage proxy for type.
GENERIC_CONCEPTS = {
    "example", "examples", "activity", "activities", "figure", "figures",
    "diagram", "diagrams", "illustration", "table", "tables", "exercise",
    "exercises", "question", "questions", "answer", "answers", "summary",
    "introduction", "overview", "note", "notes", "text", "image",
}


def build_vocabulary(resources: Sequence[Resource], source: str = "structure",
                     min_df: int = 1, max_df_frac: float = 1.0,
                     drop_structural: bool = False, drop_generic: bool = True,
                     context_descriptors: bool = True,
                     structural_frac: float = 0.5) -> Vocabulary:
    """Build the controlled concept vocabulary V.

    `gold_concepts` (recommended for EduExplain): V is the union of the corpus
    `concepts` field. Note what that field actually contains -- for a typical
    record it is the slugified chapter title, section, and subsection, plus a
    few genuine content terms, plus noise. So:

      * Structural concepts (slug equals a chapter/section/subsection title)
        are carried by every resource under that heading. Their document
        frequency is huge, so they contribute coverage mass without
        discriminating between candidate sets. They are detected and flagged;
        `max_df_frac` or `drop_structural` removes them.
      * Generic concepts naming the resource type are dropped by default.
      * Because V now comes from the same annotation source as the evaluation
        metric, this must be disclosed. Con(r) is still re-derived from text by
        embedding similarity, so no per-resource label leaks -- but the
        vocabulary's membership does.

    `structure`: V from chapter/section/subsection topics only.
    """
    n_res = max(1, len(resources))
    struct_slugs = set()
    for r in resources:
        for k in (r.chapter_title, r.section, r.subsection):
            if k:
                struct_slugs.add(slugify(k))

    if source == "gold_concepts":
        df: Dict[str, int] = {}
        ctx: Dict[str, Dict[str, int]] = {}
        # A concept is "structural" only if it names the heading of the
        # resources that actually carry it. Matching against the GLOBAL set of
        # headings is wrong at corpus scale: in a large corpus some section is
        # called "Water", which would wrongly brand the content concept
        # `water` structural wherever it appears.
        own_hits: Dict[str, int] = {}
        for r in resources:
            where = " > ".join([x for x in (r.chapter_title, r.section, r.subsection) if x])
            meta = f"Class {r.class_no} {r.subject}".strip() if r.class_no else (r.subject or "")
            here = {slugify(x) for x in (r.chapter_title, r.section, r.subsection) if x}
            for c in r.gold_concepts:
                key = slugify(c)
                if not key:
                    continue
                df[key] = df.get(key, 0) + 1
                if key in here:
                    own_hits[key] = own_hits.get(key, 0) + 1
                if where:
                    ctx.setdefault(key, {})
                    ctx[key][f"{meta}. {where}"] = ctx[key].get(f"{meta}. {where}", 0) + 1

        def is_struct(key: str) -> bool:
            """True when most occurrences of the concept sit under its own heading."""
            return own_hits.get(key, 0) / max(1, df.get(key, 1)) >= structural_frac

        dropped: Dict[str, List[str]] = {"generic": [], "low_df": [],
                                         "high_df": [], "structural": []}
        keep = []
        for key, d in df.items():
            if drop_generic and key in GENERIC_CONCEPTS:
                dropped["generic"].append(key); continue
            if d < min_df:
                dropped["low_df"].append(key); continue
            if d / n_res > max_df_frac:
                dropped["high_df"].append(key); continue
            if drop_structural and is_struct(key):
                dropped["structural"].append(key); continue
            keep.append(key)

        ids = sorted(keep)
        names = [k.replace("_", " ") for k in ids]
        descs = []
        for k, nm in zip(ids, names):
            if context_descriptors and ctx.get(k):
                # Most frequent textbook location, to give the embedding context.
                where = max(ctx[k].items(), key=lambda x: x[1])[0]
                descs.append(f"{nm}. Topic in {where}.")
            else:
                descs.append(nm)
        return Vocabulary(ids=ids, names=names, descriptors=descs,
                          df=[df[k] for k in ids],
                          is_structural=[is_struct(k) for k in ids],
                          structural_frac=[round(own_hits.get(k, 0) / max(1, df[k]), 3)
                                           for k in ids],
                          dropped={k: sorted(v) for k, v in dropped.items() if v})

    # ---------------------------------------------------------- structure
    seen: Dict[str, str] = {}
    for r in resources:
        subj = r.subject or ""
        cls = f"Class {r.class_no} " if r.class_no is not None else ""
        chap = _norm_topic(r.chapter_title) if r.chapter_title else ""
        for level, val in (("chapter", r.chapter_title),
                           ("section", r.section),
                           ("subsection", r.subsection)):
            if not val:
                continue
            name = _norm_topic(val)
            if not name:
                continue
            key = f"{level}::{chap}::{name}".lower()
            if key in seen:
                continue
            if level == "chapter":
                desc = f"{cls}{subj}. Chapter topic: {name}."
            else:
                desc = f"{cls}{subj}, chapter '{chap}'. {level.capitalize()} topic: {name}."
            seen[key] = desc
    ids = sorted(seen)
    names = [i.split("::")[-1] for i in ids]
    return Vocabulary(ids=ids, names=names, descriptors=[seen[i] for i in ids],
                      df=[0] * len(ids), is_structural=[True] * len(ids))


def load_candidates(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load precomputed first-stage pools, e.g. the benchmark's own candidate
    files for BM25 / dense / OpenCLIP / dense+reranker.

    Using these instead of recomputing retrieval is what lets the selector be
    compared across first stages without reimplementing each retriever, and it
    guarantees the pools are byte-identical to the ones the benchmark reports
    Recall@K on.

    Tolerated shapes:
        {"Q1": [{"resource_id": "...", "score": 1.2}, ...], ...}
        {"Q1": ["TXT_1", "TXT_2", ...], ...}
        [{"query_id": "Q1", "candidates": [...]}, ...]

    "results", "pool", "resources" and "retrieved_resources" are accepted in
    place of "candidates", so a competitor's own output file can be passed
    straight in: its `retrieved_resources` is the pool that system actually
    consumed, and reusing it guarantees the comparison differs by selection
    alone rather than by first stage.

    Order is taken as the ranking. A missing score falls back to descending
    rank, so an unscored list still yields a usable s(q,r).
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = {str(_first(d, "query_id", "id")):
               _first(d, "candidates", "results", "pool", "resources",
                      "retrieved_resources", default=[])
               for d in raw}
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, lst in raw.items():
        if isinstance(lst, dict):
            lst = _first(lst, "candidates", "results", "pool",
                         "retrieved_resources", default=[]) or []
        pairs = []
        for rank, c in enumerate(lst):
            if isinstance(c, dict):
                rid = str(_first(c, "resource_id", "id", "doc_id"))
                sc = _first(c, "score", "relevance", "rerank_score")
            else:
                rid, sc = str(c), None
            pairs.append((rid, float(sc) if sc is not None else float(-rank)))
        out[str(qid)] = pairs
    return out
