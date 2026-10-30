"""Competing methods (paper §4.2). All take (q, R_q) and share the frozen encoder,
the generator (gemma-3-12b-it) and the verifier (nli-deberta-v3-large).

  FreeForm          one unconstrained LLM call emits the whole storyboard; no
                    verification, no budget enforcement (measured, not corrected).
  FixedContract     fixed role progression (after CourseBlueprint). The LLM picks
                    the resources for each slot and writes the segment, under the
                    same verify/regenerate loop, on-screen rule and timing as GSC.
  RetrieveOrder     non-generative structure: greedy coverage selection, order by
                    prerequisite then relevance, uniform timing, role from type.
                    Realised with the shared generator + verifier.
  MostFreqTemplate  GSC with the predicted sequence replaced by the most frequent
                    reference sequence of the training fold.
"""
import json
import re
from typing import Callable, Dict, List, Optional, Set

from data import ModalType, Resource, Segment, Storyboard
from pipeline import build_storyboard
from realization import (NarrationVerifier, _sentence_split, realize_segment, route_onscreen,
                         verify_fidelity)
from slotfill import allocate_time, order_by_precedence_fallback

FIXED_CONTRACT = ["hook", "definition", "mechanism", "summary"]
TYPE_TO_ROLE = {"figure": "visual", "example": "example", "activity": "activity",
                "explanation": "definition", "passage": "definition", "definition": "definition",
                "exercise": "activity"}
VALID_ROLES = ("hook, context, definition, mechanism, comparison, example, visual, "
               "activity, check, misconception, summary")


def _resource_block(resources: List[Resource]) -> str:
    return "\n".join(f"[{'FIGURE' if r.modality == ModalType.IMAGE else 'TEXT'}|id={r.rid}] {r.content}"
                     for r in resources)


def _strip_fences(raw: str) -> str:
    return re.sub(r"^```(json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()


# ======================================================================= FreeForm
FREEFORM_PROMPT = (
    "You are producing a complete {min_min}-{max_min} minute educational video storyboard "
    "for the grade level given below, using ONLY the resources provided.\n"
    "Choose the composition, order, resource selection, and timing yourself -- there is no "
    "fixed structure to follow.\n\n"
    "Emit ONLY a JSON array, each element one segment:\n"
    '[{{"role": "...", "resource_ids": ["..."], "duration_sec": 0, '
    '"narration": "...", "onscreen": "..."}}, ...]\n\n'
    "Valid role values: " + VALID_ROLES + ".\n"
    "The sum of duration_sec across all segments should target a {target}-second explanation.\n"
    "No preamble, no markdown code fences -- the JSON array only.\n\n"
    'QUERY: "{query}"\nGRADE: "{grade}"\nRESOURCES:\n---\n{resources}\n---\n\nJSON:'
)


def parse_freeform(raw: str) -> List[dict]:
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
    except json.JSONDecodeError:
        pass
    start = cleaned.find("[")
    if start == -1:
        return []
    depth = 0
    for i in range(start, len(cleaned)):
        depth += {"[": 1, "]": -1}.get(cleaned[i], 0)
        if depth == 0:
            try:
                return [o for o in json.loads(cleaned[start:i + 1]) if isinstance(o, dict)]
            except json.JSONDecodeError:
                break
    out = []  # truncated: salvage complete objects
    for o in re.findall(r"\{[^{}]*\}", cleaned[start:], re.DOTALL):
        try:
            out.append(json.loads(o))
        except json.JSONDecodeError:
            pass
    return out


def run_freeform(query: str, Rq: List[Resource], grade, llm: Callable, T_target: float) -> Storyboard:
    rid2r = {r.rid: r for r in Rq}
    raw = llm(FREEFORM_PROMPT.format(min_min=2, max_min=5, target=int(T_target), query=query,
                                     grade=grade, resources=_resource_block(Rq)), 1600)
    segs = []
    for it in parse_freeform(raw):
        rids = it.get("resource_ids") or []
        rids = rids if isinstance(rids, list) else [rids]
        try:
            dur = float(it.get("duration_sec", 0) or 0)
        except (TypeError, ValueError):
            dur = 0.0
        # ids outside R_q are dropped from S_ℓ, so GF records the grounding failure
        res = [rid2r[str(x)] for x in rids if str(x) in rid2r]
        segs.append(Segment(role=str(it.get("role", "")).strip().lower() or "context", resources=res,
                            duration=max(dur, 0.0), narration=str(it.get("narration", "")).strip(),
                            onscreen=str(it.get("onscreen", "")).strip(), src="freeform",
                            trace={"raw_rids": [str(x) for x in rids]}))
    if not segs:
        segs = [Segment(role="hook", resources=[], duration=0.0, narration="", onscreen="",
                        src="freeform", trace={"parse_failed": True, "raw": raw[:2000]})]
    return Storyboard(query=query, segments=segs)


# ================================================================== FixedContract
FIXEDCONTRACT_PROMPT = (
    "You are filling ONE slot of a fixed-structure educational video storyboard.\n"
    "The slot's ROLE is given below. From the RESOURCES provided, choose the 1-3 resources "
    "(at most 1 figure) that best fill this role, then write the segment's narration and "
    "on-screen directive using ONLY those chosen resources.\n\n"
    'Emit ONLY a JSON object: {{"resource_ids": ["..."], "narration": "...", "onscreen": "..."}}\n'
    "No preamble, no markdown code fences.\n\n"
    'QUERY: "{query}"\nROLE: "{role}"\nRESOURCES:\n---\n{resources}\n---\n{feedback}\nJSON:'
)


def parse_fixedcontract(raw: str):
    cleaned = _strip_fences(raw)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    try:
        obj = json.loads(m.group(0) if m else cleaned)
        rids = obj.get("resource_ids") or []
        return ([str(x) for x in (rids if isinstance(rids, list) else [rids])],
                str(obj.get("narration", "")).strip(), str(obj.get("onscreen", "")).strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    rm = re.search(r'"resource_ids"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    rids = re.findall(r'"([^"]+)"', rm.group(1)) if rm else []
    nm = re.search(r'"narration"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    om = re.search(r'"onscreen"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned, re.DOTALL)
    return rids, (nm.group(1) if nm else ""), (om.group(1) if om else "")


def run_fixedcontract(query: str, Rq: List[Resource], verifier: NarrationVerifier, llm: Callable,
                      T_target: float, role_shares: Dict[str, float], tau_min: float = 6.0,
                      roles: Optional[List[str]] = None, max_regen: int = 3) -> Storyboard:
    rid2r = {r.rid: r for r in Rq}
    block = _resource_block(Rq)
    segs = []
    for role in roles or FIXED_CONTRACT:
        trace = {"attempts": []}
        feedback, chosen, narration, onscreen = "", [], "", ""
        for attempt in range(max_regen):
            rids, narration, onscreen = parse_fixedcontract(llm(FIXEDCONTRACT_PROMPT.format(
                query=query, role=role, resources=block, feedback=feedback), 400))
            chosen = [rid2r[x] for x in dict.fromkeys(rids) if x in rid2r]
            figs = [r for r in chosen if r.modality == ModalType.IMAGE]
            chosen = [r for r in chosen if r.modality == ModalType.TEXT] + figs[:1]
            ok, bad = verifier.verify_narration(narration, chosen) if chosen else (False, ["<no valid resource selected>"])
            trace["attempts"].append({"narration": narration, "onscreen": onscreen, "ok": ok,
                                      "unsupported": bad, "resource_ids": [r.rid for r in chosen]})
            if ok:
                trace["outcome"] = "first" if attempt == 0 else "regen"
                break
            feedback = ("\nYour previous attempt " + ("selected no valid resource id. "
                        if not chosen else f"had unsupported claims: {bad[:6]}. ")
                        + "Rewrite: select resources from the list and only state facts they contain.\n")
        else:
            if not chosen:        # never selected a valid id: ground in the most relevant resource
                chosen = [max(Rq, key=lambda r: r.relevance)]
                trace["forced_resource"] = True
            verdicts = verifier.claim_verdicts(narration, chosen)
            kept = [c for c, v in verdicts if v != "unsupported"]
            if kept and any(v == "supported" for _, v in verdicts):
                narration, trace["outcome"] = " ".join(kept), "salvage"
            else:
                narration = next(iter(_sentence_split(chosen[0].content)), narration)
                onscreen, trace["outcome"] = f"On-screen text: {narration[:160]}", "extractive"
        on, src = route_onscreen(narration, onscreen, chosen)
        trace["onscreen_raw"], trace["src_raw"] = on, src
        trace["fid_pass"] = verify_fidelity(narration, on, chosen, verifier)
        if not trace["fid_pass"]:
            on, src = f"On-screen text: {narration[:160]}", "text_render"
        segs.append(Segment(role=role, resources=chosen, narration=narration, onscreen=on,
                            src=src, trace=trace))
    allocate_time(segs, role_shares, T_target, tau_min)
    return Storyboard(query=query, segments=segs)


# ================================================================== RetrieveOrder
def greedy_cover_select(Rq: List[Resource], Uq: Dict[str, float], beta: float) -> List[Resource]:
    """Greedy coverage (+ β·relevance) over texts while the gain is positive
    (the first pick is always taken), then at most one figure."""
    covered: Set[str] = set()
    chosen: List[Resource] = []

    def gain(r):
        new = (r.concepts & Uq.keys()) - covered
        return sum(Uq[v] for v in new) + beta * r.relevance, new

    texts = [r for r in Rq if r.modality == ModalType.TEXT]
    while True:
        pool = [r for r in texts if r not in chosen]
        if not pool:
            break
        (g, new), r = max(((gain(r), r) for r in pool), key=lambda x: x[0][0])
        if chosen and g <= 0:
            break
        chosen.append(r)
        covered |= new
    figs = [r for r in Rq if r.modality == ModalType.IMAGE]
    if figs:
        (g, new), r = max(((gain(r), r) for r in figs), key=lambda x: x[0][0])
        if g > 0 or not chosen:
            chosen.append(r)
    return chosen


def plan_retrieveorder(Rq, Uq, precedence, T_target, beta=1.0, tau_min=6.0) -> List[Segment]:
    segs = [Segment(role="", resources=[r]) for r in greedy_cover_select(Rq, Uq, beta)]
    segs = order_by_precedence_fallback(segs, Uq, precedence or {})
    for s in segs:
        s.role = TYPE_TO_ROLE.get(s.resources[0].rtype, "definition")
    return allocate_time(segs, {}, T_target, tau_min)         # uniform timing


def run_retrieveorder(query, Rq, Uq, precedence, verifier, segment_generator, T_target,
                      beta=1.0, tau_min=6.0, max_regen=3) -> Storyboard:
    segs = plan_retrieveorder(Rq, Uq, precedence, T_target, beta, tau_min)
    prior: List[str] = []
    for s in segs:
        s.narration, s.onscreen, s.src = realize_segment(
            query, s.role, s.resources, verifier, segment_generator, max_regen, prior, s.trace)
        prior.append(s.narration)
    return Storyboard(query=query, segments=segs)


# =============================================================== MostFreqTemplate
def run_mostfreqtemplate(query, template, Rq, Uq, precedence, role_shares, T_target,
                         segment_generator, verifier, params) -> Storyboard:
    return build_storyboard(query, list(template), Rq, Uq, precedence, role_shares, T_target,
                            segment_generator, verifier, params)
