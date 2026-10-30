#!/usr/bin/env python3
"""Stage 3 of SetCoRe-X: one Llama-3.1-70B-Instruct call selects from D_q.

Implements the LLM-selection protocol of the method spec. The prompt carries
the query, the PREDICTED PROFILE, and each candidate's ID, type and textual
representation. The model selects exactly n_q distinct IDs, weighing joint
sufficiency against redundancy; the type composition is GUIDANCE, not a
mandatory output quota. It returns only a JSON list of IDs.

Protocol, in order:

  1. One call. Parse a JSON array of IDs out of the response.
  2. Validate: every ID present in D_q, no duplicates, size exactly n_q
     (or 1..--flex-max under --size-mode flexible).
  3. On any failure, ONE format-repair call that states what was wrong and asks
     for the bare JSON array.
  4. On persistent failure, fall back to the highest-ranked n_q members of
     D_q. Because scx_reduce.py emits D_q in retrieval order, that is its
     first n_q entries.

Repairs and fallbacks are counted AND their token cost is reported separately,
so the headline cost of the method includes the retries rather than hiding
them.

Figures carry the benchmark's existing descriptions through index_text()
(summary -> description -> image_caption -> text), so every selector sees the
same information about a figure.

Note on presentation order: candidates appear in the order D_q arrives, which
scx_reduce.py guarantees is original retrieval order for BOTH reducers. A
difference between the reducers therefore cannot be an ordering artefact.
--shuffle re-orders within the prompt to measure position sensitivity
directly.

Usage
-----
  python scx_llm_select.py \\
      --shortlists ~/scx/shortlists/setcore_m2.json \\
      --resources $R --queries $Q \\
      --llm-model ~/models/Llama-3.1-70B-Instruct/ \\
      --size-mode exact \\
      --out ~/scx/preds/scx_setcore_m2_exact.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from scx_common import (expand_tildes, index_text, load_queries, load_records,
                        load_resource_map, query_map, write_json)

PROMPT = """You are selecting educational resources that TOGETHER form a complete explanation of a learner's query.

Learner query: {question}
{guidance}
Candidate resources:

{context}

Task:
1. Consider what a complete explanation of the query must cover for a learner.
2. Choose {size_text} from the candidates above, so that together they cover what is needed with as little redundancy as possible. Prefer resources that complement each other over resources that repeat the same point.

Return ONLY a JSON array of the selected resource IDs, exactly as written above, and nothing else.

Example of the required output format:
{example}
"""

# Table 3's prompt ladder. Every row shares one guided pool, so the only thing
# that varies between them is this block -- which is the point of the table:
# it separates what the PROMPT is told from how the POOL was built.
SOFT = ("\nPredicted composition for this query: {profile}. Treat this as "
        "guidance about what a good answer looks like, not as a quota you "
        "must fill.\n")
HARD = ("\nRequired composition for this query: {profile}. This is a QUOTA: "
        "your selection must contain exactly that many of each kind. The TYPE "
        "line of each candidate tells you which kind it is.\n")
CONCEPTS = ("\nThe answer should between them cover these concepts: "
            "{concepts}.\n")


def guidance_block(mode: str, c: int, f: int, uq: Sequence[str],
                   hard_ok: bool) -> str:
    """The one block that distinguishes the Table 3 prompt rows."""
    if mode == "size":
        return "\n"
    tmpl = HARD if (mode == "hard" and hard_ok) else SOFT
    out = tmpl.format(profile=profile_text(c, f))
    if mode == "soft_concepts" and uq:
        out += CONCEPTS.format(
            concepts=", ".join(str(u).replace("_", " ") for u in uq))
    return out

REPAIR = """Your previous answer could not be used: {problem}

The valid resource IDs are exactly these:
{ids}

Return ONLY a JSON array of {size_text_short}, chosen from that list, and nothing else. No explanation, no other text.

Example of the required output format:
{example}
"""


def profile_text(c: int, f: int) -> str:
    bits = []
    if c:
        bits.append(f"{c} explanatory resource" + ("s" if c != 1 else ""))
    if f:
        bits.append(f"{f} figure" + ("s" if f != 1 else ""))
    return " and ".join(bits) if bits else "1 resource"


def size_text(mode: str, n: int, flex_max: int) -> str:
    if mode == "flexible":
        return (f"between 1 and {flex_max} resources (you decide how many)")
    return f"EXACTLY {n} resource" + ("s" if n != 1 else "")


def build_context(ids: Sequence[str], rmap: Dict[str, Dict],
                  types: Optional[Sequence[str]] = None) -> str:
    blocks = []
    for k, rid in enumerate(ids):
        r = rmap.get(rid, {"resource_id": rid})
        t = (types[k] if types and k < len(types) else None) \
            or r.get("resource_type") or r.get("type") or "resource"
        blocks.append(f"ID: {rid}\nTYPE: {t}\nTEXT: {index_text(r)}")
    return "\n\n".join(blocks)


def example_json(ids: Sequence[str], n: int) -> str:
    sample = list(ids[:max(1, min(n, len(ids)))]) or ["RESOURCE_ID"]
    return json.dumps(sample)


# ------------------------------------------------------------------- parsing

def parse_ids(response: str, valid: Sequence[str]) -> Tuple[List[str], str]:
    """Pull a JSON array of IDs out of a response. Returns (ids, how).

    Three attempts, in order of how much they trust the model:
      json      a parseable JSON array somewhere in the text
      quoted    quoted strings that happen to be valid IDs
      bare      unquoted occurrences of valid IDs, in order of appearance

    The last two are salvage, counted separately so the paper can report how
    often the stated format was actually produced.
    """
    text = (response or "").strip()
    vset = set(valid)

    # Strip a markdown fence if the model wrapped the array in one.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fence.group(1).strip() if fence else text

    for cand in ([body] if body.startswith("[") else []) + \
            re.findall(r"\[[^\[\]]*\]", body):
        try:
            got = json.loads(cand)
        except Exception:
            continue
        if isinstance(got, list):
            ids = [str(x).strip() for x in got if isinstance(x, (str, int))]
            if ids:
                return ids, "json"

    quoted = [s for s in re.findall(r'["\']([^"\']+)["\']', body)
              if s.strip() in vset]
    if quoted:
        return [s.strip() for s in quoted], "quoted"

    # [FIX] This used to use body.find(rid), which matches a SUBSTRING: with
    # ids R1 and R100 in the candidate list, a response naming only R100 was
    # parsed as [R1, R100] -- injecting a resource the model never named. Match
    # whole tokens instead, and scan longest-id-first so a longer id is
    # consumed before any of its prefixes can match.
    found: List[Tuple[int, str]] = []
    masked = body
    for rid in sorted(vset, key=len, reverse=True):
        for mm in re.finditer(r"(?<![0-9A-Za-z_])" + re.escape(rid)
                              + r"(?![0-9A-Za-z_])", masked):
            found.append((mm.start(), rid))
            masked = masked[:mm.start()] + "\0" * len(rid) \
                + masked[mm.end():]
            break
    if found:
        return [rid for _, rid in sorted(found)], "bare"
    return [], "none"


def validate(ids: Sequence[str], valid: Sequence[str], mode: str, n: int,
             flex_max: int,
             quota: Optional[Tuple[int, int, Dict[str, str]]] = None
             ) -> Tuple[List[str], Optional[str]]:
    """Returns (cleaned ids, problem) -- problem is None when acceptable."""
    vset = set(valid)
    seen, clean, dupes, unknown = set(), [], 0, []
    for x in ids:
        x = str(x).strip()
        if x not in vset:
            unknown.append(x)
            continue
        if x in seen:
            dupes += 1
            continue
        seen.add(x)
        clean.append(x)

    if unknown:
        return clean, (f"it contained {len(unknown)} ID(s) that are not in the "
                       f"candidate list, such as {unknown[0]!r}")
    if dupes:
        return clean, f"it repeated {dupes} ID(s); the IDs must be distinct"
    if not clean:
        return clean, "no valid resource IDs could be read from it"
    if mode == "flexible":
        if not (1 <= len(clean) <= flex_max):
            return clean, (f"it selected {len(clean)} resources; between 1 and "
                           f"{flex_max} are allowed")
    else:
        if len(clean) != n:
            return clean, (f"it selected {len(clean)} resources, but exactly "
                           f"{n} are required")
    if quota is not None:
        # Only under --guidance hard, and only where the pool could satisfy
        # it. The prompt states the quota as a requirement, so an answer that
        # breaks it is as invalid as a wrong size -- and goes to the same
        # repair call rather than being silently accepted.
        want_c, want_f, tmap = quota
        got_f = sum(1 for x in clean if tmap.get(x) == "figure")
        got_c = len(clean) - got_f
        if (got_c, got_f) != (want_c, want_f):
            return clean, (f"it selected {got_c} explanatory resource(s) and "
                           f"{got_f} figure(s), but exactly {want_c} and "
                           f"{want_f} are required")
    return clean, None


# ----------------------------------------------------------------------- LLM

def load_llama(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading Llama from: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        local_files_only=True)
    model.eval()
    return tok, model


def generate(tok, model, prompt: str, max_new_tokens: int):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    formatted = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt")
    n_in = int(inputs["input_ids"].shape[1])
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.eos_token_id)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True), n_in, int(gen.shape[0])


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlists", required=True)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--llm-model", "--llm_model", dest="llm_model",
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-mode", "--size_mode", dest="size_mode",
                    default="exact", choices=["exact", "flexible"],
                    help="exact: n_q as predicted. flexible: the ablation in "
                         "which the model picks 1..--flex-max itself.")
    ap.add_argument("--guidance", default="soft",
                    choices=["size", "soft", "soft_concepts", "hard"],
                    help="Table 3's prompt ladder, all on the SAME pool: "
                         "'size' states only how many to choose; 'soft' adds "
                         "the predicted composition as guidance (the "
                         "default, and what every other table uses); "
                         "'soft_concepts' also lists the inferred concepts "
                         "U_q from the shortlist file; 'hard' states the "
                         "composition as a quota AND rejects answers that "
                         "break it, falling back to soft wherever the pool "
                         "cannot satisfy it.")
    ap.add_argument("--flex-max", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--repair-max-new-tokens", type=int, default=128)
    ap.add_argument("--strict-json", action="store_true",
                    help="treat a response that is not a JSON array as "
                         "invalid, so it triggers the repair call. Off by "
                         "default because the spec's named checks are IDs, "
                         "uniqueness and size -- not the format -- but the "
                         "salvage rate is reported either way.")
    ap.add_argument("--no-repair", action="store_true",
                    help="skip the format-repair call; failures go straight to "
                         "the ranked fallback")
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the candidate order inside the prompt, to "
                         "measure position sensitivity")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--query-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and report length statistics "
                         "WITHOUT loading the model")
    ap.add_argument("--print-first", type=int, default=0)
    ns = ap.parse_args()
    expand_tildes(ns)

    rows = load_records(ns.shortlists)
    qmap = query_map(load_queries(ns.queries))
    rmap = load_resource_map(ns.resources)

    if ns.query_id:
        rows = [r for r in rows if str(r.get("query_id")) == ns.query_id]
    if ns.limit:
        rows = rows[:ns.limit]
    if not rows:
        raise SystemExit("no shortlists to process")

    print(f"shortlists : {ns.shortlists}  ({len(rows)} queries)")
    print(f"size mode  : {ns.size_mode}"
          + (f" (1..{ns.flex_max})" if ns.size_mode == "flexible" else ""))
    print(f"order      : {'shuffled' if ns.shuffle else 'as given (retrieval)'}")
    print(f"repair     : {'disabled' if ns.no_repair else 'one call on failure'}")

    import random
    rng = random.Random(ns.shuffle_seed)

    built = []
    n_unmatched = n_empty_sl = n_hard_infeasible = n_no_uq = 0
    for r in rows:
        qid = str(r.get("query_id"))
        q = qmap.get(qid)
        ids = [x for x in (r.get("shortlist") or []) if x]
        if q is None:
            n_unmatched += 1
            continue
        if not ids:
            n_empty_sl += 1
            continue
        types = r.get("types") or None
        if ns.shuffle:
            pairs = list(zip(ids, types or [None] * len(ids)))
            rng.shuffle(pairs)
            ids = [a for a, _ in pairs]
            types = [b for _, b in pairs]
        prof = r.get("profile") or {}
        c = int(prof.get("explanation", 0))
        f = int(prof.get("figure", 0))
        n = int(r.get("target_n") or (c + f) or 1)
        n = min(n, len(ids))

        # A hard quota is only imposed when the pool can actually satisfy it.
        # Demanding 1 figure from a pool with none would make every answer
        # invalid and the row would measure pool composition, not the prompt.
        # The spec's rule: require a feasible subset, else fall back to soft
        # guidance and record it.
        tlist = list(types or [])
        have_f = sum(1 for t in tlist if t == "figure")
        have_c = len(tlist) - have_f
        hard_ok = bool(tlist) and have_c >= c and have_f >= f
        if ns.guidance == "hard" and not hard_ok:
            n_hard_infeasible += 1

        prompt = PROMPT.format(
            question=str(q.get("query") or q.get("text") or ""),
            guidance=guidance_block(ns.guidance, c, f,
                                    r.get("U_q") or [], hard_ok),
            context=build_context(ids, rmap, types),
            size_text=size_text(ns.size_mode, n, ns.flex_max),
            example=example_json(ids, n))
        if ns.guidance == "soft_concepts" and not (r.get("U_q") or []):
            n_no_uq += 1
        built.append({"query_id": qid, "ids": ids, "n": n,
                      "profile": {"explanation": c, "figure": f},
                      "types": tlist, "hard_ok": hard_ok,
                      "U_q": list(r.get("U_q") or []),
                      "prompt": prompt})

    if n_unmatched or n_empty_sl:
        print(f"\n[!] {n_unmatched} shortlist rows matched no query and "
              f"{n_empty_sl} had an empty shortlist;\n    they are DROPPED, "
              f"so this run covers {len(built)} of {len(rows)} rows. A table "
              f"row averaged over\n    a smaller query set is not comparable "
              f"with one averaged over all of them.")

    print(f"\nguidance  : {ns.guidance}")
    if ns.guidance == "hard":
        print(f"  hard quota imposed on {len(built) - n_hard_infeasible}/"
              f"{len(built)} queries")
        if n_hard_infeasible:
            print(f"  [!] {n_hard_infeasible} pools cannot satisfy the "
                  f"predicted composition, so those\n      queries got SOFT "
                  f"guidance instead. The row is a mixture, and the fraction "
                  f"above\n      is what it is a mixture of.")
    if ns.guidance == "soft_concepts":
        print(f"  concepts listed for {len(built) - n_no_uq}/{len(built)} "
              f"queries")
        if n_no_uq:
            print(f"  [!] {n_no_uq} shortlist rows carry no U_q, so their "
                  f"prompt is identical to --guidance\n      soft. Only "
                  f"scx_reduce.py --reducer setcore records U_q; a top-n pool "
                  f"never will.")
            if n_no_uq == len(built):
                print(f"  [!!] NO row has U_q, so this run is "
                      f"indistinguishable from --guidance soft.\n       "
                      f"Reporting it as a separate ablation row would be "
                      f"reporting the same number twice.")

    chars = [len(b["prompt"]) for b in built]
    print(f"\nprompt length: mean {st.mean(chars):.0f} chars  "
          f"median {st.median(chars):.0f}  max {max(chars)}  "
          f"(~{st.mean(chars)/4:.0f} tokens mean, rough 4 chars/token)")
    for b in built[:ns.print_first]:
        print(f"\n{'=' * 74}\n{b['query_id']}  (choose {b['n']} of "
              f"{len(b['ids'])})\n{'=' * 74}\n{b['prompt']}")
    if ns.dry_run:
        print("\n--dry-run: no model loaded, nothing written.")
        return

    tok, model = load_llama(ns.llm_model)

    results, stats = [], Counter()
    tin, tout, secs = [], [], []
    rtin, rtout, rsecs = [], [], []

    for i, b in enumerate(built, start=1):
        print(f"[{i}/{len(built)}] {b['query_id']}  "
              f"choose {b['n']} of {len(b['ids'])}")
        t0 = time.perf_counter()
        resp, ni, no = generate(tok, model, b["prompt"], ns.max_new_tokens)
        dt = time.perf_counter() - t0
        tin.append(ni); tout.append(no); secs.append(dt)

        raw, how = parse_ids(resp, b["ids"])
        stats[f"parse_{how}"] += 1
        quota = ((b["profile"]["explanation"], b["profile"]["figure"],
                  dict(zip(b["ids"], b["types"])))
                 if (ns.guidance == "hard" and b["hard_ok"] and b["types"])
                 else None)
        sel, problem = validate(raw, b["ids"], ns.size_mode, b["n"],
                                ns.flex_max, quota)
        if problem is None and ns.strict_json and how != "json":
            problem = ("it was not a JSON array of IDs; the IDs had to be "
                       "recovered from prose")

        repaired = fell_back = False
        resp2 = None
        rep_cost: Optional[Dict[str, float]] = None
        if problem and not ns.no_repair:
            stats["repair_attempted"] += 1
            rp = REPAIR.format(
                problem=problem,
                ids="\n".join(b["ids"]),
                size_text_short=size_text(ns.size_mode, b["n"], ns.flex_max)
                    .replace("EXACTLY ", "exactly "),
                example=example_json(b["ids"], b["n"]))
            t1 = time.perf_counter()
            resp2, ni2, no2 = generate(tok, model, rp,
                                       ns.repair_max_new_tokens)
            dt2 = time.perf_counter() - t1
            rtin.append(ni2); rtout.append(no2); rsecs.append(dt2)
            rep_cost = {"tokens_in": ni2, "tokens_out": no2,
                        "seconds": round(dt2, 3)}
            raw2, how2 = parse_ids(resp2, b["ids"])
            stats[f"repair_parse_{how2}"] += 1
            sel2, problem2 = validate(raw2, b["ids"], ns.size_mode, b["n"],
                                      ns.flex_max, quota)
            if problem2 is None:
                sel, problem, repaired = sel2, None, True
                stats["repair_succeeded"] += 1
            else:
                stats["repair_failed"] += 1

        if problem is not None:
            # Persistent failure: the highest-ranked n_q members of D_q.
            # scx_reduce emits D_q in retrieval order, so that is the prefix.
            sel = list(b["ids"][:b["n"]])
            fell_back = True
            stats["fallback"] += 1

        results.append({
            "query_id": b["query_id"],
            "predicted_resource_ids": sel,
            "n_predicted": len(sel),
            "target_n": b["n"],
            "n_shortlist": len(b["ids"]),
            "profile": b["profile"],
            "parse_path": how,
            "repaired": repaired,
            "fell_back": fell_back,
            "problem": problem,
            "llm_response": resp,
            "repair_response": resp2,
            "tokens_in": ni, "tokens_out": no, "seconds": round(dt, 3),
            "repair_cost": rep_cost,
        })
        print(f"    -> {sel}"
              + ("   [repaired]" if repaired else "")
              + ("   [FALLBACK]" if fell_back else ""))
        if i % 10 == 0:
            with open(os.path.expanduser(ns.out), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)

    write_json(results, ns.out, "predictions")

    n = len(results)
    sizes = [r["n_predicted"] for r in results]
    print(f"\n{'=' * 74}")
    print(f"queries              : {n}")
    print(f"mean |R_q|           : {st.mean(sizes):.4f}   "
          f"target mean {st.mean(r['target_n'] for r in results):.4f}")
    print(f"\nFIRST CALL   tokens in mean {st.mean(tin):.0f} (max {max(tin)}), "
          f"out mean {st.mean(tout):.0f}, {st.mean(secs):.2f} s/query")
    if rtin:
        print(f"REPAIR CALLS {len(rtin)} of {n} ({len(rtin)/n:.1%})   "
              f"tokens in mean {st.mean(rtin):.0f}, out mean "
              f"{st.mean(rtout):.0f}, {st.mean(rsecs):.2f} s each")
        print(f"             added cost: {sum(rtin)+sum(rtout)} tokens "
              f"({(sum(rtin)+sum(rtout))/(sum(tin)+sum(tout)):.1%} on top of "
              f"the first calls), {sum(rsecs)/60:.1f} min")
    else:
        print("REPAIR CALLS none")
    print(f"\nTOTAL        {sum(tin)+sum(rtin)} tokens in, "
          f"{sum(tout)+sum(rtout)} out, "
          f"{(sum(secs)+sum(rsecs))/60:.1f} min")

    print(f"\nparse path of the FIRST call:")
    for k in ("json", "quoted", "bare", "none"):
        v = stats[f"parse_{k}"]
        if v:
            print(f"  {k:<7s} {v:>5d} ({v/n:.1%})")
    if stats["parse_quoted"] or stats["parse_bare"]:
        print("  'quoted' and 'bare' are salvage: the model did not emit the "
              "JSON array it was\n  asked for. Report these -- they are a "
              "format-compliance number, not a selection one.")
    print(f"\nrepairs: attempted {stats['repair_attempted']}, succeeded "
          f"{stats['repair_succeeded']}, failed {stats['repair_failed']}")
    print(f"fallbacks to the top-n_q of D_q: {stats['fallback']} "
          f"({stats['fallback']/n:.1%})")
    if stats["fallback"]:
        print("  [!] Those queries are TOP-N over the reduced pool, not LLM "
              "selections. Quote the\n      rate beside the headline number: "
              "it bounds how much of the result is the LLM.")
    exact = sum(1 for r in results if r["n_predicted"] == r["target_n"])
    print(f"returned exactly n_q: {exact}/{n} ({exact/n:.1%})")


if __name__ == "__main__":
    main()
