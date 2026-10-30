"""
E1 component-swap metrics for the demo paper's Table (tab:swap), over the 35 premade runs (7 configurations x 5 queries):

  GS-F1     Stage 1 set vs the query's acceptable gold sets: max over gold sets of 2|R∩G| / (|R|+|G|)
  Entailed  share of check-worthy narration sentences that set2story's NLI verifier accepts (entailment >= 0.5 against
            premises from the segment's own resources; set2story realization.NarrationVerifier, the paper's NG metric)
  Over      seconds of final video beyond the storyboard budget T, max(0, duration - T) (ffprobe)
  also      figures: share of storyboard figure resources that the final scene loads (ImageMobject on the exact file)

Means over the five queries. CPU only (NLI cross-encoder).

    CUDA_VISIBLE_DEVICES= /mnt/storage/aryan/set2story/venv/bin/python -u e1_metrics.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VX = HERE.parent.parent
S2S = Path("/mnt/storage/aryan/set2story")
for d in ("Methodology", "Metrics", "Baseline"):
    sys.path.insert(0, str(S2S / d))
sys.path.insert(0, "/mnt/storage/aryan/VideoModule")

from backend import BatchedNLI  # noqa: E402  (set2story Methodology/backend.py)
from realization import NarrationVerifier  # noqa: E402
from resource_converter import resources_from_gold_set  # noqa: E402

QIDS = ["Q007098", "Q006487", "Q015643", "Q006124", "Q015908"]
CONFIGS = [("A0/B1", "gold", "s2s"), ("A1", "lw", "s2s"), ("A2", "e2e", "s2s"), ("A3", "llm", "s2s"),
           ("A4", "optiset", "s2s"), ("B0", "gold", "ref"), ("B2", "gold", "ff")]
GOLD = {q["query_id"]: [[r["resource_id"] for r in g["resources_used"]] for g in q["gold_sets"]]
        for q in json.load(open("/mnt/storage/aryan/setCore/gold_sets_final_validated_new_merged.json"))}
BENCH = json.load(open(HERE / "benchmark_items.json"))


def gs_f1(pred, golds):
    p = set(pred)
    return max((2 * len(p & set(g)) / (len(p) + len(g)) for g in golds), default=0.0)


def duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def main():
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-large/snapshots").glob("*"))
    verifier = NarrationVerifier(BatchedNLI(str(snaps[-1]), device="cpu", max_batch=64), 0.5, 0.2)
    rows, per = [], {}
    for cid, s1, s2 in CONFIGS:
        acc = {"gsf1": [], "ent": [], "over": [], "dur": [], "T": [], "fig": []}
        for q in QIDS:
            d = VX / "runs" / f"premade-{s1}__{s2}__{q}"
            r1 = json.load(open(d / "stage1.json"))
            sel = r1["selection"]["selected"]
            acc["gsf1"].append(gs_f1(sel, GOLD[q]))
            res = {r.rid: r for r in resources_from_gold_set(r1["gold_set"]["gold_sets"][0])}
            sb = json.load(open(d / "storyboard.json"))["storyboard"]
            sup = tot = 0
            for seg in sb["segments"]:
                rs = [res[i] for i in seg.get("resource_ids") or [] if i in res] or list(res.values())
                for _, v in verifier.claim_verdicts(seg.get("narration", ""), rs):
                    if v != "skip":
                        tot += 1
                        sup += v == "supported"
            acc["ent"].append(sup / tot if tot else 0.0)
            r3 = json.load(open(d / "stage3.json"))
            dur, T = duration(r3["video"]), BENCH[q]["T"]
            acc["dur"].append(dur); acc["T"].append(T); acc["over"].append(max(0.0, dur - T))
            figs = [i for seg in sb["segments"] for i in (seg.get("resource_ids") or []) if i.startswith("FIG")]
            scene = (Path(r3["video"]).parent / "final_scene.py").read_text(encoding="utf-8")
            shown = [f for f in set(figs) if re.search(rf"ImageMobject\([^)]*{f}", scene) or f in scene]
            acc["fig"].append(len(shown) / len(set(figs)) if figs else None)
            per[f"{cid} {q}"] = {"gsf1": round(acc["gsf1"][-1], 3), "entailed": round(acc["ent"][-1], 3),
                                 "claims": tot, "dur": round(dur, 1), "T": T, "figs": [len(set(figs)), len(shown)]}
            print(cid, q, per[f"{cid} {q}"], flush=True)
        m = lambda xs: sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs))
        rows.append({"id": cid, "stage1": s1, "stage2": s2, "gsf1": round(m(acc["gsf1"]), 3),
                     "entailed": round(m(acc["ent"]), 3), "over": round(m(acc["over"]), 1),
                     "dur": round(m(acc["dur"]), 1), "T": round(m(acc["T"]), 1),
                     "figures_shown": round(m(acc["fig"]), 3) if any(x is not None for x in acc["fig"]) else None})
    out = {"configs": rows, "per_run": per}
    (HERE / "e1_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['id']:6s} {r['stage1']:8s} {r['stage2']:4s} GS-F1 {r['gsf1']:.3f}  Entailed {r['entailed']:.3f}  "
              f"Over {r['over']:.1f}  (mean {r['dur']:.1f}s of {r['T']:.0f}s)  figures shown {r['figures_shown']}")


if __name__ == "__main__":
    main()
