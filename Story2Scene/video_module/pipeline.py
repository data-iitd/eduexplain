"""
Generator -> checks -> verifier -> fix loop for storyboard -> Manim code.

    python pipeline.py -s storyboards/storyboard.json \
        --generator Qwen3.8-27B --verifier Qwen3.5-9B

Before round 0 the narration is synthesized (tts.py, Kokoro) and its measured sentence timings replace the
estimated beats, so visuals are generated against the real teacher voice-over. With the default "cap" pacing each
segment lasts as long as the teacher talks (natural pauses), never longer than the storyboard's duration_sec.
Per round:
  1. generator LLM writes (or fixes) the script
  2. static checks, then a headless execution (exec_check.py) -> real errors/timing/layout
  3. if it does not crash: verifier LLM reviews it against the storyboard
  4. critical/major issues go back to the generator; stop when none are left or rounds run out
Finally the best script is rendered with the voice-over to final.mp4 (render.py).
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from checks import (apply_timeline, count_by_severity, modernize_api, ordered_segments, evaluate_report, extract_code, format_issues, format_tool_report,
                    is_blocking, load_storyboard, run_exec_check, static_issues)
from figures import attach_figures, segment_figure_ids
from prompt import build_fix_messages, build_messages, build_verifier_messages


@dataclass
class PipelineConfig:
    max_rounds: int = 3              # fix rounds AFTER the first generation
    gen_max_new_tokens: int = 8000
    ver_max_new_tokens: int = 1500
    gen_temperature: float = 0.2
    ver_temperature: float = 0.1
    gen_thinking: bool = False
    ver_thinking: bool = False
    subtitles: bool = False
    no_exec: bool = False            # skip headless execution (manim not installed)
    exec_timeout: int = 300
    timing_tol: float = 1.0          # allowed deviation per segment (seconds)
    max_gap_sec: float = 8.0         # longest allowed stretch where nothing changes on screen
    render: bool = False             # render final_scene.py (+ narration) to final.mp4
    render_quality: str = "m"        # manim -q flag: l|m|h|k
    figure_grounding: bool = True    # ask the vision verifier where on_screen's annotated parts are in each figure


# ------------------------------------------------------------------ verifier I/O
def parse_verifier(raw: str):
    """Return (approved, issues) or None if the output could not be parsed."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    issues = []
    for it in data.get("issues", [])[:8]:
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity", "major")).lower()
        try:
            seg = int(it.get("segment", 0))
        except (TypeError, ValueError):
            seg = 0
        issues.append({
            "segment": seg,
            "severity": sev if sev in ("critical", "major", "minor") else "major",
            "category": str(it.get("category", "fidelity")),
            "problem": str(it.get("problem", ""))[:400],
            "fix": str(it.get("fix", ""))[:400],
            "source": "verifier",
        })
    return bool(data.get("approved", False)), issues


HEDGE = re.compile(r"\b(may|might|could|potential(ly)?|possibl[ey]|likely|risk)\b", re.I)


def _dedupe(new, existing):
    seen = {(i["segment"], i["category"]) for i in existing if i["severity"] != "minor"}
    return [i for i in new if i["severity"] == "minor" or (i["segment"], i["category"]) not in seen]


def _score(issues):
    c = count_by_severity(issues)
    return (c["critical"], c["major"], c["minor"])


# ------------------------------------------------------------------------ loop
def run_pipeline(storyboard, generator, verifier, out_dir, cfg: PipelineConfig, log=print, narration_dir=None):
    """
    generator / verifier: objects with .generate(messages, max_new_tokens=, thinking=, temperature=) -> str
    narration_dir: output of tts.build_narration (timeline.json + audio/); its timings drive the prompts and checks.
    Returns the summary dict (also written to out_dir/pipeline_summary.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if narration_dir is not None:
        timeline = json.loads((Path(narration_dir) / "timeline.json").read_text(encoding="utf-8"))
        storyboard = apply_timeline(storyboard, timeline)
    # FIG_xxxx resources: copy the images, describe and (with a vision verifier) ground the parts to annotate
    if any(segment_figure_ids(s) for s in storyboard["segments"]):
        storyboard = attach_figures(storyboard, out_dir / "figures",
                                    grounder=verifier if cfg.figure_grounding else None, log=log)
    if narration_dir is not None or any(s.get("figures") for s in storyboard["segments"]):
        (out_dir / "storyboard_timed.json").write_text(json.dumps(storyboard, indent=2), encoding="utf-8")
    gen_messages = build_messages(storyboard, subtitles=cfg.subtitles)
    messages = gen_messages
    no_exec = cfg.no_exec
    rounds, best, status = [], None, "max_rounds"
    t_start = time.time()

    for rnd in range(cfg.max_rounds + 1):
        rdir = out_dir / f"round_{rnd}"
        rdir.mkdir(exist_ok=True)
        log(f"\n--- round {rnd} ({'initial generation' if rnd == 0 else 'fix'}) ---")
        rec = {"round": rnd}

        # 1. generate
        t = time.time()
        raw = generator.generate(messages, max_new_tokens=cfg.gen_max_new_tokens,
                                 thinking=cfg.gen_thinking, temperature=cfg.gen_temperature)
        rec["gen_sec"] = round(time.time() - t, 1)
        n_tok = getattr(generator, "last_new_tokens", None)
        if n_tok:
            rec["gen_tokens"] = int(n_tok)
            rec["gen_tok_per_sec"] = round(n_tok / max(rec["gen_sec"], 1e-6), 2)
        rec["gen_prompt_tail"] = getattr(generator, "last_prompt_tail", None)
        truncated = getattr(generator, "last_finish_reason", None) == "length"
        if truncated:
            rec["gen_truncated"] = True
            log(f"  ! generator output cut off at the token limit ({n_tok} tokens)")
        code, rewrites = modernize_api(extract_code(raw))
        if rewrites:   # legacy Manim API fixed deterministically (t_min/t_max -> t_range, ShowCreation -> Create ...)
            rec["api_rewrites"] = rewrites
            log(f"    legacy API rewritten: {', '.join(rewrites)}")
        (rdir / "generator_raw.txt").write_text(raw, encoding="utf-8")
        scene_path = rdir / "scene.py"
        scene_path.write_text(code, encoding="utf-8")

        # 2. static + execution checks
        issues = static_issues(code, storyboard, subtitles=cfg.subtitles)
        if truncated:   # the end of the script is missing: say why, or the generator only sees the symptoms
            issues.insert(0, {"segment": 0, "severity": "critical", "category": "structure", "source": "static",
                              "problem": f"Your script was cut off at the output limit ({n_tok} tokens), so its end is missing.",
                              "fix": "Write the whole script more compactly: no comments, reuse small helper methods for "
                                     "repeated shapes, fewer objects per segment. Every segment_N must be complete."})
        report = None
        if not any(i["severity"] == "critical" for i in issues) and not no_exec:
            t = time.time()
            report = run_exec_check(scene_path, timeout=cfg.exec_timeout)
            rec["exec_sec"] = round(time.time() - t, 1)
            if report.get("env_error"):
                log(f"  ! execution check unavailable ({report['env_error']}); continuing with static + LLM review only")
                no_exec, report = True, None
            else:
                (rdir / "exec_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                issues += evaluate_report(report, storyboard, cfg.timing_tol, cfg.max_gap_sec)
        rec["exec_ok"] = None if report is None else bool(report.get("ok"))
        if report and report.get("ok"):
            rec["total_time"] = round(report["total_time"], 1)

        # 3. verifier (only when the script did not crash: reviewing a crash wastes time)
        crashed = any(i["severity"] == "critical" for i in issues)
        if not crashed:
            t = time.time()
            v_msgs = build_verifier_messages(storyboard, code, format_tool_report(issues, report, storyboard))
            v_raw = verifier.generate(v_msgs, max_new_tokens=cfg.ver_max_new_tokens,
                                      thinking=cfg.ver_thinking, temperature=cfg.ver_temperature)
            rec["verify_sec"] = round(time.time() - t, 1)
            v_tok = getattr(verifier, "last_new_tokens", None)
            if v_tok:
                rec["verify_tokens"] = int(v_tok)
            (rdir / "verifier_raw.txt").write_text(v_raw, encoding="utf-8")
            parsed = parse_verifier(v_raw)
            if parsed is None:
                rec["verifier_parsed"] = False
                log("  ! verifier output was not valid JSON; relying on automatic checks this round")
            else:
                rec["verifier_parsed"] = True
                v_issues = parsed[1]
                if report is not None and report.get("ok"):
                    # The scene ran: API/syntax/timing/pacing/frame-overflow are settled by the runner.
                    # A small verifier model is unreliable there (it hallucinates "class X does not exist"),
                    # so keep only arrangement + narration-sync issues and never let it raise a critical.
                    kept = [i for i in v_issues if i["category"] in ("layout", "sync")]
                    ignored = [i for i in v_issues if i not in kept]
                    rec["verifier_ignored"] = len(ignored)
                    if ignored:
                        (rdir / "verifier_ignored.json").write_text(json.dumps(ignored, indent=2), encoding="utf-8")
                    # with layout_fix active the runner measured the layout AFTER run-time placement; the verifier
                    # only reads the source, i.e. the positions before the fix, so its layout claims are stale
                    measured = any("labels_moved" in s for s in report.get("segments", []))
                    for i in kept:
                        if i["severity"] == "critical":
                            i["severity"] = "major"
                        if i["category"] == "layout" and measured and i["severity"] == "major":
                            i["severity"] = "minor"
                            i["problem"] += " (read from the source; the run-time layout after layout_fix is measured by exec_check)"
                        # overlaps/frame bounds were measured exactly by the runner; a hedged layout guess
                        # ("may overlap", "potential ...") from the small verifier must not block approval
                        if i["category"] == "layout" and HEDGE.search(i["problem"]):
                            i["severity"] = "minor"
                    v_issues = kept
                issues += _dedupe(v_issues, issues)
        else:
            rec["verifier_parsed"] = None  # skipped

        rec["issues"] = count_by_severity(issues)
        (rdir / "issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")
        feedback = format_issues(issues)
        (rdir / "feedback.txt").write_text(feedback, encoding="utf-8")
        rounds.append(rec)
        log(f"  exec_ok={rec['exec_ok']}  issues={rec['issues']}  gen={rec['gen_sec']}s")

        sc = _score(issues)
        if best is None or sc < best["score"]:
            best = {"round": rnd, "score": sc, "code": code, "report": report}

        # 4. decide
        if not is_blocking(issues):
            status = "approved"
            break
        if rnd == cfg.max_rounds:
            break
        messages = build_fix_messages(gen_messages, code, feedback)

    (out_dir / "final_scene.py").write_text(best["code"], encoding="utf-8")
    render = None
    if cfg.render:
        from render import render_video
        log(f"  rendering final_scene.py (-q{cfg.render_quality}{', with narration' if narration_dir else ''}) ...")
        # time budget is guaranteed at render: speed up any segment that overran its duration
        scale = {}
        rep = best.get("report") or {}
        if rep.get("ok"):
            budgets = [float(s["duration_sec"]) for s in ordered_segments(storyboard)]
            for rec in rep.get("segments", []):
                i, took = rec["index"], rec["end"] - rec["start"]
                if i <= len(budgets) and took > budgets[i - 1] + 0.05:
                    scale[i] = max(0.2, (budgets[i - 1] - 0.15) / took)
            if scale:
                log("  time budget: speeding up overrunning segment(s) " +
                    ", ".join(f"{i} x{1 / f:.2f}" for i, f in scale.items()))
        render = render_video(out_dir / "final_scene.py", out_dir, narration_dir, cfg.render_quality, time_scale=scale)
        log(f"  render ok={render.get('ok')}  video={render.get('video')}  "
            f"video_sec={render.get('video_sec')}  audio_sec={render.get('audio_sec')}")
    summary = {
        "status": status,
        "best_round": best["round"],
        "rounds_used": len(rounds),
        "total_sec": round(time.time() - t_start, 1),
        "execution_check": not no_exec,
        "narration": str(narration_dir) if narration_dir else None,
        "render": render,
        "timing_sec": {
            "generation": round(sum(r.get("gen_sec", 0) for r in rounds), 1),
            "exec_checks": round(sum(r.get("exec_sec", 0) for r in rounds), 1),
            "verifier": round(sum(r.get("verify_sec", 0) for r in rounds), 1),
            "render": render.get("render_sec") if render else None,
            "loop_and_render": round(time.time() - t_start, 1),
        },
        "config": asdict(cfg),
        "rounds": rounds,
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"\n=== {status.upper()} | best round {best['round']} | final code: {out_dir / 'final_scene.py'}")
    return summary


# -------------------------------------------------------------------------- CLI
def main():
    from model_loader import client_from_env, share_gpu

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--storyboard", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--generator", default="Qwen3.8-27B", help="vLLM served-model name for generator")
    ap.add_argument("--verifier", default="Qwen3.5-9B", help="vLLM served-model name for verifier")
    ap.add_argument("--gen-url", default=None, help="generator OpenAI-compatible base URL, e.g. http://host:8000/v1")
    ap.add_argument("--ver-url", default=None, help="verifier OpenAI-compatible base URL, e.g. http://host:8001/v1")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--gen-max-new-tokens", type=int, default=8000)
    ap.add_argument("--ver-max-new-tokens", type=int, default=1500)
    ap.add_argument("--gen-temperature", type=float, default=0.2)
    ap.add_argument("--ver-temperature", type=float, default=0.1)
    ap.add_argument("--gen-thinking", action="store_true")
    ap.add_argument("--ver-thinking", action="store_true")
    ap.add_argument("--subtitles", action="store_true")
    ap.add_argument("--no-exec", action="store_true", help="skip the headless Manim execution check")
    ap.add_argument("--exec-timeout", type=int, default=300)
    ap.add_argument("--timing-tol", type=float, default=1.0)
    ap.add_argument("--max-gap-sec", type=float, default=8.0)
    ap.add_argument("--no-narration", action="store_true", help="skip TTS; beats are estimated from word counts")
    ap.add_argument("--tts-engine", choices=["parler", "kokoro"], default="parler",
                    help="parler: AI4Bharat Indic Parler-TTS, Indian-English female teacher (venv-tts, GPU); "
                         "kokoro: Kokoro-82M on CPU")
    ap.add_argument("--voice", default=None, help="speaker (Parler, e.g. a named Indian-English speaker) or "
                                                  "Kokoro voice id (default af_heart / hf_alpha ...)")
    ap.add_argument("--tts-speed", type=float, default=0.92, help="teacher speaking rate (1.0 = engine default)")
    ap.add_argument("--pacing", choices=["cap", "budget", "natural", "storyboard"], default="cap",
                    help="cap (default): natural speech, each segment ends when the teacher is done but never after "
                         "its storyboard duration_sec (total <= storyboard total); budget: every segment lasts "
                         "exactly duration_sec, spare time as animated visual windows; natural: video as long as "
                         "the speech; "
                         "storyboard: legacy even pauses")
    ap.add_argument("--no-render", action="store_true", help="do not render final.mp4")
    ap.add_argument("-q", "--quality", choices=["l", "m", "h", "k"], default="m", help="render quality")
    ap.add_argument("--no-figure-grounding", action="store_true",
                    help="do not ask the vision verifier to locate annotated parts of FIG_ figures")
    ap.add_argument("--no-gpu-swap", action="store_true",
                    help="do not sleep/wake the two vLLM servers (use when they are on separate GPUs)")
    args = ap.parse_args()

    storyboard = load_storyboard(args.storyboard, args.index)
    cfg = PipelineConfig(
        max_rounds=args.max_rounds, gen_max_new_tokens=args.gen_max_new_tokens,
        ver_max_new_tokens=args.ver_max_new_tokens, gen_temperature=args.gen_temperature,
        ver_temperature=args.ver_temperature, gen_thinking=args.gen_thinking, ver_thinking=args.ver_thinking,
        subtitles=args.subtitles, no_exec=args.no_exec, exec_timeout=args.exec_timeout,
        timing_tol=args.timing_tol, max_gap_sec=args.max_gap_sec,
        render=not args.no_render, render_quality=args.quality,
        figure_grounding=not args.no_figure_grounding,
    )
    generator = client_from_env("generator", args.generator, args.gen_url)
    verifier = client_from_env("verifier", args.verifier, args.ver_url)
    if not args.no_gpu_swap:
        share_gpu(generator, verifier)
    sb_id = storyboard.get("storyboard_id", "storyboard")
    narration_dir, t_wall, tts_sec = None, time.time(), None
    if not args.no_narration:
        from tts import KokoroTTS, build_narration
        narration_dir = Path(args.out_dir) / sb_id / "narration"
        if args.tts_engine == "parler":
            # Parler runs on the GPU: make room by putting both LLM servers to sleep (they wake on demand)
            for c in (generator, verifier):
                try:
                    c.sleep()
                except Exception as exc:
                    print(f"  ! could not put {c.base_url} to sleep before TTS: {exc}")
            from tts_parler import IndicParlerTTS
            tts = IndicParlerTTS(speed=args.tts_speed, **({"voice": args.voice} if args.voice else {}))
        else:
            tts = KokoroTTS(voice=args.voice or "af_heart", speed=args.tts_speed)
        try:
            tl = build_narration(storyboard, narration_dir, tts, pacing=args.pacing)
        finally:
            getattr(tts, "close", lambda: None)()   # frees the TTS GPU memory before the LLMs run
        tts_sec = round(time.time() - t_wall, 1)
        print(f"Narration: {tl['total_sec']}s of teacher voice-over in {narration_dir} ({tts_sec}s)")
    out_dir = Path(args.out_dir) / sb_id / f"pipeline__{args.generator}__{args.verifier}"
    print(f"Generator endpoint: {generator.base_url} (model={generator.model})")
    print(f"Verifier endpoint:  {verifier.base_url} (model={verifier.model})")
    summary = run_pipeline(storyboard, generator, verifier, out_dir, cfg, narration_dir=narration_dir)
    summary["timing_sec"].update(tts=tts_sec, total_wall=round(time.time() - t_wall, 1))
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Timing (s): " + ", ".join(f"{k}={v}" for k, v in summary["timing_sec"].items()))


if __name__ == "__main__":
    main()
