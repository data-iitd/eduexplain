"""
Deterministic (non-LLM) helpers used by pipeline.py.
No torch / transformers imports here, so this module is cheap to import and test.

Issue schema used everywhere (tool checks AND verifier LLM):
    {"segment": int (0 = global), "severity": "critical"|"major"|"minor",
     "category": str, "problem": str, "fix": str, "source": "static"|"exec"|"verifier"}
"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from prompt import plan_beats

SEVERITIES = ("critical", "major", "minor")


# ----------------------------------------------------------------- storyboard
def load_storyboard(path, index=0) -> dict:
    """Accepts the wrapped format ({"storyboard": {...}}) or a bare storyboard,
    a single object or a list of them, and tolerates a trailing comma."""
    text = Path(path).read_text(encoding="utf-8").strip().rstrip(",")
    data = json.loads(text)
    if isinstance(data, list):
        data = data[index]
    sb = data.get("storyboard", data)
    if "segments" not in sb:
        raise ValueError("No 'segments' found in storyboard")
    return sb


def ordered_segments(storyboard):
    return sorted(storyboard["segments"], key=lambda s: s.get("order", 0))


# ------------------------------------------------------------------ extraction
def apply_timeline(storyboard: dict, timeline: dict) -> dict:
    """
    Copy of the storyboard whose segments carry the measured narration timing from tts.build_narration:
    `duration_sec` becomes the segment's audio length (the storyboard value unless speech had to extend it)
    and `beats` holds the real sentence start/end times. Prompts and checks then use the recorded audio.
    """
    sb = json.loads(json.dumps(storyboard))
    timed = {s["segment_id"]: s for s in timeline.get("segments", [])}
    for seg in sb["segments"]:
        t = timed.get(seg["segment_id"])
        if t is None:
            continue
        seg["duration_sec"] = t["duration_sec"]
        seg["beats"] = [{"start": b["start"], "end": b["end"], "text": b["text"]} for b in t["beats"]]
        seg["windows"] = list(t.get("windows", []))
    sb["total_duration_sec"] = round(sum(float(s["duration_sec"]) for s in sb["segments"]), 2)
    return sb


def extract_code(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", raw, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # unterminated block (hit the token limit)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", raw, flags=re.DOTALL)
    return (m.group(1) if m else raw).strip()


# ------------------------------------------------------------ static validation
def validate(code: str, storyboard: dict) -> dict:
    """Cheap static checks. Returns {"syntax_ok": bool, "issues": [str]}."""
    report = {"syntax_ok": False, "issues": []}
    try:
        tree = ast.parse(code)
        report["syntax_ok"] = True
    except SyntaxError as e:
        report["issues"].append(f"SyntaxError line {e.lineno}: {e.msg}")
        return report

    if "from manim import" not in code:
        report["issues"].append("missing `from manim import *`")
    if not any(isinstance(n, ast.ClassDef) and n.name == "StoryboardScene" for n in ast.walk(tree)):
        report["issues"].append("class StoryboardScene not found")
    if re.search(r"\b(x_shift|y_shift)\s*=", code):
        report["issues"].append("uses invented kwarg x_shift/y_shift (runtime TypeError); use .shift(...)")
    if re.search(r"\bText\([^)]*\balign\s*=", code):
        report["issues"].append("Text(...) has no `align` argument (runtime TypeError); use \\n line breaks")
    if re.search(r"\b(MathTex|Tex)\(", code):
        report["issues"].append("uses Tex/MathTex (needs LaTeX)")

    methods = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    segs = ordered_segments(storyboard)
    if "construct" not in methods:
        report["issues"].append("CRITICAL: no construct() method - scene renders nothing")
    else:
        for i in range(1, len(segs) + 1):
            if f"self.segment_{i}()" not in code:
                report["issues"].append(f"construct() never calls segment_{i}()")

    pads = [float(x) for x in re.findall(r"self\.pad\(\s*t0\s*,\s*([0-9]+(?:\.[0-9]+)?)", code)]
    for i, seg in enumerate(segs, 1):
        if f"segment_{i}" not in methods:
            report["issues"].append(f"missing segment_{i} ({seg['segment_id']})")
        else:
            dur = float(seg["duration_sec"])
            # accept pad(t0, dur) or a slightly smaller value (to leave room for a fade-out)
            if not any(dur - 1.0 <= p <= dur for p in pads):
                report["issues"].append(f"segment_{i}: no self.pad(t0, {seg['duration_sec']}) found")
    return report


def _segment_of(msg):
    m = re.search(r"segment_(\d+)", msg)
    return int(m.group(1)) if m else 0


_SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")


def _norm(text: str) -> str:
    text = text.translate(_SUBSCRIPTS).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(".")


def required_texts(seg):
    """Phrases quoted in a segment's on_screen ('like this'); these must appear exactly (modulo line breaks)."""
    found = re.findall(r"(?<!\w)'([^']{4,}?)'(?!\w)", seg.get("on_screen", ""))
    return [f for f in found if len(f.split()) >= 1]


def fidelity_issues(code: str, storyboard: dict):
    """Deterministic check that every quoted on_screen text appears in the matching segment method."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    blobs = {}
    for node in ast.walk(tree):
        m = re.fullmatch(r"segment_(\d+)", getattr(node, "name", "")) if isinstance(node, ast.FunctionDef) else None
        if m:
            strs = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            blobs[int(m.group(1))] = " ".join(_norm(x) for x in strs)
    out = []
    for i, seg in enumerate(ordered_segments(storyboard), 1):
        if i not in blobs:
            continue
        for phrase in required_texts(seg):
            if _norm(phrase) not in blobs[i]:
                out.append({
                    "segment": i, "severity": "major", "category": "fidelity",
                    "problem": f"segment_{i} must show the text \"{phrase}\" (from on_screen) but it was not found.",
                    "fix": "Add a Text(...) with exactly these words (line breaks allowed).",
                    "source": "static",
                })
    return out


def figure_issues(code: str, storyboard: dict):
    """Segments with FIG resources must load that exact image with ImageMobject inside their own method."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    bodies = {}
    for node in ast.walk(tree):
        m = re.fullmatch(r"segment_(\d+)", getattr(node, "name", "")) if isinstance(node, ast.FunctionDef) else None
        if m:
            bodies[int(m.group(1))] = ast.get_source_segment(code, node) or ""
    out = []
    for i, seg in enumerate(ordered_segments(storyboard), 1):
        mine = {f["file"] for f in seg.get("figures", []) if not f.get("missing")}
        foreign = sorted({m for m in re.findall(r"figures/(FIG_\w+\.png)", bodies.get(i, "")) if m not in mine})
        if foreign:   # a figure belongs to the segment whose resource_ids list it
            out.append({"segment": i, "severity": "major", "category": "fidelity",
                        "problem": f"segment_{i} shows figure(s) {', '.join(foreign)} that belong to another segment.",
                        "fix": "Remove that ImageMobject: show a figure only in the segment that lists it under FIGURES.",
                        "source": "static"})
        for fig in seg.get("figures", []):
            if fig.get("missing"):
                continue
            body = bodies.get(i, "")
            if "ImageMobject" not in body or fig["file"] not in body:
                out.append({
                    "segment": i, "severity": "major", "category": "fidelity",
                    "problem": f"segment_{i} must show the figure {fig['id']} but does not load it with "
                               f"ImageMobject(r\"{fig['path']}\").",
                    "fix": f"img = ImageMobject(r\"{fig['path']}\").set(height=5); FadeIn it at its beat and annotate "
                           f"it as on_screen says (rule 17).",
                    "source": "static",
                })
    return out


NARRATION_COPY_WORDS = 7


def _norm_words(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def narration_copy_issues(code: str, storyboard: dict):
    """On-screen strings that copy the segment's narration (>= NARRATION_COPY_WORDS words, not quoted in on_screen).
    The voice-over already says them: a screen of narration captions is a transcript, not an explanation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    segs = ordered_segments(storyboard)
    out = []
    for node in ast.walk(tree):
        m = re.fullmatch(r"segment_(\d+)", node.name) if isinstance(node, ast.FunctionDef) else None
        if not m or not 1 <= int(m.group(1)) <= len(segs):
            continue
        i = int(m.group(1))
        seg = segs[i - 1]
        narr = " " + _norm_words(seg.get("narration", "")) + " "
        allowed = [_norm_words(t) for t in required_texts(seg)]
        copies = []
        for c in ast.walk(node):
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                k = _norm_words(c.value)
                if (len(k.split()) >= NARRATION_COPY_WORDS and f" {k} " in narr
                        and not any(k in a or a in k for a in allowed)):
                    copies.append(c.value.replace("\n", " "))
        if copies:
            listed = "; ".join(f'"{t[:60]}"' for t in copies[:3])
            out.append({
                "segment": i, "severity": "major", "category": "narration",
                "problem": (f"segment_{i} writes {len(copies)} narration sentence(s) on screen as Text: {listed}. The teacher "
                            "voice-over already says them; the screen must not be a transcript."),
                "fix": ("Delete these Text objects. Show only what on_screen describes: the figure, highlights and "
                        "short labels or key terms (at most about 5 words each), animated in time with the narration."),
                "source": "static",
            })
    return out


def static_issues(code: str, storyboard: dict, subtitles: bool = False):
    """validate() output converted to the common issue schema, plus the quoted-text check."""
    out = []
    for msg in validate(code, storyboard)["issues"]:
        critical = any(k in msg for k in ("SyntaxError", "CRITICAL", "construct", "StoryboardScene",
                                           "x_shift", "align", "missing segment_", "from manim"))
        out.append({
            "segment": _segment_of(msg),
            "severity": "critical" if critical else "major",
            "category": "structure",
            "problem": msg,
            "fix": "Fix this so the script matches the required structure.",
            "source": "static",
        })
    if re.search(r"\badd_sound\s*\(", code):
        out.append({
            "segment": 0, "severity": "major", "category": "structure",
            "problem": "The script calls add_sound(); the teacher voice-over is added by the renderer, so audio would play twice.",
            "fix": "Remove every add_sound(...) call. Do not load any audio files.",
            "source": "static",
        })
    out += [] if subtitles else narration_copy_issues(code, storyboard)
    return out + fidelity_issues(code, storyboard) + figure_issues(code, storyboard)


# --------------------------------------------------------------------- exec check
def run_exec_check(scene_path, timeout=300):
    """Run exec_check.py in a subprocess (own interpreter, timeout) and return its report."""
    scene_path = Path(scene_path).resolve()
    report_path = scene_path.with_suffix(".exec.json")
    if report_path.exists():
        report_path.unlink()
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("exec_check.py")), str(scene_path), str(report_path)],
            capture_output=True, text=True, timeout=timeout, cwd=scene_path.parent,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error_type": "Timeout",
                "error_message": f"Scene did not finish within {timeout}s (infinite loop or extremely heavy animation)."}
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"ok": False, "error_type": "RunnerCrash", "error_message": (proc.stderr or proc.stdout)[-1500:]}


# Legacy (manimlib / old Manim CE) calls the generator keeps writing, rewritten before any check: one keyword pair
# becomes the range list current Manim CE expects, and renamed animation/text classes get their current name.
_ARG = r"(-?[\w.]+(?:\s*[*/+-]\s*-?[\w.]+)*)"
_RANGE_PAIRS = [(rf"\b{a}_min\s*=\s*{_ARG}\s*,\s*{a}_max\s*=\s*{_ARG}", rf"{a}_range=[\1, \2]") for a in "txy"]
_RENAMES = {"ShowCreation": "Create", "TextMobject": "Text", "ShowCreationThenDestruction": "ShowPassingFlash"}


def modernize_api(code):
    """Rewrite known legacy Manim API in generated code. Returns (code, list of rewrites)."""
    done = []
    for pat, rep in _RANGE_PAIRS:
        code, n = re.subn(pat, rep, code)
        if n:
            done.append(f"{rep.split('=')[0]}: {n}x")
    for old, new in _RENAMES.items():
        code, n = re.subn(rf"\b{old}\(", f"{new}(", code)
        if n:
            done.append(f"{old} -> {new}: {n}x")
    return code, done


_NAME_HINTS = {
    "Wave": "Manim CE has no Wave class: draw a wave as ParametricFunction(lambda t: np.array([t, A * np.sin(k * t), 0]), "
            "t_range=[a, b], color=...) or FunctionGraph(lambda x: A * np.sin(k * x), x_range=[a, b]).",
    "SineWave": "Manim CE has no SineWave class: use FunctionGraph(lambda x: A * np.sin(k * x), x_range=[a, b]).",
    "MathTex": "LaTeX is not installed: use Text(...) (Unicode symbols such as ² ₂ → are fine).",
    "Tex": "LaTeX is not installed: use Text(...) (Unicode symbols such as ² ₂ → are fine).",
    "Spiral": "Manim CE has no Spiral class: draw one as ParametricFunction(lambda t: np.array([r * t * np.cos(t), "
              "r * t * np.sin(t), 0]), t_range=[0, 4 * PI]).",
}


def _runtime_fix_hint(etype, emsg):
    m = re.search(r"unexpected keyword argument '(\w+)'", emsg)
    if m and m.group(1)[0] in "txy" and m.group(1)[1:] in ("_min", "_max"):
        a = m.group(1)[0]
        return (f"Manim CE takes {a}_range=[start, end] (optionally [start, end, step]) instead of "
                f"{a}_min/{a}_max. Replace both keywords with {a}_range.")
    m = re.search(r"name '(\w+)' is not defined", emsg)
    if m and m.group(1) in _NAME_HINTS:
        return _NAME_HINTS[m.group(1)]
    m = re.search(r"Unexpected argument (\w+) passed to Scene\.play", emsg)
    if m:
        return (f"self.play() takes animations, not a {m.group(1)}: a mobject method such as .move_to(...), "
                ".set_color(...) or .shift(...) changes the object instantly and returns the object. Animate it "
                "with self.play(obj.animate.move_to(...)), or wrap the object in an animation (FadeIn, Create, ...).")
    if etype == "TypeError" and "unexpected keyword argument" in emsg:
        return ("Remove the invalid keyword argument. Use only documented Manim CE arguments "
                "(offset shapes with .shift(...), set text alignment with \\n line breaks).")
    if etype in ("NameError", "AttributeError", "ImportError"):
        return "Use only names/methods that exist in Manim Community Edition; replace or remove the unknown one."
    if etype == "Timeout":
        return "Remove infinite loops / reduce the number of animated objects."
    return "Fix the exception shown; keep the rest of the script unchanged."


def sync_issues(rec, seg, early=0.75, late=2.0):
    """Beats with no animation starting in [start - early, start + late]: the visual does not follow the voice."""
    if not seg.get("beats") or "play_starts" not in rec:
        return []
    i, plays = rec["index"], rec["play_starts"]
    missed = [b for b in plan_beats(seg) if not any(b["start"] - early <= p <= b["start"] + late for p in plays)]
    if not missed:
        return []
    listed = "; ".join(f"t={b['start']:.1f}s \"{b['text'][:50]}\"" for b in missed[:4])
    return [{
        "segment": i, "severity": "major" if len(missed) > 1 else "minor", "category": "sync",
        "problem": (f"Segment {i}: {len(missed)} narration beat(s) have no animation starting when the teacher says "
                    f"them: {listed}."),
        "fix": "Before each of those beats call self.until(t0, <beat start>) and play the visual for that sentence.",
        "source": "exec",
    }]


def window_issues(rec, seg, min_cover=0.6):
    """Silent visual windows (teacher pauses, see tts budget pacing) must be filled with motion: at least
    `min_cover` of each window has to be covered by running animations, otherwise the screen looks frozen."""
    spans, out = rec.get("play_spans"), []
    if not seg.get("windows") or spans is None:
        return []
    for w in seg["windows"]:
        length = w["end"] - w["start"]
        covered, cursor = 0.0, w["start"]
        for a, b in sorted(spans):
            a, b = max(a, cursor), min(b, w["end"])
            if b > a:
                covered += b - a
                cursor = b
        if length > 0 and covered / length < min_cover:
            out.append(f"t={w['start']:.1f}-{w['end']:.1f}s ({w['kind']}, {100 * covered / length:.0f}% animated)")
    if not out:
        return []
    return [{
        "segment": rec["index"], "severity": "major", "category": "sync",
        "problem": f"Segment {rec['index']}: the teacher is silent in visual window(s) {'; '.join(out[:4])} "
                   f"but the screen is mostly still.",
        "fix": "Fill each VISUAL WINDOW with one continuous demonstration animation that spans it: self.until(t0, "
               "<window start>) then play e.g. electrons moving into the shared pair, bonds drawn one by one, "
               "a LaggedStart build-up or a recap highlight with run_time close to the window length.",
        "source": "exec",
    }]


OVERRUN_OK = 1.2   # segment overrun the render's time-budget guard absorbs without blocking approval


def evaluate_report(report, storyboard, timing_tol=1.0, max_gap_sec=8.0, total_tol=0.5):
    """Turn an exec_check report into issues. Pure function (unit-testable)."""
    segs = ordered_segments(storyboard)
    expected = [float(s["duration_sec"]) for s in segs]
    issues = []

    if not report.get("ok"):
        etype, emsg = report.get("error_type", "Error"), report.get("error_message", "")
        where = ""
        if report.get("error_line"):
            where = f" at line {report['error_line']}: `{report.get('error_code', '').strip()}`"
        issues.append({
            "segment": int(report.get("error_segment") or 0),
            "severity": "critical",
            "category": "api",
            "problem": f"Runtime error {etype}: {emsg}{where}",
            "fix": _runtime_fix_hint(etype, emsg),
            "source": "exec",
        })
        return issues

    recs = report.get("segments", [])
    if len(recs) != len(expected):
        issues.append({
            "segment": 0, "severity": "critical", "category": "structure",
            "problem": f"{len(recs)} segments ran but the storyboard has {len(expected)}.",
            "fix": "construct() must call segment_1..segment_N, one per storyboard segment.",
            "source": "exec",
        })

    for rec in recs:
        i = rec["index"]
        if i > len(expected):
            continue
        exp = expected[i - 1]
        dur = rec["end"] - rec["start"]
        # the fade-out of the previous segment is allowed to eat into the next one, so use a
        # cumulative check below; here we only flag large per-segment deviations
        if abs(dur - exp) > timing_tol:
            if dur > exp:
                fix = (f"The animations in segment_{i} add up to more than {exp:g}s. Shorten run_time/wait values so "
                       f"they total less than {exp:g}s, then end with self.pad(t0, {exp:g}).")
            else:
                fix = f"End segment_{i} with self.pad(t0, {exp:g}) so it lasts exactly {exp:g}s."
            # a small overrun never reaches the video: render.py's time-budget guard plays the segment exp/dur
            # faster (<= 1.2x here, barely visible), so it does not block approval; bigger ones and short segments do
            small = exp < dur <= exp * OVERRUN_OK
            issues.append({
                "segment": i, "severity": "minor" if small else "major", "category": "timing",
                "problem": f"Segment {i} lasts {dur:.1f}s but the storyboard says {exp:g}s."
                           + (f" (the render will play it {dur / exp:.2f}x faster to keep the budget)" if small else ""),
                "fix": fix, "source": "exec",
            })
        if rec.get("mobjects_at_end", 1) == 0:
            issues.append({
                "segment": i, "severity": "major", "category": "pacing",
                "problem": f"Segment {i} ends with an empty screen (content removed before the padding wait).",
                "fix": "Call self.pad(t0, duration) BEFORE any FadeOut; fade out at the start of the next segment.",
                "source": "exec",
            })
        elif rec.get("max_gap_sec", 0.0) > max_gap_sec:
            g, g0 = rec["max_gap_sec"], rec.get("max_gap_start", 0.0)
            beats = [b for b in plan_beats(segs[i - 1]) if g0 + 1.0 <= b["start"] <= g0 + g - 1.0]
            hint = "; ".join(f"t={b['start']:.1f}s \"{b['text'][:45]}\"" for b in beats[:3])
            issues.append({
                "segment": i, "severity": "major", "category": "pacing",
                "problem": (f"Segment {i}: nothing changes on screen from t={g0:.1f}s to t={g0 + g:.1f}s "
                            f"({g:.1f}s frozen; limit {max_gap_sec:g}s)."),
                "fix": ("Reveal the next visual at the narration beats inside that window"
                        + (f": {hint}." if hint else ".")
                        + " Use self.until(t0, <beat start>) before each reveal."),
                "source": "exec",
            })
        issues += sync_issues(rec, segs[i - 1])
        issues += window_issues(rec, segs[i - 1])
        clashes = rec.get("text_on_shape", [])
        if clashes:
            issues.append({
                "segment": i, "severity": "major", "category": "layout",
                "problem": f"Labels collide with shapes in segment {i}: "
                           + ", ".join(f'"{a}" overlaps a {b}' for a, b in clashes) + ".",
                "fix": "Give every label clear space: place it with .next_to(<the whole group>, DOWN/UP, buff=0.35) "
                       "relative to the group it must not touch (not relative to another label), stack labels with "
                       "VGroup(...).arrange(DOWN, buff=0.3), and keep text out of circles unless it is centred inside.",
                "source": "exec",
            })
        for label in rec.get("out_of_frame", []):
            issues.append({
                "segment": i, "severity": "major", "category": "layout",
                "problem": f"Text \"{label}\" extends outside the frame in segment {i}.",
                "fix": "Reduce font_size, break the text over lines with \\n, or call .scale_to_fit_width(12).",
                "source": "exec",
            })
        figs = [f for f in segs[i - 1].get("figures", []) if not f.get("missing")]
        if figs and "images" in rec:
            shown = [im for im in rec["images"] if im["opacity"] > 0.3]
            if len(shown) < len(figs):
                issues.append({
                    "segment": i, "severity": "major", "category": "fidelity",
                    "problem": f"Segment {i} must keep its figure(s) {', '.join(f['id'] for f in figs)} on screen, but "
                               f"{len(shown)} image(s) are visible at the end of the segment.",
                    "fix": "Add the ImageMobject with self.play(FadeIn(img)) at its beat and do not remove it before "
                           "self.pad.", "source": "exec"})
            elif min(im["height_frac"] for im in shown) < 0.4:
                issues.append({
                    "segment": i, "severity": "major", "category": "layout",
                    "problem": f"The figure in segment {i} is too small to read "
                               f"({100 * min(im['height_frac'] for im in shown):.0f}% of the frame height).",
                    "fix": "Scale it with img.set(height=5) (at most width 12) so its details and annotations are readable.",
                    "source": "exec"})
        shapes = rec.get("shapes_out_of_frame", [])
        if shapes:
            issues.append({
                "segment": i, "severity": "major", "category": "layout",
                "problem": (f"Shapes extend outside the frame in segment {i} (frame x in [-7.1, 7.1], y in [-4, 4]): "
                            + ", ".join(shapes) + "."),
                "fix": "Shrink or re-space that group so it fits: e.g. VGroup(...).arrange(RIGHT, buff=0.8) then "
                       ".scale_to_fit_width(12) if it is wider than 12, and keep it centred.",
                "source": "exec",
            })
        for a, b in rec.get("overlaps", []):
            issues.append({
                "segment": i, "severity": "major", "category": "layout",
                "problem": f"Texts \"{a}\" and \"{b}\" overlap in segment {i}.",
                "fix": "Reposition with next_to()/to_edge()/arrange() so they do not overlap.",
                "source": "exec",
            })

    total_exp = sum(expected)
    total = report.get("total_time")
    if total is not None and abs(total - total_exp) > total_tol and not any(
            i["category"] == "timing" for i in issues):
        issues.append({
            "segment": 0, "severity": "major", "category": "timing",
            "problem": f"Total runtime is {total:.1f}s but the storyboard total is {total_exp:g}s.",
            "fix": "Make sure every segment ends with self.pad(t0, duration_sec).",
            "source": "exec",
        })
    return issues


# --------------------------------------------------------------------- formatting
def sort_issues(issues):
    return sorted(issues, key=lambda i: (SEVERITIES.index(i["severity"]), i.get("segment", 0)))


def is_blocking(issues):
    return any(i["severity"] in ("critical", "major") for i in issues)


def count_by_severity(issues):
    return {s: sum(1 for i in issues if i["severity"] == s) for s in SEVERITIES}


def format_issues(issues, only=("critical", "major")):
    lines = []
    for n, i in enumerate([x for x in sort_issues(issues) if x["severity"] in only], 1):
        seg = f"segment {i['segment']}" if i.get("segment") else "global"
        lines.append(f"{n}. [{i['severity'].upper()}] [{seg}] [{i['category']}] {i['problem']}\n   FIX: {i['fix']}")
    return "\n".join(lines) if lines else "(none)"


def format_tool_report(issues, report, storyboard):
    """Compact text version of the automatic checks for the verifier LLM."""
    segs = ordered_segments(storyboard)
    lines = []
    if report is None:
        lines.append("Automatic runner: NOT RUN (static checks only).")
    elif report.get("ok"):
        lines.append(f"Automatic runner: scene executed successfully, total {report.get('total_time', 0):.1f}s "
                     f"(storyboard {sum(float(s['duration_sec']) for s in segs):g}s).")
        for rec in report.get("segments", []):
            i = rec["index"]
            exp = segs[i - 1]["duration_sec"] if i <= len(segs) else "?"
            lines.append(f"  segment {i}: lasts {rec['end'] - rec['start']:.1f}s (expected {exp}s), "
                         f"longest frozen stretch {rec.get('max_gap_sec', 0):.1f}s, objects on screen at end: {rec.get('mobjects_at_end')}")
    else:
        lines.append("Automatic runner: scene CRASHED (see issues).")
    tool = [i for i in issues if i["source"] in ("static", "exec")]
    lines.append("Automatic issues:\n" + format_issues(tool, only=SEVERITIES))
    return "\n".join(lines)