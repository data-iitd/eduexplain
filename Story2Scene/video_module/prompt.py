"""
Prompt templates for storyboard -> Manim code generation.
Edit this file to iterate on the prompt; nothing here touches model loading.
"""
import math
import re

from figures import figure_prompt_block

SYSTEM_PROMPT = """You are an expert Manim Community Edition (manim >= 0.18) developer.
You turn a structured educational-video storyboard into a single, runnable Manim script.

The storyboard is the SOURCE OF TRUTH. Follow it exactly:
- TIMING: every segment has a fixed duration in seconds. The rendered animation of
  each segment must last exactly that long, and the whole video exactly the total duration.
- ON-SCREEN: the `on_screen` field describes what must be visible. Build precisely that
  with Manim objects. Do not invent extra content, extra slides or extra facts.
- NARRATION: the narration is a recorded teacher voice-over that is laid under the video
  afterwards. The beat times listed for each segment are measured from that recording, so a
  visual must appear when the teacher says it (within about 1 second), never long before or
  after. Do NOT display narration text on screen unless subtitles are explicitly requested:
  never put the narration's sentences in Text objects. The screen shows the on_screen items
  (figures, highlights, diagrams, short labels / key terms of at most about 5 words), not a
  transcript of what the teacher says.
- ORDER: segments play in the given order, one after another.

Hard technical rules:
1. Output exactly ONE ```python code block and nothing else.
2. Start with `from manim import *`. Use only Manim Community Edition APIs. No external
   files, SVGs, audio (no add_sound; the voice-over is added by the renderer) or network access. The ONLY
   images allowed are the FIGURES listed for a segment (rule 17), loaded with ImageMobject(<given path>).
3. Do NOT use Tex / MathTex / LaTeX (it may not be installed). Use `Text(...)` instead.
   Write chemical formulas with unicode subscripts, e.g. Text("C₂H₂").
4. Define exactly one Scene class named `StoryboardScene`.
5. Create one method per segment named `segment_1`, `segment_2`, ... in storyboard order.
   The class MUST define `construct` (Manim renders nothing without it) that calls every
   segment in order, exactly like this skeleton (extend for the real segment count):

       class StoryboardScene(Scene):
           def construct(self):
               self.segment_1()
               self.segment_2()
               ...

           def pad(self, start, duration): ...
           def segment_1(self): ...

6. Timing helpers: include BOTH methods in the class and use them exactly as shown.

       def pad(self, start, duration):
           remaining = duration - (self.renderer.time - start)
           if remaining > 0:
               self.wait(remaining)

       def until(self, start, sec):
           remaining = sec - (self.renderer.time - start)
           if remaining > 0:
               self.wait(remaining)

   Every segment method must begin with `t0 = self.renderer.time` and end with
   `self.pad(t0, <duration_sec>)`. The sum of run_time values and waits inside a segment
   must never exceed its duration (include the FadeOut of the previous content).
7. At the start of each segment (except the first) fade out everything from the previous
   segment, guarded so it never plays an empty animation:
   `if self.mobjects: self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.5)`.
8. Everything must stay inside the frame (width ~14, height ~8). Titles font_size 40-48,
   body text font_size 28-34, and wrap long text over several lines with `\\n` or
   `VGroup(...).arrange(DOWN)`. Keep a clean dark background and good contrast.
9. Use `Create`, `Write`, `FadeIn`, `Transform`, `LaggedStart` etc. to reveal content
   progressively in sync with the narration. Avoid static slides where something simple
   can be animated (e.g. draw the three parallel lines one after another).
10. Add a short comment at the top of each segment method: `# SEG_00X | role | duration s`.
11. The code must run without errors with: manim -ql file.py StoryboardScene
12. BEATS (timing): every segment lists narration BEATS, each with a start and end time in seconds from the start
    of the segment. Reveal visuals on those beats: before each beat call `self.until(t0, <beat start>)`, then play
    the animation(s) for that beat with a total run_time no longer than the beat. Something on screen must change at
    every beat. Keep the motion SMOOTH, like a teacher drawing while talking: give each reveal a run_time of about
    half the beat (1-3s, e.g. Write/Create/GrowArrow), and when a beat is longer than ~4s follow it with a gentle
    emphasis (Indicate, Circumscribe, a colour change) so the screen is not still while the teacher speaks.
    Every animation must finish before the segment's duration; `self.pad` only absorbs the small remainder.
15. VISUAL WINDOWS: some segments list VISUAL WINDOWS - moments where the teacher deliberately stops talking and the
    board does the teaching (intro: the scene builds before she speaks; demo: after she finishes an idea, show it
    happening; clause: a short visual follow-up; recap: highlight the key result at the end). For each window call
    `self.until(t0, <window start>)` and play ONE continuous, meaningful animation whose run_time is close to the
    window length (e.g. electron dots sliding into the shared region, the three bond lines drawn one after another
    with LaggedStart, a Transform from separate atoms to the bonded molecule, a Circumscribe/Indicate sweep over
    the key items for a recap). Never leave the screen still during a window. Never use narration-less windows to
    show new facts that are not in `on_screen`/narration.
16. LABEL SPACING: a label must never touch or cross a shape. Place labels relative to the WHOLE group they describe
    (`label.next_to(group, DOWN, buff=0.35)`), stack several labels with `VGroup(...).arrange(DOWN, buff=0.3)` and
    position that stack as one unit; element letters go exactly at the atom centre (`.move_to(atom)`).
17. FIGURES: a segment that lists FIGURES must SHOW each one: img = ImageMobject(r"<path>") with exactly the given
    path, scaled with img.set(height=...) so it is large (height 4.5-6, width at most 12), placed in the main area
    (e.g. img.move_to(DOWN * 0.3) under the title), revealed with FadeIn at the beat that introduces it, and kept
    on screen until the segment ends. Annotate it EXACTLY as on_screen says (arrows, highlight boxes, labels),
    each at the beat that mentions the part. Use the located part boxes (image pixels) through these helpers,
    which you MUST add to the class when a segment has a figure:

        def fig_point(self, img, x, y):
            h, w = img.pixel_array.shape[:2]
            return img.get_corner(UL) + RIGHT * img.width * x / w + DOWN * img.height * y / h

        def fig_box(self, img, box, **kw):
            p0, p1 = self.fig_point(img, box[0], box[1]), self.fig_point(img, box[2], box[3])
            return Rectangle(width=abs(p1[0] - p0[0]) + 0.1, height=abs(p1[1] - p0[1]) + 0.1, **kw).move_to((p0 + p1) / 2)

    Highlight: self.play(Create(self.fig_box(img, [x0, y0, x1, y1], color=YELLOW, stroke_width=6))).
    Arrow to a part: Arrow(start=<a point outside the image>, end=self.fig_point(img, cx, cy), buff=0.1, color=YELLOW).
    Labels go beside the image with a short arrow to the part, never on top of the drawn part, and must stay
    fully inside the frame (|x| <= 6.6, |y| <= 3.6): put them on the side of the image that has free space,
    e.g. tag.next_to(img, RIGHT, buff=0.3).set_y(self.fig_point(img, cx, cy)[1]), and shrink the image a little
    (height 4.5) if there is no room. A tag that must sit over the picture gets a dark backing:
    VGroup(BackgroundRectangle(tag, fill_opacity=0.85, buff=0.12), tag). The image already prints its own
    labels; do not cover them. If no part boxes are given, annotate with callouts beside the image and a
    SurroundingRectangle around it.
14. API SAFETY: use only keyword arguments that exist in Manim CE. `Line`, `Circle`, `Text`
    etc. have NO `x_shift`/`y_shift`/`offset` arguments. To make parallel lines, create each
    `Line(start, end)` and then offset it with `.shift(UP * 0.1)` / `.shift(DOWN * 0.1)`.
    Keep long `Text` within the frame with `.scale_to_fit_width(12)` if it is wider than 12.
13. CONTENT: show every text that is quoted in `on_screen` EXACTLY as written (same words; line breaks are allowed).
    You MAY add small supporting visuals that help explain the idea (labels, arrows, electron dots, highlights),
    but keep them simple and never contradict `on_screen`. Connect bonds/lines to the EDGES of atoms (use `buff`,
    or start/end at circle boundaries), not through their centres. Keep text and shapes from overlapping:
    position with `to_edge`, `next_to`, `arrange`.
"""


USER_PROMPT_TEMPLATE = """Generate the Manim script for the storyboard below.

QUERY: {query}
INTENT: {intent}
TOTAL DURATION: {total_duration_sec} seconds ({n_segments} segments)
SUBTITLES: {subtitles_instruction}

SEGMENTS (timeline is absolute and authoritative):
{segments_block}

Reminder: exactly one ```python block, class `StoryboardScene`, methods `segment_1`..`segment_{n_segments}`,
each segment padded to its exact duration with `self.pad(t0, duration)`.
"""


SEGMENT_TEMPLATE = """--- {segment_id} (method: segment_{index}) ---
role: {role}
timeline: {start:.1f}s -> {end:.1f}s   (duration_sec = {duration_sec})
concepts: {concepts}
narration ({narration_kind}): {narration}
on_screen (quoted texts must appear exactly; small extra visuals are allowed): {on_screen}
beats (call self.until(t0, start) then reveal the matching visual):
{beats}{windows}
{figures}"""


def plan_beats(seg, lead=0.6, tail=1.5, max_beat=10.0):
    """
    Split a segment's narration into reveal beats with absolute start/end times (seconds from segment start).
    Time is distributed in proportion to word counts over [lead, duration - tail]; clauses are merged when tiny
    and split when a beat would be longer than `max_beat` seconds.
    A segment that carries measured `beats` (from the TTS timeline, see checks.apply_timeline) uses those as-is.
    """
    if seg.get("beats"):
        return [{"start": float(b["start"]), "end": float(b["end"]), "text": b["text"]} for b in seg["beats"]]
    dur = float(seg["duration_sec"])
    text = (seg.get("narration") or "").strip()
    usable = max(dur - lead - tail, 1.0)
    clauses = [c.strip() for c in re.split(r"(?<=[.!?;:,])\s+", text) if c.strip()] or [""]

    merged, carry = [], ""
    for idx, c in enumerate(clauses):
        c = (carry + " " + c).strip() if carry else c
        if len(c.split()) < 4 and idx < len(clauses) - 1:
            carry = c
            continue
        merged.append(c)
        carry = ""
    if carry:
        merged[-1:] = [(merged[-1] + " " + carry).strip()] if merged else [carry]

    weights = [max(len(c.split()), 1) for c in merged]
    total, cum, raw = sum(weights), 0, []
    for c, w in zip(merged, weights):
        start = lead + usable * cum / total
        cum += w
        raw.append((start, lead + usable * cum / total, c))

    beats = []
    for start, end, c in raw:
        k = max(1, math.ceil((end - start) / max_beat))
        words = c.split()
        for j in range(k):
            chunk = " ".join(words[j * len(words) // k:(j + 1) * len(words) // k]) or c
            beats.append({"start": round(start + (end - start) * j / k, 1),
                          "end": round(start + (end - start) * (j + 1) / k, 1),
                          "text": chunk})
    return beats


def _figures_block(seg):
    """FIGURES block with the absolute image path ImageMobject must load."""
    block = figure_prompt_block(seg, "px")
    for f in seg.get("figures", []):
        if f.get("path"):
            block = block.replace(f"file {f['file']}", f"path r\"{f['path']}\"")
    return block


def build_segments_block(segments):
    """Format segments with absolute start/end times derived from duration_sec."""
    blocks, cursor = [], 0.0
    for i, seg in enumerate(sorted(segments, key=lambda s: s.get("order", 0)), start=1):
        dur = float(seg["duration_sec"])
        blocks.append(
            SEGMENT_TEMPLATE.format(
                segment_id=seg["segment_id"],
                index=i,
                role=seg.get("role", ""),
                start=cursor,
                end=cursor + dur,
                duration_sec=seg["duration_sec"],
                concepts=", ".join(seg.get("concepts", [])),
                narration=seg.get("narration", ""),
                narration_kind=("teacher voice-over, beat times measured from the recording" if seg.get("beats")
                                else "spoken, for pacing"),
                on_screen=seg.get("on_screen", ""),
                figures=_figures_block(seg),
                beats="\n".join(f"  t={b['start']:.1f}s-{b['end']:.1f}s: {b['text']}" for b in plan_beats(seg)),
                windows=("\nVISUAL WINDOWS (teacher silent; one continuous animation spanning each window):\n"
                         + "\n".join(f"  t={w['start']:.1f}s-{w['end']:.1f}s: {w['kind']}"
                                      + (f" after \"{plan_beats(seg)[w['after_beat'] - 1]['text'][:60]}\""
                                         if 0 < w['after_beat'] <= len(plan_beats(seg)) else "")
                                      for w in seg["windows"])) if seg.get("windows") else "",
            )
        )
        cursor += dur
    return "\n".join(blocks), cursor


def build_messages(storyboard: dict, subtitles: bool = False):
    """Return chat messages (system + user) for a storyboard dict."""
    segments_block, computed_total = build_segments_block(storyboard["segments"])
    total = storyboard.get("total_duration_sec", computed_total)

    if subtitles:
        sub = ("ON - show each segment's narration as a small subtitle (font_size 24) at the "
               "bottom of the frame, replaced at each segment change.")
    else:
        sub = "OFF - do not display narration text."

    user = USER_PROMPT_TEMPLATE.format(
        query=storyboard.get("query", ""),
        intent=storyboard.get("intent", ""),
        total_duration_sec=total,
        n_segments=len(storyboard["segments"]),
        subtitles_instruction=sub,
        segments_block=segments_block,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# =====================================================================
# Verifier + feedback-loop prompts (used by pipeline.py)
# =====================================================================

VERIFIER_SYSTEM_PROMPT = """You are a reviewer of Manim Community Edition (>= 0.18) scripts generated from a storyboard.
You do NOT rewrite code. You report problems as JSON.

Be lenient. The storyboard's `on_screen` text is only a rough description, so reasonable extra visuals (titles,
labels, arrows, electron dots, highlights, small icons) are FINE. NEVER report extra content and NEVER report that
wording differs slightly. Wording and syntax were already checked by tools.

You check ONLY these two things:
1. ARRANGEMENT (category "layout"): elements overlapping badly, text colliding with shapes, content crammed into one
   area or far off-centre, bond lines that pass through atom centres instead of connecting atom edges, a label
   placed far from the object it describes, text too small to read (font_size below 20).
2. SYNC (category "sync"): each segment lists narration BEATS with start times. Visuals should be revealed in the
   order of the beats, each at about its start time (via self.until(t0, start)). Report a beat that has no visual
   change, visuals revealed in the wrong order, or everything dumped at the start. If VISUAL WINDOWS are listed,
   report a window that is not filled by one continuous animation, or one that shows content unrelated to the
   segment.

If the AUTOMATIC CHECK REPORT says the scene crashed, you may also report the obvious cause with category "api".
When the scene executed successfully, syntax, API and timing arithmetic were settled by really running it: do NOT
discuss them. NEVER claim that a Manim class, function or argument does not exist or is invalid (Text, Dot, Circle,
Arrow, Line, VGroup, SurroundingRectangle, Write, FadeIn, font_size, color, buff all exist). Do not suggest Tex/MathTex.

Output format - JSON only, no markdown fences, no commentary:
{"approved": true|false,
 "issues": [{"segment": <int, 0 for global>, "severity": "major"|"minor",
             "category": "layout"|"sync"|"api",
             "problem": "<one sentence, point to the code>", "fix": "<one concrete instruction>"}]}

Rules: at most 6 issues, most severe first. Only report problems you can point to in the code. Never invent problems.
"major" = clearly visible defect; "minor" = polish. "approved" is true when there are no major issues.
"""


VERIFIER_USER_TEMPLATE = """STORYBOARD: {query}
TOTAL DURATION: {total_duration_sec} seconds

SEGMENTS (authoritative):
{segments_block}

AUTOMATIC CHECK REPORT:
{tool_report}

SCRIPT UNDER REVIEW:
```python
{code}
```

Return the JSON review now."""


FIX_USER_TEMPLATE = """A reviewer and an automatic runner found problems in your script. They are listed below.

ISSUES TO FIX:
{issues}

Instructions:
- Fix every CRITICAL and MAJOR issue.
- Keep everything that already works unchanged (same class, same method names, same timing helper).
- Obey all the original rules (no Tex/MathTex, only documented Manim CE arguments, each segment ends with self.pad).
- Return the COMPLETE corrected script in exactly ONE ```python code block and nothing else."""


def build_verifier_messages(storyboard: dict, code: str, tool_report: str):
    segments_block, computed_total = build_segments_block(storyboard["segments"])
    user = VERIFIER_USER_TEMPLATE.format(
        query=storyboard.get("query", ""),
        total_duration_sec=storyboard.get("total_duration_sec", computed_total),
        segments_block=segments_block,
        tool_report=tool_report,
        code=code,
    )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_fix_messages(generator_messages, previous_code: str, issues_text: str):
    """Original prompt + the previous attempt + feedback. History stays bounded: only the last attempt is kept."""
    return list(generator_messages) + [
        {"role": "assistant", "content": f"```python\n{previous_code}\n```"},
        {"role": "user", "content": FIX_USER_TEMPLATE.format(issues=issues_text)},
    ]