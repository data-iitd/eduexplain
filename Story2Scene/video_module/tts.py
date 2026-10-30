"""
Teacher-voice narration for the storyboard pipeline.

Narration is synthesized first (Kokoro-82M via kokoro-onnx, CPU), measured, and
laid out on the storyboard's per-segment time slots. The resulting timeline
(sentence start/end per segment) is what drives when visuals appear.

stdlib + numpy only at import time; kokoro_onnx is imported lazily by KokoroTTS.

CLI: python tts.py -s storyboards/storyboard.json -o outputs/<id>/narration [--fake]
"""
import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np

SR_FAKE = 24000

# ------------------------------------------------------------ text handling
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_FORMULA = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?\d*)+)(?![A-Za-z0-9])")
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def _num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    return " ".join(_ONES[int(d)] for d in str(n))


def _speak_formula(m) -> str:
    tok = m.group(1)
    if not re.search(r"\d", tok):
        return tok
    parts = []
    for el, num in _ELEMENT.findall(tok):
        parts.append(" ".join(el) if len(el) == 1 else el)  # "C" -> "C", "Cl" kept
        if num:
            parts.append(_num_words(int(num)))
    return " ".join(parts)


def normalize_for_speech(text: str) -> str:
    """Make text speakable: chemical formulas spelled out, markup/quotes dropped."""
    text = text.translate(_SUB)
    text = _FORMULA.sub(_speak_formula, text)
    text = re.sub(r"[\"“”‘’`*_#~^|\\<>{}\[\]]", "", text)
    text = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", "", text)  # keep apostrophes in words
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_beats(narration: str, max_words: int = 16) -> list:
    """Sentence split; over-long sentences are split (recursively) at the comma nearest the middle.
    Short beats give the teacher more natural pauses, each paired with its own visual reveal."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", narration.strip()) if s.strip()]

    def split(s):
        words = s.split()
        commas = [i + 1 for i, w in enumerate(words[:-1]) if w.endswith(",") and 3 <= i + 1 <= len(words) - 3]
        if len(words) <= max_words or not commas:
            return [s]
        cut = min(commas, key=lambda i: abs(i - len(words) / 2))
        return split(" ".join(words[:cut])) + split(" ".join(words[cut:]))

    return [b for s in sentences for b in split(s)]


# ------------------------------------------------------------------ engines
class KokoroTTS:
    name = "kokoro-82m"

    def __init__(self, model_dir="models/kokoro", voice="af_heart", speed=0.9, lang="en-us"):
        from kokoro_onnx import Kokoro  # lazy: tests run without it
        d = Path(model_dir)
        self._k = Kokoro(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
        self.voice, self.speed, self.lang = voice, speed, lang

    def synthesize(self, text, speed=None):
        audio, sr = self._k.create(text, voice=self.voice, speed=speed or self.speed, lang=self.lang)
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


class FakeTTS:
    """No dependencies: silence of 0.4 s per word (scaled by speed) at 24 kHz."""
    name = "fake"

    def __init__(self, voice="fake", speed=0.9, **_):
        self.voice, self.speed = voice, speed

    def synthesize(self, text, speed=None):
        n = len(text.split())
        secs = 0.4 * n * (self.speed / (speed or self.speed))
        return np.zeros(int(secs * SR_FAKE), dtype=np.float32), SR_FAKE


# ----------------------------------------------------------------- timeline
def _trim(audio, sr, thresh=1e-3, keep=0.04):
    idx = np.flatnonzero(np.abs(audio) > thresh)
    if idx.size == 0:
        return audio
    k = int(keep * sr)
    return audio[max(0, idx[0] - k): idx[-1] + 1 + k]


def _normalize(audio, target_rms=0.08, peak=0.95):
    """Even loudness across sentences (Parler's level varies per generation); silence is left alone."""
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 1e-4:
        return audio
    out = audio * (target_rms / rms)
    m = float(np.max(np.abs(out)))
    return out * (peak / m) if m > peak else out


def _synth_all(tts, spoken, speed):
    clips, sr = [], None
    for t in spoken:
        a, sr = tts.synthesize(t, speed=speed)
        clips.append(_normalize(_trim(a, sr)))
    return clips, sr


def _write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _cache_key(sb_segments, tts, *params):
    payload = [tts.name, tts.voice, tts.speed, *params,
               [(s.get("narration", ""), s.get("duration_sec")) for s in sb_segments]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_timeline(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Natural teaching rhythm: short breath at a comma, a beat at a full stop, a short settle around slide changes.
NATURAL = {"lead": 0.6, "tail": 1.2, "clause_gap": 0.45, "sentence_gap": 0.8}


def _natural_gaps(beats, clause_gap, sentence_gap):
    """Pause after each beat except the last: longer when the beat ends a sentence."""
    return [sentence_gap if re.search(r"[.!?;:]\W*$", b) else clause_gap for b in beats[:-1]]


# Budget pacing: where a teacher naturally stops talking while the board/animation does the work.
#   weight = share of the segment's spare time, cap = longest silence (s) before caps are relaxed.
WINDOWS = {"intro": (1.0, 3.0), "demo": (2.0, 6.0), "clause": (0.6, 2.5), "recap": (1.5, 5.0)}
MIN_PAUSES = {"lead": 0.3, "tail": 0.5, "clause_gap": 0.25, "sentence_gap": 0.4}
WINDOW_MIN_SEC = 1.0   # spare time below this is just a natural pause, not a "visual window"


def _fill(slack, slots):
    """Water-fill `slack` seconds into slots [(weight, cap)] proportionally to weight, respecting caps;
    if every slot is capped the rest is spread by weight again (the budget always wins)."""
    alloc = [0.0] * len(slots)
    free = [k for k, (w, _) in enumerate(slots) if w > 0]
    left = slack
    while left > 1e-6 and free:
        wsum = sum(slots[k][0] for k in free)
        capped = []
        for k in free:
            share = left * slots[k][0] / wsum
            room = slots[k][1] - alloc[k]
            if share >= room:
                alloc[k] += room
                capped.append(k)
        if not capped:
            for k in free:
                alloc[k] += left * slots[k][0] / wsum
            left = 0.0
            break
        left = slack - sum(alloc)
        free = [k for k in free if k not in capped]
    if left > 1e-6:   # all capped: ignore caps
        wsum = sum(w for w, _ in slots) or 1.0
        alloc = [a + left * w / wsum for a, (w, _) in zip(alloc, slots)]
    return alloc


def _layout_budget(beats, clip_secs, D, lead, tail, clause_gap, sentence_gap):
    """
    Place beats inside a segment of exactly D seconds. Returns (starts, windows, need) where need > D means the
    speech does not fit even with minimum pauses (caller speeds up or borrows time).
    Pause after beat k: natural gap + its share of the spare time (a 'demo' window after a sentence, a short
    'clause' window after a comma); spare time also goes to an 'intro' before the first beat and a 'recap' after
    the last one, so silence lands where a teacher would let the animation speak.
    """
    n = len(beats)
    ends_sentence = [bool(re.search(r"[.!?;:]\W*$", b)) for b in beats]
    gaps = [sentence_gap if ends_sentence[k] else clause_gap for k in range(n - 1)]
    speech = sum(clip_secs)
    need = lead + speech + sum(gaps) + tail
    if need > D:   # squeeze pauses towards their minimums
        mins = (MIN_PAUSES["lead"] + MIN_PAUSES["tail"]
                + sum(MIN_PAUSES["sentence_gap"] if ends_sentence[k] else MIN_PAUSES["clause_gap"] for k in range(n - 1)))
        flex = (lead + tail + sum(gaps)) - mins
        over = need - D
        f = 1.0 - min(1.0, over / flex) if flex > 0 else 0.0
        sq = lambda v, m: m + (v - m) * f
        lead, tail = sq(lead, MIN_PAUSES["lead"]), sq(tail, MIN_PAUSES["tail"])
        gaps = [sq(g, MIN_PAUSES["sentence_gap"] if ends_sentence[k] else MIN_PAUSES["clause_gap"])
                for k, g in enumerate(gaps)]
        need = lead + speech + sum(gaps) + tail
        extra = [0.0] * (n + 1)
    else:
        slots = [WINDOWS["intro"]] + [WINDOWS["demo"] if ends_sentence[k] else WINDOWS["clause"]
                                      for k in range(n - 1)] + [WINDOWS["recap"]]
        extra = _fill(D - need, slots)
    starts, windows = [], []
    t = lead + extra[0]
    if extra[0] >= WINDOW_MIN_SEC:
        windows.append({"start": round(lead, 2), "end": round(t, 2), "kind": "intro", "after_beat": 0})
    for k in range(n):
        starts.append(t)
        t += clip_secs[k]
        if k < n - 1:
            w0 = t + gaps[k] / 2
            t += gaps[k] + extra[k + 1]
            if extra[k + 1] >= WINDOW_MIN_SEC:
                windows.append({"start": round(w0, 2), "end": round(t - gaps[k] / 2, 2),
                                "kind": "demo" if ends_sentence[k] else "clause", "after_beat": k + 1})
    if extra[-1] >= WINDOW_MIN_SEC:
        windows.append({"start": round(t + 0.3, 2), "end": round(min(t + 0.3 + extra[-1], D - 0.2), 2),
                        "kind": "recap", "after_beat": n})
    return starts, windows, need


def _rebalance(needs, durations):
    """Segments whose speech cannot fit borrow time from the segments with the most spare time, so that the
    total stays exactly sum(durations). Returns the new per-segment durations."""
    d = list(durations)
    for i, need in enumerate(needs):
        over = need - d[i]
        while over > 1e-3:
            donors = sorted((d[j] - needs[j], -abs(j - i), j) for j in range(len(d)) if j != i and d[j] - needs[j] > 0.5)
            if not donors:
                break
            spare, _, j = donors[-1]
            take = min(over, spare - 0.5)
            d[j] -= take
            d[i] += take
            over -= take
    return [round(x, 2) for x in d]


PACINGS = ("cap", "budget", "natural", "storyboard")


def build_narration(storyboard, out_dir, tts, pacing="cap", lead=None, tail=None, max_gap=6.0,
                    min_gap=0.35, max_speed_up=1.15, clause_gap=None, sentence_gap=None) -> dict:
    """
    pacing="cap" (default): the storyboard duration_sec is an upper bound, not a target. Each segment lasts as long
      as its speech with natural pauses (no stretched silences), never longer than duration_sec; speech that does
      not fit is squeezed / sped up exactly as in "budget" and may borrow from segments that finish early, so the
      total is always <= the storyboard total. No visual windows.
    pacing="budget": every segment keeps the storyboard's duration_sec, so the video lasts exactly the
      storyboard total. The teacher speaks with natural pauses; the spare time becomes visual windows (intro,
      demo after a sentence, recap) in which the generated animation keeps moving. Speech that does not fit is
      sped up (<= max_speed_up) and, as a last resort, borrows time from the segment with the most spare time
      (total unchanged).
    pacing="natural": segments last as long as the speech + short natural pauses (video shorter than storyboard).
    pacing="storyboard": legacy - keep duration_sec, spread spare time evenly between sentences (up to max_gap).
    """
    from checks import ordered_segments
    if pacing not in PACINGS:
        raise ValueError(f"unknown pacing {pacing!r}")
    lead = (0.8 if pacing == "storyboard" else NATURAL["lead"]) if lead is None else lead
    tail = (1.5 if pacing == "storyboard" else NATURAL["tail"]) if tail is None else tail
    clause_gap = NATURAL["clause_gap"] if clause_gap is None else clause_gap
    sentence_gap = NATURAL["sentence_gap"] if sentence_gap is None else sentence_gap
    segs = ordered_segments(storyboard)
    out = Path(out_dir)
    key = _cache_key(segs, tts, pacing, lead, tail, max_gap, min_gap, max_speed_up, clause_gap, sentence_gap,
                     WINDOWS, MIN_PAUSES)
    tl_path = out / "timeline.json"
    if tl_path.exists():
        old = load_timeline(tl_path)
        if old.get("cache_key") == key and (out / "narration.wav").exists() and all(
                (out / s["audio"]).exists() for s in old["segments"]):
            return old
    (out / "audio").mkdir(parents=True, exist_ok=True)

    # pass 1: synthesize; in budget mode speed up any segment whose speech cannot fit its slot
    plans, sr = [], None
    for seg in segs:
        D = float(seg["duration_sec"])
        beats = split_beats(seg.get("narration", ""))
        spoken = [normalize_for_speech(b) for b in beats]
        speed = tts.speed
        clips, sr = _synth_all(tts, spoken, speed)
        secs = [len(c) / sr for c in clips]
        if pacing in ("cap", "budget"):
            need = _layout_budget(beats, secs, D, lead, tail, clause_gap, sentence_gap)[2]
            for _ in range(3):   # speech length is not exactly proportional to 1/speed: iterate
                if need <= D or speed >= tts.speed * max_speed_up - 1e-3:
                    break
                speech = sum(secs)
                speed = min(speed * speech / max(speech - (need - D) - 0.05, 1e-3), tts.speed * max_speed_up)
                clips, sr = _synth_all(tts, spoken, speed)
                secs = [len(c) / sr for c in clips]
                need = _layout_budget(beats, secs, D, lead, tail, clause_gap, sentence_gap)[2]
        else:
            need = None
        plans.append({"seg": seg, "D": D, "beats": beats, "spoken": spoken, "clips": clips, "secs": secs,
                      "speed": speed, "need": need})

    if pacing in ("cap", "budget"):
        for p, d in zip(plans, _rebalance([p["need"] for p in plans], [p["D"] for p in plans])):
            p["dur"] = min(d, round(p["need"], 2)) if pacing == "cap" else d   # cap: no stretched pauses

    # pass 2: lay out every segment and write the tracks
    tracks, entries, t_abs = [], [], 0.0
    for i, p in enumerate(plans, 1):
        beats, clips, secs, D, n = p["beats"], p["clips"], p["secs"], p["D"], len(p["beats"])
        speech, windows, extended = sum(secs), [], False
        if pacing in ("cap", "budget"):
            dur = p["dur"]
            need = _layout_budget(beats, secs, dur, lead, tail, clause_gap, sentence_gap)[2]
            for _ in range(3):   # last resort before the voice-over would be cut: speak faster (<= 1.3x)
                if need <= dur + 0.05 or p["speed"] >= tts.speed * 1.3 - 1e-3:
                    break
                speech = sum(secs)
                p["speed"] = min(p["speed"] * speech / max(speech - (need - dur) - 0.05, 1e-3), tts.speed * 1.3)
                clips, sr = _synth_all(tts, p["spoken"], p["speed"])
                secs = [len(c) / sr for c in clips]
                speech = sum(secs)
                need = _layout_budget(beats, secs, dur, lead, tail, clause_gap, sentence_gap)[2]
            if need > dur + 0.05:
                print(f"  ! narration of segment {i} needs {need:.1f}s but only {dur:.1f}s fit the time budget even "
                      f"at {p['speed']:.2f}x: its voice-over will be cut. Shorten the storyboard narration.")
            starts, windows, _ = _layout_budget(beats, secs, dur, lead, tail, clause_gap, sentence_gap)
        else:
            if pacing == "natural":
                gaps = _natural_gaps(beats, clause_gap, sentence_gap)
                dur = lead + speech + sum(gaps) + tail
            else:
                usable, need_gaps = D - lead - tail, min_gap * (n - 1)
                if speech + need_gaps > usable:
                    target = p["speed"] * speech / max(usable - need_gaps, 1e-3)
                    p["speed"] = min(target, tts.speed * max_speed_up)
                    clips, sr = _synth_all(tts, p["spoken"], p["speed"])
                    secs = [len(c) / sr for c in clips]
                    speech = sum(secs)
                extended = speech + need_gaps > usable
                if extended:
                    dur, gap = lead + speech + need_gaps + tail, min_gap
                else:
                    dur, gap = D, (min(max_gap, max(min_gap, (usable - speech) / n)) if n > 1 else 0.0)
                gaps = [gap] * (n - 1)
            starts, t = [], lead
            for k in range(n):
                starts.append(t)
                t += secs[k] + (gaps[k] if k < n - 1 else 0.0)
        dur = round(dur, 2)

        track = np.zeros(int(round(dur * sr)), dtype=np.float32)
        tl_beats = []
        for b, sp, c, st in zip(beats, p["spoken"], clips, starts):
            s0 = int(round(st * sr))
            track[s0:s0 + len(c)] = c[:max(0, len(track) - s0)]
            tl_beats.append({"start": round(st, 2), "end": round(st + len(c) / sr, 2), "text": b, "spoken": sp})
        _write_wav(out / "audio" / f"seg_{i}.wav", track, sr)
        tracks.append(track)
        entries.append({"index": i, "segment_id": p["seg"].get("segment_id"),
                        "storyboard_duration_sec": D, "duration_sec": dur,
                        "start_sec": round(t_abs, 2), "speech_sec": round(speech, 2),
                        "speed": round(p["speed"], 3), "extended": extended or (pacing in ("cap", "budget") and dur > D + 0.01),
                        "audio": f"audio/seg_{i}.wav", "beats": tl_beats, "windows": windows})
        t_abs += len(track) / sr

    _write_wav(out / "narration.wav", np.concatenate(tracks), sr)
    cues = []
    for e in entries:
        for b in e["beats"]:
            cues.append((e["start_sec"] + b["start"], e["start_sec"] + b["end"], b["text"]))
    (out / "narration.srt").write_text("\n".join(
        f"{k}\n{_srt_time(a)} --> {_srt_time(b)}\n{txt}\n" for k, (a, b, txt) in enumerate(cues, 1)),
        encoding="utf-8")
    tl = {"engine": tts.name, "voice": tts.voice, "speed": tts.speed, "pacing": pacing, "sample_rate": sr,
          "total_sec": round(sum(len(x) for x in tracks) / sr, 2),
          "storyboard_total_sec": round(sum(p["D"] for p in plans), 2), "cache_key": key,
          "segments": entries}
    tl_path.write_text(json.dumps(tl, indent=2, ensure_ascii=False), encoding="utf-8")
    return tl


def main():
    from checks import load_storyboard
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-s", "--storyboard", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--engine", choices=["parler", "kokoro"], default="parler")
    ap.add_argument("--voice", default=None, help="Parler speaker name or Kokoro voice id")
    ap.add_argument("--speed", type=float, default=0.92)
    ap.add_argument("--pacing", choices=PACINGS, default="cap",
                    help="cap: natural speech, each segment <= its storyboard duration (default); "
                         "budget: exact storyboard durations with visual windows; natural: segments last as long "
                         "as the teacher talks; storyboard: legacy even pauses")
    ap.add_argument("--model-dir", default="models/kokoro")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    sb = load_storyboard(a.storyboard, a.index)
    if a.fake:
        tts = FakeTTS(voice=a.voice or "fake", speed=a.speed)
    elif a.engine == "parler":
        from tts_parler import IndicParlerTTS
        tts = IndicParlerTTS(speed=a.speed, **({"voice": a.voice} if a.voice else {}))
    else:
        tts = KokoroTTS(a.model_dir, a.voice or "af_heart", a.speed)
    try:
        tl = build_narration(sb, a.out, tts, pacing=a.pacing)
    finally:
        getattr(tts, "close", lambda: None)()
    for s in tl["segments"]:
        print(f"seg {s['index']}: speech {s['speech_sec']:.1f}s / {s['duration_sec']:.1f}s "
              f"(storyboard {s['storyboard_duration_sec']:.0f}s), {len(s['beats'])} beats, "
              f"speed {s['speed']}{', EXTENDED' if s['extended'] else ''}; windows: "
              + ", ".join(f"{w['kind']} {w['start']:.1f}-{w['end']:.1f}" for w in s.get("windows", [])))
    print(f"total {tl['total_sec']}s -> {a.out}")


if __name__ == "__main__":
    main()
