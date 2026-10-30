"""
Render a generated scene to a narrated MP4.

    python render.py outputs/<id>/pipeline__.../final_scene.py --narration outputs/<id>/narration -q m

The scene is rendered through a small harness that subclasses `StoryboardScene` and, at the start of every
`segment_N`, adds that segment's voice-over track (`audio/seg_N.wav` from tts.build_narration) at the renderer's
current time. Audio is therefore anchored to where each segment really starts, so timing drift in one segment
cannot shift the narration of the next. The narration SRT is muxed in as a soft subtitle track.
Outputs in out_dir: final.mp4, render.log, render_report.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUALITY_DIRS = {"l": "480p15", "m": "720p30", "h": "1080p60", "k": "2160p60"}

HARNESS = '''import importlib.util
import sys
from manim import *
sys.path.insert(0, {here!r})
import layout_fix

_spec = importlib.util.spec_from_file_location("generated_scene", {scene!r})
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AUDIO = {audio!r}
# Time-budget guard: a segment that overran its duration in the headless check is played faster by this factor
# (every play() run time and plain wait(); until() targets are scaled too, pad() is not), so each segment still ends
# exactly at its budget and stays aligned with its voice-over.
SCALE = {scale!r}
# A segment that returns before its budget (e.g. the generator left out its final pad) is held on its last frame
# until the budget, so the video track never ends before the voice-over and later segments start on time.
BUDGET = {budget!r}


def _narrated(i, orig):
    def seg(self, *a, **k):
        self._vx_scale = SCALE.get(i, 1.0)
        self.add_sound(AUDIO[i]) if i in AUDIO else None
        t0 = self.renderer.time
        try:
            out = orig(self, *a, **k)
        finally:
            self._vx_scale = 1.0
        short = BUDGET.get(i, 0.0) - (self.renderer.time - t0)
        if short > 1 / 30:
            self.wait(short)
        return out
    return seg


class NarratedScene(_mod.StoryboardScene):
    _vx_scale = 1.0

    def play(self, *anims, **kw):
        if anims and all(type(a).__name__ == "Wait" for a in anims):
            return super().play(*anims, **kw)   # wait() is play(Wait(...)): already scaled once in wait()
        # the same run-time label placement exec_check.py measured the scene with
        anims = anims + tuple(layout_fix.place_labels(self, anims, float(config.frame_width), float(config.frame_height)))
        f = self._vx_scale
        if f < 0.999:
            rt = kw.get("run_time")
            if rt is None:
                rts = [float(getattr(a, "run_time", 1.0)) for a in anims if hasattr(a, "run_time")]
                rt = max(rts) if rts else 1.0
            kw["run_time"] = max(float(rt) * f, 1 / 30)
        return super().play(*anims, **kw)

    def wait(self, duration=1.0, *a, **kw):
        return super().wait(duration * self._vx_scale if self._vx_scale < 0.999 else duration, *a, **kw)

    def until(self, start, sec):
        f, self._vx_scale = self._vx_scale, 1.0
        try:
            return super().until(start, sec * f)
        finally:
            self._vx_scale = f

    def pad(self, start, duration):
        f, self._vx_scale = self._vx_scale, 1.0
        try:
            return super().pad(start, duration)
        finally:
            self._vx_scale = f


for _i in sorted(set(AUDIO) | set(SCALE) | set(BUDGET)):
    setattr(NarratedScene, "segment_%d" % _i, _narrated(_i, getattr(_mod.StoryboardScene, "segment_%d" % _i)))
'''


def probe_duration(path, stream="format"):
    """Duration in seconds of the container ("format") or of the first audio stream ("a")."""
    sel = [] if stream == "format" else ["-select_streams", stream]
    entry = "format=duration" if stream == "format" else "stream=duration"
    out = subprocess.run(["ffprobe", "-v", "error", *sel, "-show_entries", entry, "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip().splitlines()
    try:
        return float(out[0])
    except (IndexError, ValueError):
        return None


def render_video(scene_path, out_dir, narration_dir=None, quality="m", timeout=3600, time_scale=None):
    """Render scene_path (with narration when narration_dir has a timeline.json). Returns a report dict."""
    scene_path, out_dir = Path(scene_path).resolve(), Path(out_dir).resolve()
    work = out_dir / "render"
    work.mkdir(parents=True, exist_ok=True)
    report = {"scene": str(scene_path), "quality": quality, "narrated": False}

    timeline, audio, budget = None, {}, {}
    if narration_dir and (Path(narration_dir) / "timeline.json").exists():
        narration_dir = Path(narration_dir).resolve()
        timeline = json.loads((narration_dir / "timeline.json").read_text(encoding="utf-8"))
        audio = {s["index"]: str(narration_dir / s["audio"]) for s in timeline["segments"]}
        budget = {s["index"]: float(s["duration_sec"]) for s in timeline["segments"] if s.get("duration_sec")}
        report["narrated"] = True

    harness = work / "render_scene.py"
    harness.write_text(HARNESS.format(scene=str(scene_path), here=str(Path(__file__).resolve().parent), audio=audio, scale=dict(time_scale or {}),
                                       budget=budget),
                       encoding="utf-8")
    if time_scale:
        report["time_scale"] = {str(k): round(v, 3) for k, v in time_scale.items()}
    t = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "manim", "render", f"-q{quality}", "--disable_caching", "--media_dir",
             str(work / "media"), "-o", "scene", str(harness), "NarratedScene"],
            capture_output=True, text=True, cwd=work, timeout=timeout)
        log, ok = proc.stdout + "\n" + proc.stderr, proc.returncode == 0
    except subprocess.TimeoutExpired:
        log, ok = f"render timed out after {timeout}s", False
    (out_dir / "render.log").write_text(log, encoding="utf-8")
    report["render_sec"] = round(time.time() - t, 1)
    video = work / "media" / "videos" / "render_scene" / QUALITY_DIRS[quality] / "scene.mp4"
    if not ok or not video.exists():
        report.update(ok=False, error=log.strip()[-1500:])
        _write(out_dir, report)
        return report

    final = out_dir / "final.mp4"
    srt = Path(narration_dir) / "narration.srt" if timeline else None
    if srt is not None and srt.exists():
        mux = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(srt), "-map", "0",
                              "-map", "1", "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                              str(final)], capture_output=True, text=True)
        if mux.returncode != 0:
            report["subtitle_mux_error"] = mux.stderr[-500:]
            shutil.copyfile(video, final)
    else:
        shutil.copyfile(video, final)

    report.update(ok=True, video=str(final), video_sec=probe_duration(final), audio_sec=probe_duration(final, "a"),
                  video_track_sec=probe_duration(final, "v"))
    if timeline:
        report["narration_sec"] = timeline["total_sec"]
        report["audio_ok"] = report["audio_sec"] is not None and abs(report["audio_sec"] - timeline["total_sec"]) < 1.5
        # the container lasts as long as its longest track: check the picture itself reaches the narration's end
        report["video_ok"] = (report["video_track_sec"] is not None
                              and abs(report["video_track_sec"] - timeline["total_sec"]) < 1.5)
    _write(out_dir, report)
    return report


def _write(out_dir, report):
    (out_dir / "render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--narration", default=None, help="directory with timeline.json + audio/ from tts.py")
    ap.add_argument("-o", "--out-dir", default=None, help="default: the scene's directory")
    ap.add_argument("-q", "--quality", choices=sorted(QUALITY_DIRS), default="m")
    args = ap.parse_args()
    out = args.out_dir or Path(args.scene).parent
    print(json.dumps(render_video(args.scene, out, args.narration, args.quality), indent=2))


if __name__ == "__main__":
    main()
