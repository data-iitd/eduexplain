"""
Headless execution check for a generated Manim scene. NO video is written.

Usage (called by checks.run_exec_check, or manually):
    python exec_check.py scene.py report.json

It executes the whole scene at a tiny resolution (so it is fast), wraps each
`segment_N` method and measures:
  * runtime errors (with the offending line number inside the generated file)
  * real duration of every segment and of the whole scene
  * longest frozen stretch per segment (consecutive wait time with no animation)
  * how many objects are on screen at the end of each segment
  * start time of every animation, relative to its segment (narration-sync check)
  * Text objects and shapes that fall outside the frame, overlapping Text objects, and labels colliding with shapes
  * [start, end] of every animation (how much of a silent visual window is actually animated)
Before every play() the labels about to appear are moved to free, in-frame spots (layout_fix.place_labels, the same
pass render.py's harness runs), and the moves are recorded per segment as labels_moved; --no-layout-fix turns it off.
Requires `manim` to be installed in the interpreter that runs this file.
"""
import importlib.util
import json
import linecache
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout_fix   # noqa: E402  (shared collision geometry + run-time label placement)


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def set_cfg(config, name, value):
    try:
        setattr(config, name, value)
    except Exception:
        try:
            config[name] = value
        except Exception:
            pass


def inspect_frame(scene, config, Text, MarkupText):
    """Frame-bounds and text-overlap checks on whatever is on screen right now."""
    fw, fh = float(config.frame_width), float(config.frame_height)
    boxes, out_of_frame = [], []
    for top in scene.mobjects:
        for sub in top.get_family():
            if not isinstance(sub, (Text, MarkupText)):
                continue
            try:
                l, r = float(sub.get_left()[0]), float(sub.get_right()[0])
                t, b = float(sub.get_top()[1]), float(sub.get_bottom()[1])
            except Exception:
                continue
            label = str(getattr(sub, "text", ""))[:40].replace("\n", " ")
            if l < -fw / 2 - 0.05 or r > fw / 2 + 0.05 or t > fh / 2 + 0.05 or b < -fh / 2 - 0.05:
                out_of_frame.append(label)
            boxes.append((label, l, r, b, t))

    # non-text shapes (atoms, bond lines, arrows) that stick out of the frame; glyphs of Text are skipped
    text_parts = {id(m) for top in scene.mobjects for sub in top.get_family()
                  if isinstance(sub, (Text, MarkupText)) for m in sub.get_family()}
    shapes_out = []
    for top in scene.mobjects:
        for sub in top.get_family():
            if id(sub) in text_parts or type(sub).__name__ == "ImageMobject" or not getattr(sub, "has_points", lambda: False)():
                continue
            try:
                l, r = float(sub.get_left()[0]), float(sub.get_right()[0])
                t, b = float(sub.get_top()[1]), float(sub.get_bottom()[1])
                cx, cy = float(sub.get_center()[0]), float(sub.get_center()[1])
            except Exception:
                continue
            if l < -fw / 2 - 0.05 or r > fw / 2 + 0.05 or t > fh / 2 + 0.05 or b < -fh / 2 - 0.05:
                shapes_out.append(f"{type(sub).__name__} at ({cx:.1f}, {cy:.1f})")

    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if layout_fix.texts_overlap(boxes[i][1:], boxes[j][1:]):
                overlaps.append([boxes[i][0], boxes[j][0]])
    # labels colliding with shapes (atoms, bonds, arrows). Containers that fully hold the text (boxes, highlight
    # rectangles, atom circles with the element letter inside) are fine; partial overlap is a collision.
    text_on_shape = []
    for top in scene.mobjects:
        for sub in top.get_family():
            if id(sub) in text_parts or type(sub).__name__ == "ImageMobject" or not getattr(sub, "has_points", lambda: False)():
                continue
            try:
                sbox = layout_fix.shape_box(sub)   # bond lines get their visible stroke
            except Exception:
                continue
            for label, l, r, b, t in boxes:
                # containers that fully hold the label are fine; touching within the clearance is a collision
                if layout_fix.text_hits_shape((l, r, b, t), sbox):
                    item = [label, type(sub).__name__]
                    if item not in text_on_shape:
                        text_on_shape.append(item)
    images = []
    for top in scene.mobjects:
        for sub in top.get_family():
            if type(sub).__name__ == "ImageMobject":
                try:
                    images.append({"height_frac": round(float(sub.height) / fh, 3),
                                   "width_frac": round(float(sub.width) / fw, 3),
                                   "opacity": round(float(getattr(sub, "fill_opacity", 1) or 1), 2)})
                except Exception:
                    pass
    return {"out_of_frame": out_of_frame[:5], "overlaps": overlaps[:5], "shapes_out_of_frame": shapes_out[:5],
            "text_on_shape": text_on_shape[:5], "images": images}


def main(scene_path, report_path, fix_layout=True):
    report = {"ok": False}
    try:
        from manim import config, Scene, Text, MarkupText
    except Exception as e:  # manim not installed in this interpreter
        report["env_error"] = f"{type(e).__name__}: {e}"
        write_report(report_path, report)
        return

    tmp = tempfile.mkdtemp(prefix="manim_check_")
    # Order matters: pixel size first, then frame rate.
    for name, val in [("media_dir", tmp), ("write_to_movie", False), ("save_last_frame", False),
                      ("disable_caching", True), ("progress_bar", "none"), ("verbosity", "ERROR"),
                      ("pixel_width", 256), ("pixel_height", 144), ("frame_rate", 10), ("dry_run", True)]:
        set_cfg(config, name, val)

    state = {"current": 0, "idle": {}, "seg_start": 0.0, "plays": {}, "spans": {}, "nudges": {}}
    records = []
    scene = None
    try:
        spec = importlib.util.spec_from_file_location("generated_scene", scene_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cls = getattr(module, "StoryboardScene", None)
        if cls is None:
            cands = [v for v in vars(module).values()
                     if isinstance(v, type) and issubclass(v, Scene) and v is not Scene and v.__module__ == module.__name__]
            if not cands:
                raise RuntimeError("No Scene subclass found in generated file")
            cls = cands[0]
        scene = cls()

        # ---- wrap pad() to measure idle time
        if hasattr(scene, "pad"):
            orig_pad = scene.pad

            def pad(start, duration, _orig=orig_pad):
                t = scene.renderer.time
                _orig(start, duration)
                state["idle"][state["current"]] = state["idle"].get(state["current"], 0.0) + (scene.renderer.time - t)
            scene.pad = pad

        # ---- track the longest frozen stretch (consecutive wait() time with no play() in between)
        gap = {"acc": 0.0, "acc_start": 0.0}
        seg_gap = {}
        orig_play, orig_wait = scene.play, scene.wait

        def play(*a, **k):
            # Manim CE implements wait() as play(Wait(...)): a play of Wait animations only is a pause, so it must not
            # count as an animation start/span (sync and window checks) nor reset the frozen-stretch counter
            if a and all(type(x).__name__ == "Wait" for x in a):
                return orig_play(*a, **k)
            gap["acc"] = 0.0
            if fix_layout:   # same run-time label placement as the final render (render.py harness)
                a = a + tuple(layout_fix.place_labels(scene, a, float(config.frame_width), float(config.frame_height),
                                                      state["nudges"].setdefault(state["current"], [])))
            # start times of animations, relative to the segment start (used for narration-sync checks)
            t_rel = scene.renderer.time - state["seg_start"]
            state["plays"].setdefault(state["current"], []).append(round(t_rel, 2))
            r = orig_play(*a, **k)
            # [start, end] of every animation: used to measure how much of a silent visual window is animated
            state["spans"].setdefault(state["current"], []).append(
                [round(t_rel, 2), round(scene.renderer.time - state["seg_start"], 2)])
            return r

        def wait(*a, **k):
            t = scene.renderer.time
            r = orig_wait(*a, **k)
            dt = scene.renderer.time - t
            if gap["acc"] == 0.0:
                gap["acc_start"] = t
            gap["acc"] += dt
            best = seg_gap.get(state["current"])
            if best is None or gap["acc"] > best[0]:
                seg_gap[state["current"]] = (gap["acc"], gap["acc_start"])
            return r

        scene.play, scene.wait = play, wait

        # ---- wrap segment_N methods
        def make_seg(i, orig):
            def seg(*a, **k):
                state["current"] = i
                gap["acc"] = 0.0
                start = scene.renderer.time
                state["seg_start"] = start
                try:
                    return orig(*a, **k)
                finally:
                    g = seg_gap.get(i, (0.0, start))
                    rec = {"index": i, "start": start, "end": scene.renderer.time,
                           "idle_sec": state["idle"].get(i, 0.0), "mobjects_at_end": len(scene.mobjects),
                           "max_gap_sec": g[0], "max_gap_start": g[1] - start,
                           "play_starts": state["plays"].get(i, []),
                           "play_spans": state["spans"].get(i, []),
                           "labels_moved": state["nudges"].get(i, [])}
                    try:
                        rec.update(inspect_frame(scene, config, Text, MarkupText))
                    except Exception:
                        pass
                    records.append(rec)
            return seg

        i = 1
        while hasattr(scene, f"segment_{i}"):
            setattr(scene, f"segment_{i}", make_seg(i, getattr(scene, f"segment_{i}")))
            i += 1

        scene.render()
        report.update(ok=True, total_time=float(scene.renderer.time), segments=records,
                      frame=[float(config.frame_width), float(config.frame_height)])
    except Exception as e:
        tb = e.__traceback__
        line = None
        while tb is not None:  # innermost frame that belongs to the generated file
            if Path(tb.tb_frame.f_code.co_filename).resolve() == Path(scene_path).resolve():
                line = tb.tb_lineno
            tb = tb.tb_next
        report.update(
            ok=False,
            error_type=type(e).__name__,
            error_message=str(e)[:500],
            error_line=line,
            error_code=linecache.getline(str(scene_path), line) if line else "",
            error_segment=state["current"],
            traceback="".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:],
            segments=records,
        )
    write_report(report_path, report)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--no-layout-fix"):
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], fix_layout=len(sys.argv) == 3)