"""
Storyboard figure resources: a segment whose `resource_ids` contains FIG_xxxx must SHOW that textbook figure and
annotate it as its `on_screen` says (arrows, highlights, labels). Shared by the Manim pipeline and VideoHyperFrames.

  resolve   FIG id -> PNG via figure_id_mapping.csv (image_path is relative to the folder above figures/)
  describe  the figure's text description from the Stage-1 resource file (context for the generator)
  ground    locate the parts on_screen/narration point at, so the text-only generator can place every annotation
            on the right part: the verifier VLM (Qwen3.5-9B with vision) names the parts with coarse boxes; a part
            that matches one of the figure's PRINTED labels (RapidOCR) is anchored exactly where that label's
            leader line ends (the VLM alone is too imprecise for small parts); others keep the VLM box
Results are attached to each segment as seg["figures"] = [{id, path, file, width, height, description, regions}],
with regions = [{label, box: [x0, y0, x1, y1]}] in image pixels (origin top-left).
"""
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

FIG_ROOT = Path(os.getenv("VIDEO_FIGURES_DIR", "/mnt/storage/aryan/figures"))
MAPPING_CSV = Path(os.getenv("VIDEO_FIGURE_MAPPING", str(FIG_ROOT / "figure_id_mapping.csv")))
RESOURCES_JSON = Path(os.getenv("VIDEO_RESOURCES_JSON",
                                "/mnt/storage/aryan/setCore/all_resources_unique_ids_clean1_merged.json"))
FIG_RE = re.compile(r"^FIG_\d+$")


@lru_cache(maxsize=1)
def load_mapping():
    """{FIG id: absolute PNG path}."""
    out = {}
    with open(MAPPING_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = (r.get("image_path") or "").replace("\\", "/")
            cands = [FIG_ROOT.parent / rel, FIG_ROOT / rel,
                     FIG_ROOT / (r.get("document_id") or "") / (r.get("file_name") or "")]
            path = next((p.resolve() for p in cands if p.is_file()), None)
            if path:
                out[r["resource_id"].strip()] = str(path)
    return out


@lru_cache(maxsize=1)
def load_descriptions():
    """{FIG id: description} from the Stage-1 resource file (optional; empty if it is not there)."""
    try:
        data = json.loads(RESOURCES_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for r in data if isinstance(data, list) else []:
        rid = str(r.get("resource_id", ""))
        if FIG_RE.match(rid):
            txt = next((str(r[k]) for k in ("description", "summary", "image_caption")
                        if r.get(k) and str(r[k]) != "None"), "")
            out[rid] = txt[:600]
    return out


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", head[16:24])
    raise ValueError(f"not a PNG: {path}")


def segment_figure_ids(seg):
    return [str(r).strip() for r in seg.get("resource_ids", []) if FIG_RE.match(str(r).strip())]


# ------------------------------------------------------- printed labels + leader lines
# NCERT figures label their own parts ("Magnesium oxide ---->"). OCR finds the printed label exactly and the
# leader line leaving it ends on the part itself, which is far more precise than a 9B VLM box for small parts.

def merge_lines(items, gap=14):
    """Join OCR words stacked as one multi-line label (same left edge, small vertical gap)."""
    items = sorted(items, key=lambda r: (r["box"][1], r["box"][0]))
    out = []
    for it in items:
        for o in out:
            ox0, oy0, ox1, oy1 = o["box"]; x0, y0, x1, y1 = it["box"]
            if abs(x0 - ox0) <= 12 and 0 <= y0 - oy1 <= gap:
                o["text"] += " " + it["text"]; o["box"] = [min(ox0, x0), oy0, max(ox1, x1), y1]; break
        else:
            out.append({"text": it["text"], "box": list(it["box"])})
    return out

def trace_leader(gray, box, dark=110, min_len=25, max_gap=14, tol=1.6):
    """Follow a straight leader line leaving the label box; returns (end_x, end_y, length) or None.
    The line may start a few px from the OCR box; its direction is locked from the first 12 px so the trace
    stops where the straight line ends instead of sliding along the drawing's outlines."""
    H, W = gray.shape
    x0, y0, x1, y1 = box
    best = None
    rows = range(max(0, y0 - 6), min(H, min(y1, y0 + 30) + 6))   # leader lines attach near the first text line
    for side in (-1, 1):
        edge = x0 if side < 0 else x1
        for y in rows:
            sx = next((edge + side * k for k in range(2, max_gap) if 0 <= edge + side * k < W
                       and gray[y, edge + side * k] <= dark), None)
            if sx is None:
                continue
            x, cy, misses, length, slope, y_start, pts = sx, float(y), 0, 0, None, float(y), []
            while 0 <= x + side < W and misses < 4:
                x += side
                pred = cy if slope is None else y_start + slope * (x - sx) * side
                cands = [yy for yy in range(int(round(pred - tol)), int(round(pred + tol)) + 1)
                         if 0 <= yy < H and gray[yy, x] <= dark]
                if cands:
                    cy = min(cands, key=lambda yy: abs(yy - pred)); misses = 0; length += 1; pts.append((x, cy))
                    if slope is None and length == 12:
                        slope = (cy - y_start) / 12.0
                else:
                    misses += 1
            # a leader line is thin: reject runs that are really the edge of a filled shape
            if length >= min_len and (best is None or length > best[2]):
                ex, ey = pts[-1]
                best = (ex, int(round(ey)), length)
    return best


def _leaders(path, ocr_items):
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.int16)
    out = []
    for lab in merge_lines(ocr_items):
        t = trace_leader(gray, lab["box"])
        out.append({**lab, "end": None if t is None else [t[0], t[1]], "line_len": None if t is None else t[2]})
    return out


@lru_cache(maxsize=1)
def _ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR   # Apache-2.0, CPU, offline
    except ImportError:
        return None
    return RapidOCR()


def printed_labels(path):
    """[{text, box, end}] for the figure's printed labels; end = where its leader line points (or None)."""
    ocr = _ocr()
    if ocr is None:
        return []
    res, _ = ocr(str(path))
    items = [{"text": t, "box": [round(min(q[0] for q in b)), round(min(q[1] for q in b)),
                                 round(max(q[0] for q in b)), round(max(q[1] for q in b))]}
             for b, t, c in (res or []) if float(c) > 0.6 and len(t.strip()) >= 2]
    return _leaders(path, items)


def _tokens(text):
    return [w for w in re.findall(r"[a-z]+", text.lower().replace("-", " ")) if len(w) >= 3]


def _matches(name, label_text):
    """'the white powder in the watch glass' vs printed 'Watch-glass': every label word prefix-matches a name word."""
    lt, nt = _tokens(label_text), _tokens(name)
    return bool(lt) and all(any(n[:4] == l[:4] for n in nt) for l in lt)


# ------------------------------------------------------------------ grounding
GROUND_PROMPT = """This is a textbook figure that an educational video will show and annotate.

ANNOTATION INSTRUCTIONS (on_screen): {on_screen}
NARRATION: {narration}

For every part of the figure that the instructions or the narration point at (things to arrow, highlight,
circle, label or zoom into), give its bounding box. Use the figure's own printed labels and leader lines to find
the part itself (the drawn object, not its printed label text). Only include parts that are visible.

Answer with JSON only:
{{"regions": [{{"label": "<name as used in the instructions>", "box": [x0, y0, x1, y1]}}]}}
Coordinates are normalised to 0-1000 over the whole image: x0,y0 = top-left, x1,y1 = bottom-right, y grows down."""


def _parse_regions(raw, width, height):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw), flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in data.get("regions", []) if isinstance(data, dict) else []:
        try:
            x0, y0, x1, y1 = [float(v) for v in it["box"]]
        except (KeyError, TypeError, ValueError):
            continue
        x0, x1 = sorted((max(0.0, min(1000.0, x0)), max(0.0, min(1000.0, x1))))
        y0, y1 = sorted((max(0.0, min(1000.0, y0)), max(0.0, min(1000.0, y1))))
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue
        out.append({"label": str(it.get("label", ""))[:60],
                    "box": [round(x0 * width / 1000), round(y0 * height / 1000),
                            round(x1 * width / 1000), round(y1 * height / 1000)]})
    return out[:8]


def _box_around(x, y, w, h, r):
    return [max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)]


def ground_figure(client, path, width, height, on_screen, narration, max_tokens=800):
    """Locate the parts on_screen/narration annotate. The VLM names the parts (with coarse boxes); a part that
    matches a printed label is anchored where that label's leader line ends (precise), otherwise the VLM box is
    used. Regions: {label, box, point, source: label-leader|vlm}. Returns [] when nothing can be located."""
    vlm = []
    if client is not None:
        b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": GROUND_PROMPT.format(on_screen=on_screen, narration=narration)},
        ]}]
        try:
            vlm = _parse_regions(client.generate(messages, max_new_tokens=max_tokens, temperature=0.0), width, height)
        except Exception as exc:   # grounding is an aid: the video still shows the figure without it
            print(f"  ! figure grounding (VLM) failed for {Path(path).name}: {exc}")
    labels = [l for l in printed_labels(path) if l.get("end")]
    out, used = [], set()
    for r in vlm:
        lab = next((l for l in labels if _matches(r["label"], l["text"]) and l["text"] not in used), None)
        if lab:
            used.add(lab["text"])
            x, y = lab["end"]
            x0, y0, x1, y1 = r["box"]
            box = r["box"] if (x0 <= x <= x1 and y0 <= y <= y1) else _box_around(x, y, width, height, 34)
            out.append({"label": r["label"], "box": box, "point": [x, y], "source": "label-leader",
                        "printed_label": lab["text"], "printed_label_box": lab["box"]})
        else:
            x0, y0, x1, y1 = r["box"]
            out.append({**r, "point": [(x0 + x1) // 2, (y0 + y1) // 2], "source": "vlm"})
    # parts named in the instructions that the VLM missed but the figure labels itself
    want = f"{on_screen} {narration}"
    for lab in labels:
        if lab["text"] not in used and _matches(want, lab["text"]):
            x, y = lab["end"]
            out.append({"label": lab["text"], "box": _box_around(x, y, width, height, 34), "point": [x, y],
                        "source": "label-leader", "printed_label": lab["text"], "printed_label_box": lab["box"]})
    return out[:8]


def attach_figures(storyboard, assets_dir, grounder=None, log=print):
    """Copy of the storyboard whose figure segments carry seg["figures"]; images are copied to assets_dir.
    grounder: a vision-capable client (.generate with image content) or None to skip grounding.
    Grounding results are cached in assets_dir/grounding.json, keyed by figure id + on_screen + narration."""
    sb = json.loads(json.dumps(storyboard))
    mapping, desc = load_mapping(), load_descriptions()
    assets = Path(assets_dir)
    cache_path = assets / "grounding.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for seg in sb["segments"]:
        figs = []
        for fid in segment_figure_ids(seg):
            src = mapping.get(fid)
            if not src:
                log(f"  ! {seg.get('segment_id')}: figure {fid} is not in {MAPPING_CSV}")
                figs.append({"id": fid, "missing": True})
                continue
            assets.mkdir(parents=True, exist_ok=True)
            dst = assets / f"{fid}.png"
            if not dst.exists():
                shutil.copyfile(src, dst)
            w, h = png_size(dst)
            key = hashlib.sha1(f"{fid}|{seg.get('on_screen', '')}|{seg.get('narration', '')}".encode()).hexdigest()
            regions = cache.get(key)
            if regions is None and grounder is not None:
                regions = ground_figure(grounder, dst, w, h, seg.get("on_screen", ""), seg.get("narration", ""))
                cache[key] = regions
                log(f"  figure {fid}: grounded {len(regions)} region(s) " +
                    ", ".join(r["label"] for r in regions))
            figs.append({"id": fid, "path": str(dst.resolve()), "file": dst.name, "width": w, "height": h,
                         "description": desc.get(fid, ""), "regions": regions or []})
        if figs:
            seg["figures"] = figs
    if cache:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return sb


def figure_prompt_block(seg, coords="px"):
    """Text block describing a segment's figures for the generator / verifier prompts."""
    figs = [f for f in seg.get("figures", []) if not f.get("missing")]
    if not figs:
        return ""
    lines = ["FIGURES (textbook images that MUST be shown in this segment and annotated as on_screen says):"]
    for f in figs:
        lines.append(f"  {f['id']}: file {f['file']}, {f['width']}x{f['height']} px"
                     + (f"; shows: {f['description']}" if f.get("description") else ""))
        if f.get("regions"):
            lines.append("    parts located in this image (pixel boxes [x0, y0, x1, y1], origin top-left):")
            for r in f["regions"]:
                x0, y0, x1, y1 = r["box"]
                if coords == "pct":
                    pct = [round(100 * v / d, 1) for v, d in ((x0, f["width"]), (y0, f["height"]),
                                                              (x1 - x0, f["width"]), (y1 - y0, f["height"]))]
                    lines.append(f"      {r['label']}: box {r['box']}  (left {pct[0]}%, top {pct[1]}%, "
                                 f"width {pct[2]}%, height {pct[3]}% of the image)")
                else:
                    pt = r.get("point") or [(r["box"][0] + r["box"][2]) // 2, (r["box"][1] + r["box"][3]) // 2]
                    lines.append(f"      {r['label']}: point {pt} (aim arrows/dots here), box {r['box']}"
                                 + (" [exact: from the figure's printed label]" if r.get("source") == "label-leader"
                                    else " [approximate]"))
        else:
            lines.append("    (no located parts: annotate with callouts beside the image and a frame highlight)")
    return "\n".join(lines) + "\n"
