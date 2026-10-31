#!/usr/bin/env python3
"""Multimodal retrieval with OpenCLIP (ViT-L-14). Table 4, row 3.

Figures go through the IMAGE encoder, text resources and the query through the
TEXT encoder, and both land in CLIP's shared space, so a query can be scored
against a diagram directly rather than against its caption.

Image paths are read from --image-field (searched in the resource record) and
resolved against --image-root. Point --image-field at whatever your corpus
calls it; the defaults cover the common spellings.


Usage
-----
  python openclip_retriever.py --gold $Q --resources $R \\
      --image-root /path/to/figures --output pools/clip_top100.json

Install: pip install open_clip_torch pillow
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from retrieval_io import (first, index_text, is_figure, load_corpus,
                          query_text, report_texts, require_local_model,
                          write_candidates)

IMAGE_FIELDS = ("image_path", "image_file", "file_path", "figure_path",
                "path", "image", "filename", "file_name", "img_path")


def _candidates_for(v: str, root: Optional[str],
                    repo_root: Optional[str]) -> List[str]:
    """Every sensible place a recorded path might actually live.

    EduExplain records figure paths two ways: the resources JSON has
    `"./figures\\CLASS_10_SCI_CH_01\\..._figure_0.png"` -- backslashes from a
    Windows extraction run -- and the id-map CSV has the same path with
    forward slashes. On Linux a backslash path is ONE filename and resolves to
    nothing, so without normalising it every figure falls back to its caption
    and the multimodal row quietly becomes a text row.
    """
    v = v.strip().replace("\\", "/")
    while v.startswith("./"):
        v = v[2:]
    out = [v]
    for base in [b for b in (root, repo_root) if b]:
        out.append(os.path.join(base, v))
        out.append(os.path.join(base, os.path.basename(v)))
        # `--image-root` is often the figures/ directory itself, in which case
        # the recorded "figures/CLASS_.../x.png" needs its first part dropped.
        parts = v.split("/", 1)
        if len(parts) == 2:
            out.append(os.path.join(base, parts[1]))
    return out


class ImageResolver:
    """resource_id -> image file, from an id-map CSV and/or the resources.

    The CSV is authoritative when given: it is the mapping that was built
    alongside the figures, so it knows about renames the resources JSON may
    not. The resource record's own fields are the fallback.
    """

    def __init__(self, csv_path: Optional[str], root: Optional[str],
                 repo_root: Optional[str], fields: Sequence[str]):
        self.root, self.repo_root, self.fields = root, repo_root, fields
        self.map: Dict[str, List[str]] = {}
        if not csv_path:
            return
        import csv as _csv
        with open(os.path.expanduser(csv_path), newline="",
                  encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;|")
            except _csv.Error:
                dialect = _csv.excel_tab if "\t" in sample else _csv.excel
            rows = list(_csv.DictReader(fh, dialect=dialect))
        if not rows:
            raise SystemExit(f"{csv_path}: no rows")
        cols = {c.lower().strip(): c for c in rows[0]}
        rid_c = next((cols[c] for c in ("resource_id", "id") if c in cols),
                     None)
        if rid_c is None:
            raise SystemExit(
                f"{csv_path}: no resource_id column (saw "
                f"{', '.join(rows[0].keys())})")
        path_c = next((cols[c] for c in ("image_path", "path", "filepath")
                       if c in cols), None)
        file_c = next((cols[c] for c in ("file_name", "filename", "file")
                       if c in cols), None)
        doc_c = next((cols[c] for c in ("document_id", "doc_id")
                      if c in cols), None)
        for r in rows:
            rid = str(r.get(rid_c) or "").strip()
            if not rid:
                continue
            paths: List[str] = []
            if path_c and r.get(path_c):
                paths.append(str(r[path_c]))
            if file_c and r.get(file_c):
                fn = str(r[file_c])
                paths.append(fn)
                if doc_c and r.get(doc_c):
                    paths.append(f"figures/{r[doc_c]}/{fn}")
                    paths.append(f"{r[doc_c]}/{fn}")
            self.map[rid] = paths
        print(f"image map: {len(self.map)} ids from "
              f"{os.path.basename(csv_path)}"
              f" (columns: {rid_c}"
              + (f", {path_c}" if path_c else "")
              + (f", {file_c}" if file_c else "") + ")")

    def resolve(self, res: Dict) -> Optional[str]:
        rid = str(res.get("resource_id") or "")
        recorded = list(self.map.get(rid, []))
        for k in self.fields:
            v = res.get(k)
            if isinstance(v, str) and v.strip() and v.strip() != "None":
                recorded.append(v)
        for v in recorded:
            for cand in _candidates_for(v, self.root, self.repo_root):
                if os.path.exists(cand):
                    return cand
        return None


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--resources", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k",
                    help="OpenCLIP weight tag, or -- on a node with no "
                         "network -- the path to a downloaded checkpoint "
                         "(open_clip_pytorch_model.bin). A bare tag makes "
                         "open_clip reach for the Hub.")
    ap.add_argument("--cache-dir", default=None,
                    help="where open_clip looks for / writes weights")
    ap.add_argument("--image-root", default=None,
                    help="directory the recorded paths are relative to; "
                         "usually the EduExplain repo root (which contains "
                         "figures/) or the figures/ directory itself -- both "
                         "are tried")
    ap.add_argument("--image-map", default=None,
                    help="CSV mapping resource_id to the figure file, with "
                         "columns resource_id and image_path (and/or "
                         "file_name + document_id). Authoritative over the "
                         "paths recorded in the resources file.")
    ap.add_argument("--image-field", action="append", default=[],
                    help="resource key holding the image path; repeatable. "
                         f"Defaults to {', '.join(IMAGE_FIELDS)}")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--max-fallback", type=float, default=0.25,
                    help="refuse to finish if more than this fraction of "
                         "figures fell back to their caption")
    ap.add_argument("--allow-fallback", action="store_true")
    ns = ap.parse_args()

    import open_clip
    from PIL import Image

    gold, resources = load_corpus(ns.gold, ns.resources)
    texts = report_texts(resources)
    fields = ns.image_field or list(IMAGE_FIELDS)

    # Resolve the images BEFORE loading a multi-GB model: if none of them
    # resolve, the run is going to refuse anyway and there is no reason to
    # have paid for the weights first.
    repo_root = os.path.dirname(os.path.abspath(os.path.expanduser(
        ns.resources)))
    repo_root = os.path.dirname(repo_root) if os.path.basename(repo_root) \
        == "data" else repo_root
    resolver = ImageResolver(ns.image_map, ns.image_root, repo_root, fields)
    fig_idx = [i for i, r in enumerate(resources) if is_figure(r)]
    paths = {i: resolver.resolve(resources[i]) for i in fig_idx}
    if fig_idx and not any(paths.values()):
        raise SystemExit(
            f"None of the {len(fig_idx)} figures resolved to a file.\n"
            f"  --image-root {ns.image_root!r}   --image-map "
            f"{ns.image_map!r}\n  Recorded path of the first figure: "
            f"{resources[fig_idx[0]].get('image_path')!r}\n  Point "
            f"--image-root at the directory those paths are relative to.")

    if os.sep in ns.pretrained or ns.pretrained.startswith("~"):
        ns.pretrained = require_local_model(ns.pretrained,
                                            "--pretrained (OpenCLIP)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading OpenCLIP {ns.model} / {ns.pretrained} on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        ns.model, pretrained=ns.pretrained, device=device,
        cache_dir=ns.cache_dir)
    model.eval()
    tokenizer = open_clip.get_tokenizer(ns.model)
    with_image = [i for i in fig_idx if paths[i]]
    fell_back = [i for i in fig_idx if not paths[i]]
    print(f"figures: {len(fig_idx)}   with a resolvable image file: "
          f"{len(with_image)}   falling back to caption: {len(fell_back)}")
    if fig_idx and fell_back:
        frac = len(fell_back) / len(fig_idx)
        print(f"  [!] {frac:.1%} of figures have no image file. Those go "
              f"through the TEXT tower, so\n      this row is only partly "
              f"multimodal.")
        if fell_back[:3]:
            print("      e.g. " + ", ".join(
                str(resources[i].get("resource_id")) for i in fell_back[:3]))
        if frac > ns.max_fallback and not ns.allow_fallback:
            raise SystemExit(
                f"\nRefusing to write a 'multimodal retrieval' pool in which "
                f"{frac:.0%} of figures\nnever reached the image encoder. "
                f"Either point --image-root / --image-field at the\nfigure "
                f"files, or pass --allow-fallback and report the fallback "
                f"rate in the\ncaption.")

    dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else None
    E: List[Optional[np.ndarray]] = [None] * len(resources)

    # ---- images
    with torch.no_grad():
        for b in range(0, len(with_image), ns.batch_size):
            chunk = with_image[b:b + ns.batch_size]
            ims, ok = [], []
            for i in chunk:
                try:
                    ims.append(preprocess(Image.open(paths[i]).convert("RGB")))
                    ok.append(i)
                except Exception as exc:        # unreadable file, not fatal
                    print(f"  [skip] {paths[i]}: {type(exc).__name__}")
                    fell_back.append(i)
            if not ims:
                continue
            v = model.encode_image(torch.stack(ims).to(device))
            v = v.float().cpu().numpy()
            for k, i in enumerate(ok):
                E[i] = v[k]
            print(f"  images [{min(b + ns.batch_size, len(with_image))}/"
                  f"{len(with_image)}]")

    # ---- text (everything else, plus the figures that fell back)
    text_idx = [i for i in range(len(resources)) if E[i] is None]
    with torch.no_grad():
        for b in range(0, len(text_idx), ns.batch_size):
            chunk = text_idx[b:b + ns.batch_size]
            tok = tokenizer([texts[i] for i in chunk]).to(device)
            v = model.encode_text(tok).float().cpu().numpy()
            for k, i in enumerate(chunk):
                E[i] = v[k]
            print(f"  text [{min(b + ns.batch_size, len(text_idx))}/"
                  f"{len(text_idx)}]")

    R = l2(np.stack(E).astype(np.float32))
    qtexts = [query_text(it) for it in gold]
    with torch.no_grad():
        Q = []
        for b in range(0, len(qtexts), ns.batch_size):
            tok = tokenizer(qtexts[b:b + ns.batch_size]).to(device)
            Q.append(model.encode_text(tok).float().cpu().numpy())
    Q = l2(np.concatenate(Q).astype(np.float32))

    S = Q @ R.T
    k = min(ns.top_k, R.shape[0])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order, scores = [], []
    for i in range(len(qtexts)):
        idx = part[i][np.argsort(-S[i, part[i]], kind="stable")]
        order.append(idx.tolist())
        scores.append(S[i, idx].tolist())

    write_candidates(ns.output, gold, qtexts, resources, order, scores,
                     f"openclip:{ns.model}")

    # The fallback rate belongs WITH the pool, not only in a terminal that
    # scrolls away: the caption has to state it and the number must be
    # recoverable months later.
    side = os.path.splitext(ns.output)[0] + ".retriever.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump({"retriever": "openclip", "model": ns.model,
                   "pretrained": ns.pretrained, "dim": dim,
                   "n_figures": len(fig_idx),
                   "n_figures_as_image": len(fig_idx) - len(set(fell_back)),
                   "n_figures_fell_back": len(set(fell_back)),
                   "fallback_rate": (len(set(fell_back)) / len(fig_idx))
                   if fig_idx else 0.0,
                   "text_truncation_tokens": 77}, f, indent=2)
    print(f"retriever card -> {side}")
    print("CLIP's text tower truncates at 77 tokens; long summaries are cut.")


if __name__ == "__main__":
    main()
