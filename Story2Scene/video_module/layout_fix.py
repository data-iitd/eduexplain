"""
Deterministic label placement, applied to generated scenes at run time (exec_check.py and render.py's harness).

The generator places labels with next_to/move_to and often lands them on an arrow, a bond or a circle, or past the
frame edge; told so in text feedback, it rarely fixes it. So before every play(), each label the animation is about
to show (a Text / MarkupText not on screen yet, or a small label unit such as text + BackgroundRectangle) is moved
to the nearest position that is inside the frame and clear of everything on screen. The collision geometry is the
one exec_check.py measures with (it imports it from here), so the check and the fixer agree. Running the same pass in
the headless check and in the final render keeps both identical (Text sizes are in scene units, not pixels).
"""
import math

CLEAR = 0.08        # label-to-shape clearance (a label touching a bond reads as overlapping on screen)
FRAME_MARGIN = 0.12
TEXT_TYPES = ("Text", "MarkupText", "Paragraph")


def _is_text(m):
    return type(m).__name__ in TEXT_TYPES


def box(m):
    """(l, r, b, t) of a mobject in scene units."""
    return (float(m.get_left()[0]), float(m.get_right()[0]), float(m.get_bottom()[1]), float(m.get_top()[1]))


def shape_box(m):
    """Box of a non-text shape; near-zero-thickness lines get their visible stroke."""
    l, r, b, t = box(m)
    if t - b < 0.06:
        c = (t + b) / 2
        b, t = c - 0.03, c + 0.03
    if r - l < 0.06:
        c = (l + r) / 2
        l, r = c - 0.03, c + 0.03
    return l, r, b, t


def contains(outer, inner, tol=0.02):
    sl, sr, sb, st = outer
    l, r, b, t = inner
    return l >= sl - tol and r <= sr + tol and b >= sb - tol and t <= st + tol


def text_hits_shape(tb, sb_):
    """exec_check's label/shape collision rule (containers that fully hold the label are fine)."""
    if contains(sb_, tb):
        return False
    l, r, b, t = tb
    sl, sr, sbot, st = sb_
    m = CLEAR
    w = min(r + m, sr) - max(l - m, sl)
    h = min(t + m, st) - max(b - m, sbot)
    smaller = min((r - l + 2 * m) * (t - b + 2 * m), (sr - sl) * (st - sbot))
    return w > 0.03 and h > 0.03 and (w * h) / max(smaller, 1e-6) > 0.1


def texts_overlap(a, b_):
    """exec_check's text/text overlap rule."""
    la, ra, ba, ta = a
    lb, rb, bb, tb = b_
    w = min(ra, rb) - max(la, lb)
    h = min(ta, tb) - max(ba, bb)
    if w > 0.05 and h > 0.05:
        small = min((ra - la) * (ta - ba), (rb - lb) * (tb - bb))
        return small > 0 and (w * h) / small > 0.2
    return False


def out_of_frame(bx, fw, fh, tol=0.05):
    l, r, b, t = bx
    return l < -fw / 2 - tol or r > fw / 2 + tol or b < -fh / 2 - tol or t > fh / 2 + tol


def split_family(mobjects):
    """Texts and drawable non-text shapes among the families of mobjects (images and glyphs excluded)."""
    texts, text_parts = [], set()
    for top in mobjects:
        for sub in top.get_family():
            if _is_text(sub):
                texts.append(sub)
                text_parts.update(id(x) for x in sub.get_family())
    shapes = []
    for top in mobjects:
        for sub in top.get_family():
            if id(sub) in text_parts or type(sub).__name__ == "ImageMobject":
                continue
            if not getattr(sub, "has_points", lambda: False)():
                continue
            shapes.append(sub)
    return texts, shapes


# ------------------------------------------------------------------------------------------------ placement
def _anim_mobjects(anims):
    """Every mobject the animations will put on screen."""
    out = []
    for a in anims:
        subs = getattr(a, "animations", None)
        if subs:                                   # AnimationGroup / LaggedStart / Succession
            out += _anim_mobjects(subs)
            continue
        if type(a).__name__ == "_AnimationBuilder":   # mob.animate...: moves something already on screen
            continue
        tgt = getattr(a, "target_mobject", None)
        # Transform(a, b): b's layout is what ends on screen. FadeIn & co. are Transforms too, but their target is
        # an empty placeholder until begin(), so only a target with content counts.
        if (tgt is not None and getattr(a, "mobject", None) is not None and tgt is not a.mobject
                and (tgt.submobjects or getattr(tgt, "has_points", lambda: False)())):
            out.append(tgt)
        elif getattr(a, "mobject", None) is not None:
            out.append(a.mobject)
        elif hasattr(a, "get_family"):             # a bare Mobject passed to play()
            out.append(a)
    return out


def _units(new_top, on_screen_ids, fw, fh):
    """Movable label units inside a newly shown mobject: the whole mobject when it is label-sized (text plus its
    box/background), otherwise each new text that is not held by a container of the same mobject."""
    fam_texts = [s for s in new_top.get_family() if _is_text(s) and id(s) not in on_screen_ids]
    if not fam_texts:
        return []
    try:
        l, r, b, t = box(new_top)
    except Exception:
        return []
    if (r - l) <= 0.55 * fw and (t - b) <= 0.3 * fh and len(fam_texts) <= 2:
        return [new_top]
    _, own_shapes = split_family([new_top])
    units = []
    for tx in fam_texts:
        tb = box(tx)
        if any(contains(shape_box(s), tb) for s in own_shapes):
            continue                               # e.g. "CO2" inside its atom circle: part of the drawing
        units.append(tx)
    return units


CLOSED = ("Circle", "Ellipse", "Rectangle", "RoundedRectangle", "Square", "Polygon", "RegularPolygon", "Triangle",
          "Annulus", "Sector", "Star")


def _fit_container(unit, containers):
    """A bare label centred inside a closed shape (move_to(circle)) but wider than it was meant to sit inside it:
    shrink it (to >= 55%) until the shape fully holds it, instead of pushing it out."""
    if not _is_text(unit):
        return False
    l, r, b, t = box(unit)
    cx, cy = (l + r) / 2, (b + t) / 2
    for kind, (sl, sr, sb, st) in containers:
        if not (sl < cx < sr and sb < cy < st) or (st - sb) < 1.3 * (t - b) or contains((sl, sr, sb, st), (l, r, b, t)):
            continue
        inner = 0.72 if kind in ("Circle", "Ellipse", "Annulus", "Sector") else 0.88
        f = min(inner * (sr - sl) / (r - l), inner * (st - sb) / (t - b), 1.0)
        if f < 0.55:
            continue
        unit.scale(f)
        unit.move_to([(sl + sr) / 2 if abs(cx - (sl + sr) / 2) < 0.3 * (sr - sl) else cx,
                      (sb + st) / 2 if abs(cy - (sb + st) / 2) < 0.3 * (st - sb) else cy, 0])
        return True
    return False


def _collides(ubox, obstacles_shapes, obstacles_texts, fw, fh):
    if out_of_frame(ubox, fw, fh, tol=-FRAME_MARGIN):
        return True
    for sb_ in obstacles_shapes:
        if text_hits_shape(ubox, sb_):
            return True
    for tb in obstacles_texts:
        if texts_overlap(ubox, tb) or texts_overlap(tb, ubox):
            return True
        # keep a little air between labels too
        l, r, b, t = ubox
        if min(r, tb[1]) - max(l, tb[0]) > -0.04 and min(t, tb[3]) - max(b, tb[2]) > -0.04:
            return True
    return False


def _search(ubox, ok, fw, fh, max_r=3.2, step=0.12, dirs=24):
    """Smallest shift (dx, dy) that makes ok(box) true; None when nothing within max_r works."""
    if ok(ubox):
        return 0.0, 0.0
    l, r, b, t = ubox
    k = step
    while k <= max_r + 1e-9:
        best = None
        for j in range(dirs):
            ang = 2 * math.pi * j / dirs
            dx, dy = k * math.cos(ang), k * math.sin(ang)
            cand = (l + dx, r + dx, b + dy, t + dy)
            if ok(cand):
                # prefer vertical/horizontal-ish moves over diagonal at the same radius (labels read better)
                score = abs(dx) * abs(dy)
                if best is None or score < best[0]:
                    best = (score, dx, dy)
        if best:
            return best[1], best[2]
        k += step
    return None


def _fit_frame(unit, fw, fh):
    """Shrink a unit that cannot fit the frame at all, then shift it fully inside."""
    l, r, b, t = box(unit)
    w, h = r - l, t - b
    maxw, maxh = fw - 2 * FRAME_MARGIN - 0.1, fh - 2 * FRAME_MARGIN - 0.1
    if w > maxw or h > maxh:
        unit.scale(min(maxw / max(w, 1e-6), maxh / max(h, 1e-6)))
        l, r, b, t = box(unit)
    dx = max(0.0, (-fw / 2 + FRAME_MARGIN) - l) - max(0.0, r - (fw / 2 - FRAME_MARGIN))
    dy = max(0.0, (-fh / 2 + FRAME_MARGIN) - b) - max(0.0, t - (fh / 2 - FRAME_MARGIN))
    if dx or dy:
        unit.shift([dx, dy, 0])


def _follow(m, moves):
    """A shape built from a label's position before the label was moved (Arrow(label.get_right(), ...),
    SurroundingRectangle(label)) moves with it: line ends inside a moved label's old box shift by its move, and a
    closed shape around the old box shifts whole."""
    if not moves or _is_text(m):
        return
    try:
        if hasattr(m, "get_start") and hasattr(m, "put_start_and_end_on"):
            p0, p1 = m.get_start(), m.get_end()

            def nearest(p):   # the moved label whose old box is closest to this end (within 0.5)
                best = None
                for (l, r, b, t), dx, dy in moves:
                    # an end made from a label (label.get_right() + buff) sits on or just outside its edge; one deep
                    # inside the old box was anchored to something else (e.g. the circle the label overflowed)
                    if min(p[0] - l, r - p[0], p[1] - b, t - p[1]) > 0.05:
                        continue
                    d = math.hypot(max(l - p[0], 0.0, p[0] - r), max(b - p[1], 0.0, p[1] - t))
                    if d <= 0.5 and (best is None or d < best[0]):
                        best = (d, (dx, dy))
                return best[1] if best else None
            d0, d1 = nearest(p0), nearest(p1)
            if d0 or d1:
                d0, d1 = d0 or (0.0, 0.0), d1 or (0.0, 0.0)
                if d0 == d1:
                    m.shift([d0[0], d0[1], 0])
                else:
                    q0, q1 = p0.copy(), p1.copy()
                    q0[0] += d0[0]; q0[1] += d0[1]; q1[0] += d1[0]; q1[1] += d1[1]
                    m.put_start_and_end_on(q0, q1)
            return
        mb = box(m)
        for ob, dx, dy in moves:
            if contains(mb, ob, tol=0.0) and abs((mb[0] + mb[1]) - (ob[0] + ob[1])) < 0.6 \
                    and abs((mb[2] + mb[3]) - (ob[2] + ob[3])) < 0.6:
                m.shift([dx, dy, 0])
                return
    except Exception:
        pass


def place_labels(scene, anims, fw, fh, log=None):
    """Move the labels these animations are about to show to free, in-frame spots, let shapes built around a moved
    label's old spot follow it, and make on-screen labels that a new shape would cover yield. Returns the extra
    animations (label glides) to play together with anims."""
    try:
        moves = scene.__dict__.setdefault("_layout_moves", [])
        on_screen = list(scene.mobjects)
        on_ids = {id(s) for top in on_screen for s in top.get_family()}
        texts, shapes = split_family(on_screen)
        obst_shapes = [shape_box(s) for s in shapes]
        containers = [(type(s).__name__, box(s)) for s in shapes if type(s).__name__ in CLOSED]
        obst_texts = [box(x) for x in texts]
        moved = 0
        new_tops = [m for m in _anim_mobjects(anims) if id(m) not in on_ids]
        for top in new_tops:   # arrows/boxes made from labels moved in an earlier play()
            _follow(top, moves)
        for top in new_tops:
            # shapes arriving in this same play() are obstacles too (e.g. Create(arrow) next to Write(label))
            others = [shape_box(s) for m in new_tops if m is not top for s in split_family([m])[1]]
            for unit in _units(top, on_ids, fw, fh):
                unit_ids = {id(s) for s in unit.get_family()}
                # the unit's own drawing (other parts of a big group) still counts, minus the unit itself
                own_texts, own_shapes = split_family([top]) if unit is not top else ([], [])
                extra_s = others + [shape_box(s) for s in own_shapes if id(s) not in unit_ids]
                extra_t = [box(x) for x in own_texts if id(x) not in unit_ids]
                before = box(unit)
                _fit_container(unit, containers + [(type(x).__name__, box(x)) for m in new_tops
                                                   for x in split_family([m])[1] if type(x).__name__ in CLOSED
                                                   and id(x) not in unit_ids])
                _fit_frame(unit, fw, fh)
                ok = lambda bx: not _collides(bx, obst_shapes + extra_s, obst_texts + extra_t, fw, fh)
                shift = _search(box(unit), ok, fw, fh)
                if shift and (shift[0] or shift[1]):
                    unit.shift([shift[0], shift[1], 0])
                after = box(unit)
                dx, dy = (after[0] + after[1] - before[0] - before[1]) / 2, (after[2] + after[3] - before[2] - before[3]) / 2
                if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                    moved += 1
                    moves.append((before, dx, dy))
                    del moves[:-40]
                    if log is not None:
                        inner = [s for s in unit.get_family() if _is_text(s)]
                        log.append({"label": str(getattr(inner[0], "text", ""))[:40] if inner else "",
                                    "dx": round(dx, 2), "dy": round(dy, 2)})
                # what is now placed is an obstacle for the next label in this play()
                obst_texts.append(after)
                on_ids |= unit_ids
        if moved:   # arrows created in this same play() from the labels just moved
            for top in new_tops:
                if not any(_is_text(s) for s in top.get_family()):
                    _follow(top, moves[-moved:])
        # the other direction: a shape drawn onto a label that is already on screen. The label yields: it glides
        # to the nearest free spot during this same play() (an extra .animate.shift appended to the animations).
        extra = []
        new_ids = {id(x) for m in new_tops for x in m.get_family()}
        new_shape_boxes = [shape_box(x) for m in new_tops for x in split_family([m])[1]]
        stuck, yielded = [], set()
        if new_shape_boxes:
            all_shapes = obst_shapes + new_shape_boxes
            for tx in texts:
                tb = box(tx)
                if id(tx) in new_ids or not any(text_hits_shape(tb, sb_) for sb_ in new_shape_boxes):
                    continue
                if any(contains(sb_, tb) for sb_ in obst_shapes):
                    # the label sits in its own box/background (a card): moving it alone would break that, so the
                    # whole card glides when it is label-sized; otherwise the incoming shapes move (below)
                    card = next((top for top in on_screen if id(tx) in {id(x) for x in top.get_family()}), None)
                    cb = box(card) if card is not None else None
                    if card is not None and id(card) not in yielded and (cb[1] - cb[0]) <= 0.55 * fw \
                            and (cb[3] - cb[2]) <= 0.3 * fh:
                        fam = {id(x) for x in card.get_family()}
                        c_shapes = [shape_box(x) for x in shapes if id(x) not in fam] + new_shape_boxes
                        c_texts = [box(x) for x in texts if id(x) not in fam]
                        shift = _search(cb, lambda bx: not _collides(bx, c_shapes, c_texts, fw, fh), fw, fh, max_r=1.5)
                        if shift and (shift[0] or shift[1]):
                            extra.append(card.animate.shift([shift[0], shift[1], 0]))
                            yielded.add(id(card))
                            if log is not None:
                                log.append({"label": str(getattr(tx, "text", ""))[:40], "card_dx": round(shift[0], 2),
                                            "card_dy": round(shift[1], 2), "yield": True})
                            continue
                    if id(card) in yielded:
                        continue
                    stuck.append(tb)
                    if log is not None:
                        log.append({"label": str(getattr(tx, "text", ""))[:40], "stuck": "inside a container"})
                    continue
                others = [bx for bx in obst_texts if bx != tb]
                shift = _search(tb, lambda bx: not _collides(bx, all_shapes, others, fw, fh), fw, fh, max_r=1.5)
                if not shift:
                    stuck.append(tb)
                    if log is not None:
                        log.append({"label": str(getattr(tx, "text", ""))[:40], "stuck": "no free spot within 1.5"})
                if shift and (shift[0] or shift[1]):
                    extra.append(tx.animate.shift([shift[0], shift[1], 0]))
                    obst_texts = [bx for bx in obst_texts if bx != tb] + [(tb[0] + shift[0], tb[1] + shift[0],
                                                                            tb[2] + shift[1], tb[3] + shift[1])]
                    if log is not None:
                        log.append({"label": str(getattr(tx, "text", ""))[:40], "dx": round(shift[0], 2),
                                    "dy": round(shift[1], 2), "yield": True})
        if stuck and new_tops:
            # a covered label has nowhere to go: move what this play() brings in instead (e.g. a box and its title),
            # all together, by the smallest shift that clears the on-screen labels and stays in the frame
            parts = [(m, box(m)) for m in new_tops]
            l = min(b[0] for _, b in parts); r = max(b[1] for _, b in parts)
            bot = min(b[2] for _, b in parts); t = max(b[3] for _, b in parts)
            moved_texts = [bx for bx in obst_texts if bx not in stuck] + stuck

            def clear(bx):
                dx, dy = bx[0] - l, bx[2] - bot
                if out_of_frame(bx, fw, fh, tol=0.02):
                    return False
                return not any(text_hits_shape(tb, (sb_[0] + dx, sb_[1] + dx, sb_[2] + dy, sb_[3] + dy))
                               for tb in moved_texts for sb_ in new_shape_boxes)
            shift = _search((l, r, bot, t), clear, fw, fh, max_r=1.5)
            if shift and (shift[0] or shift[1]):
                for m, b in parts:
                    m.shift([shift[0], shift[1], 0])
                    moves.append((b, shift[0], shift[1]))
                del moves[:-40]
                if log is not None:
                    log.append({"shapes_moved": len(parts), "dx": round(shift[0], 2), "dy": round(shift[1], 2)})
        return extra
    except Exception as exc:   # never break a scene over layout
        if log is not None:
            log.append({"error": f"{type(exc).__name__}: {exc}"[:200]})
        return []
