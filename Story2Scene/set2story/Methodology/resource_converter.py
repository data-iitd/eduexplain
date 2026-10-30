from typing import List, Dict, Any

from data import Resource, ModalType


# Paper §2: T = {passage, definition, example, figure, exercise, activity}; the
# benchmark additionally names passages "explanation". Every non-figure type is
# textual. Unknown types fall back on the id prefix rather than being dropped.
IMAGE_TYPES = {"figure", "image", "diagram"}
TEXT_TYPES = {"explanation", "passage", "definition", "example", "exercise", "activity"}


def _modality(resource_type: str, resource_id: str) -> ModalType:
    t = (resource_type or "").lower()
    if t in IMAGE_TYPES:
        return ModalType.IMAGE
    if t in TEXT_TYPES:
        return ModalType.TEXT
    if (resource_id or "").upper().startswith("FIG"):
        return ModalType.IMAGE
    if t or resource_id:
        return ModalType.TEXT
    raise ValueError(f"Cannot infer modality for resource_type={resource_type!r}, id={resource_id!r}")


def resource_from_json(data: Dict[str, Any]) -> Resource:
    """Convert one gold-set resource dict into a Resource.

    desc(r) is the resource's `summary` (passage text / figure description);
    `description` is used when `summary` is empty. This holds for every
    resource type, not only explanation/figure.
    """
    rid = data["resource_id"]
    rtype = data.get("resource_type", "")
    content = (data.get("summary") or data.get("description") or data.get("text") or "").strip()
    return Resource(
        rid=rid,
        modality=_modality(rtype, rid),
        content=content,
        concepts=set(data.get("concepts", [])),
        relevance=float(data.get("relevance", 0.0)),
        rtype=rtype,
    )


def resources_from_gold_set(gold_set: Dict[str, Any]) -> List[Resource]:
    """All resources of one gold set, de-duplicated by id (order preserved)."""
    out, seen = [], set()
    for r in gold_set.get("resources_used", []):
        if r["resource_id"] in seen:
            continue
        seen.add(r["resource_id"])
        out.append(resource_from_json(r))
    return out
