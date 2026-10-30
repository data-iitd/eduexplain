from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class ModalType(str, Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass
class Resource:
    """A single item of the fixed explanation set R_q for a query."""
    rid: str
    modality: ModalType
    content: str                       # passage text, or caption/description for a figure
    concepts: Set[str] = field(default_factory=set)   # Con(r): concepts this resource's summary mentions
    relevance: float = 0.0             # s(q, r) = cos(E(q), E(desc(r))), set by pipeline.attach_relevance
    rtype: str = ""                    # benchmark resource_type (explanation/example/activity/figure/...)


@dataclass
class Segment:
    """A filled, timed, realised slot: sigma_l = <rho_l, S_l, tau_l, u_l, o_l>."""
    role: str                                   # rho_l
    resources: List[Resource] = field(default_factory=list)   # S_l (grounding sub-collection)
    duration: float = 0.0                       # tau_l
    narration: Optional[str] = None             # u_l
    onscreen: Optional[str] = None               # o_l
    src: Optional[str] = None                    # 'figure' | 'text_render' | 'synthetic'
    trace: dict = field(default_factory=dict)    # realisation trace: drafts, attempts, raw on-screen

    @property
    def text_resources(self) -> List[Resource]:
        return [r for r in self.resources if r.modality == ModalType.TEXT]

    @property
    def figure(self) -> Optional[Resource]:
        return next((r for r in self.resources if r.modality == ModalType.IMAGE), None)


@dataclass
class Storyboard:
    """B_q = (sigma_1, ..., sigma_L)."""
    query: str
    segments: List[Segment] = field(default_factory=list)

    def total_time(self) -> float:
        return sum(s.duration for s in self.segments)

    def role_sequence(self) -> List[str]:
        return [s.role for s in self.segments]

    def summary(self) -> str:
        lines = [f'Storyboard for: "{self.query}"  (total {self.total_time():.1f}s)']
        for i, s in enumerate(self.segments, 1):
            res_ids = ", ".join(r.rid for r in s.resources) or "-"
            lines.append(
                f"  [{i}] {s.role:<12} {s.duration:5.1f}s  src={str(s.src):<11} "
                f"grounded_in=[{res_ids}]"
            )
            lines.append(f"       narration: {s.narration}")
            lines.append(f"       onscreen : {s.onscreen}")
        return "\n".join(lines)
