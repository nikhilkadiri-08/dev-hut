"""Open-literature reference values, kept separate from the workbook.

Why this exists
---------------
The supplied workbook has one worked circuit and two impulse shapes. A new test
object, a different Um, or a later IEC edition will not be in that file. This
module loads published typical values from ``data/literature/open_references.json``
so the optimiser and the UI have something cited to compare against.

What it is not
--------------
It is not a substitute for a measured capacitance, a nameplate, or the clause
text of the edition the laboratory actually uses. A lookup result always carries
its sources and a ``needs_verification`` flag. Nothing here silently replaces a
workbook formula or a value the operator typed.

The JSON file is the place to add a new paper. Do not put unsourced numbers in
Python.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from backend.units import us_to_s
from physics_model.calculations import Caveat, Severity
from physics_model.inputs import ImpulseTarget, ImpulseType, Veracity

__all__ = [
    "LITERATURE_PATH",
    "Citation",
    "LiteratureWindow",
    "InsulationLevel",
    "TypicalRange",
    "LiteratureHit",
    "load_catalogue",
    "impulse_window",
    "literature_impulse_target",
    "insulation_level_for_um",
    "suggest_test_voltage_kv",
    "typical_ranges",
    "lookup",
    "workbook_conflicts",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
LITERATURE_PATH = _REPO_ROOT / "data" / "literature" / "open_references.json"


@dataclass(frozen=True)
class Citation:
    title: str
    url: str
    year: int | None
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "year": self.year,
            "excerpt": self.excerpt,
        }


def _citation(raw: dict[str, Any]) -> Citation:
    year = raw.get("year")
    return Citation(
        title=str(raw["title"]),
        url=str(raw.get("url") or ""),
        year=int(year) if year is not None else None,
        excerpt=str(raw.get("excerpt") or ""),
    )


@dataclass(frozen=True)
class LiteratureWindow:
    """An IEC-shaped acceptance window taken from a public citation, not H13."""

    id: str
    impulse_family: str
    edition: str
    designation: str
    front_time_us: float | None
    front_relative: float | None
    front_definition: str
    tail_time_us: float | None
    tail_relative: float | None
    crest_relative: float
    keywords: tuple[str, ...]
    sources: tuple[Citation, ...]
    conflict_with_workbook: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "impulse_family": self.impulse_family,
            "edition": self.edition,
            "designation": self.designation,
            "front_time_us": self.front_time_us,
            "front_min_us": self.front_min_us,
            "front_max_us": self.front_max_us,
            "front_definition": self.front_definition,
            "tail_time_us": self.tail_time_us,
            "tail_min_us": self.tail_min_us,
            "tail_max_us": self.tail_max_us,
            "crest_relative": self.crest_relative,
            "conflict_with_workbook": self.conflict_with_workbook,
            "sources": [s.to_dict() for s in self.sources],
            "veracity": Veracity.UNKNOWN.value,
        }

    @property
    def front_min_us(self) -> float | None:
        if self.front_time_us is None or self.front_relative is None:
            return None
        return self.front_time_us * (1.0 - self.front_relative)

    @property
    def front_max_us(self) -> float | None:
        if self.front_time_us is None or self.front_relative is None:
            return None
        return self.front_time_us * (1.0 + self.front_relative)

    @property
    def tail_min_us(self) -> float | None:
        if self.tail_time_us is None or self.tail_relative is None:
            return None
        return self.tail_time_us * (1.0 - self.tail_relative)

    @property
    def tail_max_us(self) -> float | None:
        if self.tail_time_us is None or self.tail_relative is None:
            return None
        return self.tail_time_us * (1.0 + self.tail_relative)


@dataclass(frozen=True)
class InsulationLevel:
    id: str
    um_kv: float
    switching_impulse_kv: tuple[float, ...]
    lightning_impulse_kv: tuple[float, ...]
    keywords: tuple[str, ...]
    source: Citation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "um_kv": self.um_kv,
            "standard_switching_impulse_kv": list(self.switching_impulse_kv),
            "standard_lightning_impulse_kv": list(self.lightning_impulse_kv),
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class TypicalRange:
    id: str
    kind: str
    label: str
    quantity: str
    typical_min: float
    typical_max: float
    unit: str
    note: str
    keywords: tuple[str, ...]
    sources: tuple[Citation, ...]

    def contains(self, value: float) -> bool:
        return self.typical_min <= value <= self.typical_max

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "quantity": self.quantity,
            "typical_min": self.typical_min,
            "typical_max": self.typical_max,
            "unit": self.unit,
            "note": self.note,
            "sources": [s.to_dict() for s in self.sources],
            "veracity": Veracity.UNKNOWN.value,
        }


@dataclass
class LiteratureHit:
    """What to show when a new object or rating is not in the workbook."""

    query: str
    windows: list[LiteratureWindow] = field(default_factory=list)
    insulation_levels: list[InsulationLevel] = field(default_factory=list)
    ranges: list[TypicalRange] = field(default_factory=list)
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return bool(self.windows or self.insulation_levels or self.ranges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matched": self.matched,
            "windows": [w.to_dict() for w in self.windows],
            "insulation_levels": [level.to_dict() for level in self.insulation_levels],
            "ranges": [item.to_dict() for item in self.ranges],
            "caveats": [
                {"code": c.code, "message": c.message, "severity": c.severity.value}
                for c in self.caveats
            ],
        }


@dataclass
class LiteratureCatalogue:
    disclaimer: str
    retrieved: str
    windows: tuple[LiteratureWindow, ...]
    insulation_levels: tuple[InsulationLevel, ...]
    ranges: tuple[TypicalRange, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "retrieved": self.retrieved,
            "impulse_windows": [w.to_dict() for w in self.windows],
            "insulation_levels": [level.to_dict() for level in self.insulation_levels],
            "ranges": [item.to_dict() for item in self.ranges],
        }


def _tokens(text: str) -> set[str]:
    return {part for part in re.findall(r"[a-z0-9./]+", text.lower()) if len(part) > 1}


def _score(query_tokens: set[str], keywords: Iterable[str]) -> int:
    haystack = _tokens(" ".join(keywords))
    return len(query_tokens & haystack)


@lru_cache(maxsize=1)
def load_catalogue() -> LiteratureCatalogue:
    if not LITERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Literature catalogue missing at {LITERATURE_PATH}. "
            "Add cited entries there; do not invent them in code."
        )
    raw = json.loads(LITERATURE_PATH.read_text(encoding="utf-8"))
    windows = tuple(
        LiteratureWindow(
            id=item["id"],
            impulse_family=item["impulse_family"],
            edition=item["edition"],
            designation=item["designation"],
            front_time_us=item.get("front_time_us"),
            front_relative=item.get("front_relative"),
            front_definition=item.get("front_definition") or "",
            tail_time_us=item.get("tail_time_us"),
            tail_relative=item.get("tail_relative"),
            crest_relative=float(item["crest_relative"]),
            keywords=tuple(item.get("keywords") or ()),
            sources=tuple(_citation(src) for src in item.get("sources") or ()),
            conflict_with_workbook=item.get("conflict_with_workbook"),
        )
        for item in raw["impulse_windows"]
    )
    levels = tuple(
        InsulationLevel(
            id=item["id"],
            um_kv=float(item["um_kv"]),
            switching_impulse_kv=tuple(item["standard_switching_impulse_kv"]),
            lightning_impulse_kv=tuple(item["standard_lightning_impulse_kv"]),
            keywords=tuple(item.get("keywords") or ()),
            source=_citation(item["source"]),
        )
        for item in raw["insulation_levels"]
    )
    ranges = tuple(
        TypicalRange(
            id=item["id"],
            kind=item.get("kind") or item.get("quantity") or item["id"],
            label=item.get("label") or item["id"],
            quantity=item["quantity"],
            typical_min=float(item["typical_min"]),
            typical_max=float(item["typical_max"]),
            unit=item["unit"],
            note=item.get("note") or "",
            keywords=tuple(item.get("keywords") or ()),
            sources=tuple(_citation(src) for src in item.get("sources") or ()),
        )
        for item in [*raw["test_objects"], *raw["circuit_typical"]]
    )
    return LiteratureCatalogue(
        disclaimer=raw["disclaimer"],
        retrieved=raw["retrieved"],
        windows=windows,
        insulation_levels=levels,
        ranges=ranges,
    )


def impulse_window(impulse_family: str, *, edition_contains: str = "2010") -> LiteratureWindow:
    """The cited window for one impulse family, defaulting to the 2010 edition."""
    family = impulse_family.lower()
    matches = [
        window
        for window in load_catalogue().windows
        if window.impulse_family == family and edition_contains in window.edition
    ]
    if not matches:
        raise KeyError(
            f"No literature window for impulse_family={impulse_family!r} "
            f"edition containing {edition_contains!r}"
        )
    return matches[0]


def literature_impulse_target(impulse_type: ImpulseType) -> ImpulseTarget:
    """IEC-2010 window as an ``ImpulseTarget``, distinct from the workbook's H13.

    Switching tail uses +/-60% here. The optimiser still uses the workbook target
    so that the synthetic dataset and the ML residuals stay consistent. Call this
    when comparing a prediction against the publicly cited standard, not when
    generating the ML feature row.
    """
    family = (
        "lightning" if impulse_type is ImpulseType.LIGHTNING else "switching"
    )
    window = impulse_window(family)
    if window.front_time_us is None or window.tail_time_us is None:
        raise ValueError(f"{window.id} is an edition note, not a complete window")
    return ImpulseTarget(
        impulse_type=impulse_type,
        front_time_s=us_to_s(window.front_time_us),
        tail_time_s=us_to_s(window.tail_time_us),
        front_min_s=us_to_s(window.front_min_us or 0.0),
        front_max_s=us_to_s(window.front_max_us or 0.0),
        tail_fraction=window.tail_relative or 0.0,
        crest_fraction=window.crest_relative,
        tolerance_veracity=Veracity.UNKNOWN,
    )


def insulation_level_for_um(um_kv: float, *, tolerance_kv: float = 20.0) -> InsulationLevel | None:
    """Nearest published Um table within ``tolerance_kv``, or ``None``."""
    ranked = sorted(
        load_catalogue().insulation_levels,
        key=lambda level: abs(level.um_kv - um_kv),
    )
    if not ranked:
        return None
    best = ranked[0]
    if abs(best.um_kv - um_kv) > tolerance_kv:
        return None
    return best


def suggest_test_voltage_kv(
    um_kv: float, impulse_type: ImpulseType
) -> tuple[float, ...] | None:
    """Standard rated withstand voltages for a highest voltage for equipment.

    For a 400 kV system this returns the Um = 420 kV column of IEC 60071-1
    Table 3, which includes the workbook's 1425 kV lightning level. That does
    not make 1425 kV a measured laboratory value.
    """
    level = insulation_level_for_um(um_kv)
    if level is None:
        return None
    if impulse_type is ImpulseType.LIGHTNING:
        return level.lightning_impulse_kv
    return level.switching_impulse_kv


def typical_ranges(kind: str | None = None, quantity: str | None = None) -> list[TypicalRange]:
    items = list(load_catalogue().ranges)
    if kind is not None:
        items = [item for item in items if item.kind == kind]
    if quantity is not None:
        items = [item for item in items if item.quantity == quantity]
    return items


_STANDING_CAVEATS = (
    Caveat(
        "LITERATURE_IS_NOT_A_MEASUREMENT",
        "These figures come from public papers, manufacturer pages and publicly "
        "cited standard tables. They are not measurements of this generator or "
        "this test object, and they do not replace a nameplate or a verification "
        "shot.",
        Severity.WARNING,
    ),
    Caveat(
        "WORKBOOK_FORMULAS_UNCHANGED",
        "Literature values are a reference overlay. The physics equations remain "
        "the workbook ports in calculations.py; the ML feature row still uses "
        "those ports.",
        Severity.NOTICE,
    ),
)


def lookup(query: str) -> LiteratureHit:
    """Match a free-text description of a new object, voltage or impulse type.

    Scoring is keyword overlap only. A weak match is still returned with sources
    so an engineer can reject it; a miss returns the standing caveats and nothing
    else, rather than a guessed number.
    """
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("lookup query must not be empty")
    tokens = _tokens(cleaned)
    catalogue = load_catalogue()

    windows = [
        window
        for window in catalogue.windows
        if _score(tokens, (*window.keywords, window.impulse_family, window.edition)) > 0
    ]
    levels = [
        level
        for level in catalogue.insulation_levels
        if _score(tokens, (*level.keywords, f"{level.um_kv:g} kv")) > 0
    ]
    # Also accept a bare voltage number that sits on a published Um.
    voltage_match = re.search(r"(\d+(?:\.\d+)?)\s*kv", cleaned.lower())
    if voltage_match:
        um = float(voltage_match.group(1))
        nearby = insulation_level_for_um(um, tolerance_kv=30.0)
        if nearby is not None and nearby not in levels:
            levels.append(nearby)

    ranges = [
        item
        for item in catalogue.ranges
        if _score(tokens, (*item.keywords, item.kind, item.quantity, item.label)) > 0
    ]
    ranges.sort(key=lambda item: (-_score(tokens, (*item.keywords, item.kind)), item.id))
    windows.sort(key=lambda window: window.id)
    levels.sort(key=lambda level: level.um_kv)

    caveats = list(_STANDING_CAVEATS)
    if not (windows or levels or ranges):
        caveats.append(
            Caveat(
                "NO_LITERATURE_MATCH",
                f"Nothing in the open-reference catalogue matched {cleaned!r}. "
                "Do not invent a default. Measure the object or add a cited paper "
                "to data/literature/open_references.json.",
                Severity.WARNING,
            )
        )
    return LiteratureHit(
        query=cleaned,
        windows=windows,
        insulation_levels=levels,
        ranges=ranges,
        caveats=caveats,
    )


def workbook_conflicts() -> list[dict[str, str]]:
    """Places where a public citation disagrees with the workbook, for the UI."""
    return [
        {
            "id": window.id,
            "edition": window.edition,
            "detail": window.conflict_with_workbook,
        }
        for window in load_catalogue().windows
        if window.conflict_with_workbook
    ]
