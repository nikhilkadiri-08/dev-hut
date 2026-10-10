"""Structured, validated inputs for the impulse-generator physics model.

Everything in this package works in SI units internally: volts, farads, henries,
ohms, seconds and joules. The laboratory and the source workbook speak kV, pF, uF,
uH and us, so every entry point here offers an explicit ``from_lab_units``
constructor and no implicit conversion happens anywhere else.

Provenance discipline
---------------------
The source workbook mixes three very different kinds of number: ratings that came
from the problem statement, values that are just the one worked example, and
quantities that were never supplied at all. Treating them alike is how a prototype
ends up quietly asserting that an unknown is a specification, so every field that
carries engineering weight is tagged with a :class:`Veracity`.

This mirrors the coarse intent of ``backend/provenance.py`` but is kept local so the
physics model has no dependency on the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from backend.units import kv_to_v, pf_to_f, uf_to_f, uh_to_h, us_to_s

__all__ = [
    "Veracity",
    "ImpulseType",
    "ImpulseTarget",
    "IMPULSE_TARGETS",
    "IntraStageTopology",
    "CapacitorRatings",
    "GeneratorRatings",
    "ResistorStock",
    "ResistorInventory",
    "TestCircuit",
    "GeneratorConfiguration",
    "WORKBOOK_EXAMPLE_CIRCUIT",
    "WORKBOOK_EXAMPLE_RATINGS",
    "WORKBOOK_EXAMPLE_INVENTORY",
]


class Veracity(str, Enum):
    """How much weight a numeric input can bear.

    ``SPECIFIED``   stated as an equipment rating in the supplied material.
    ``ILLUSTRATIVE`` the value used by the workbook's single worked example. Fine
                    for exercising the model, not evidence about any real circuit.
    ``UNKNOWN``     not supplied. Must never be silently replaced by a default.
    """

    SPECIFIED = "specified"
    ILLUSTRATIVE = "illustrative"
    UNKNOWN = "unknown"


class ImpulseType(str, Enum):
    """Impulse shape.

    The string values match the workbook's ``B6`` and the ``Impulse_Type`` column of
    the synthetic dataset exactly, because the ML layer keys on them.
    """

    LIGHTNING = "Lightning"
    SWITCHING = "Switching"


@dataclass(frozen=True)
class ImpulseTarget:
    """Target waveform parameters and the acceptance window around them.

    Targets are workbook ``E5``/``E6``. The windows are workbook ``H13``, which
    applies an absolute window to the front/peak time and +/-20% to the tail time.

    The windows are shaped like IEC 60060-1 (front +/-30% for lightning, peak
    +/-20% for switching, peak voltage +/-3%), but the workbook never recorded which
    edition it followed, and the +/-20% it applies to the *switching* tail is
    tighter than some editions allow. ``tolerance_veracity`` records that doubt.
    """

    impulse_type: ImpulseType
    front_time_s: float
    tail_time_s: float
    front_min_s: float
    front_max_s: float
    tail_fraction: float
    crest_fraction: float
    tolerance_veracity: Veracity = Veracity.UNKNOWN

    @property
    def tail_min_s(self) -> float:
        return self.tail_time_s * (1.0 - self.tail_fraction)

    @property
    def tail_max_s(self) -> float:
        return self.tail_time_s * (1.0 + self.tail_fraction)

    def crest_window_v(self, requested_v: float) -> tuple[float, float]:
        return (
            requested_v * (1.0 - self.crest_fraction),
            requested_v * (1.0 + self.crest_fraction),
        )


IMPULSE_TARGETS: dict[ImpulseType, ImpulseTarget] = {
    ImpulseType.LIGHTNING: ImpulseTarget(
        impulse_type=ImpulseType.LIGHTNING,
        front_time_s=us_to_s(1.2),
        tail_time_s=us_to_s(50.0),
        front_min_s=us_to_s(0.84),
        front_max_s=us_to_s(1.56),
        tail_fraction=0.20,
        crest_fraction=0.03,
    ),
    ImpulseType.SWITCHING: ImpulseTarget(
        impulse_type=ImpulseType.SWITCHING,
        front_time_s=us_to_s(250.0),
        tail_time_s=us_to_s(2500.0),
        front_min_s=us_to_s(200.0),
        front_max_s=us_to_s(300.0),
        tail_fraction=0.20,
        crest_fraction=0.03,
    ),
}


class IntraStageTopology(str, Enum):
    """How the capacitors inside one Marx stage are connected.

    This matters and the workbook does not say. It records 2 capacitors per stage
    and a 3 uF stage capacitance, which is consistent with either two 1.5 uF units
    in parallel (each seeing the full stage voltage) or two 6 uF units in series
    (each seeing half). Those give different per-capacitor voltage stresses, so
    ``UNSPECIFIED`` makes per-capacitor checks return "unknown" rather than guess.
    """

    SERIES = "series"
    PARALLEL = "parallel"
    UNSPECIFIED = "unspecified"


def _require(
    name: str,
    value: float,
    unit: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
    allow_equal_min: bool = False,
) -> float:
    """Range-check one scalar, naming the offending quantity and its unit."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    too_small = value < minimum if allow_equal_min else value <= minimum
    if too_small:
        comparison = "at least" if allow_equal_min else "greater than"
        raise ValueError(
            f"{name} must be {comparison} {minimum} {unit}, got {value} {unit}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must be at most {maximum} {unit}, got {value} {unit}. "
            "This is a sanity limit on the input, not an equipment rating."
        )
    return float(value)


@dataclass(frozen=True)
class CapacitorRatings:
    """Per-capacitor limits. ``None`` means the datasheet value was not supplied.

    None of these were present in the supplied problem statement or workbook. They
    are carried so that :mod:`physics_model.capacitor_stress` can report a real
    check when a laboratory supplies them, and report "unknown" until then.
    """

    rated_voltage_v: float | None = None
    rated_energy_j: float | None = None
    rated_peak_current_a: float | None = None
    max_shots_per_hour: int | None = None
    veracity: Veracity = Veracity.UNKNOWN

    def __post_init__(self) -> None:
        if self.rated_voltage_v is not None:
            _require("rated_voltage_v", self.rated_voltage_v, "V", maximum=1e7)
        if self.rated_energy_j is not None:
            _require("rated_energy_j", self.rated_energy_j, "J", maximum=1e9)
        if self.rated_peak_current_a is not None:
            _require("rated_peak_current_a", self.rated_peak_current_a, "A", maximum=1e7)
        if self.max_shots_per_hour is not None:
            _require("max_shots_per_hour", self.max_shots_per_hour, "shots/h", maximum=1e5)

    @property
    def any_known(self) -> bool:
        return any(
            v is not None
            for v in (
                self.rated_voltage_v,
                self.rated_energy_j,
                self.rated_peak_current_a,
                self.max_shots_per_hour,
            )
        )

    @classmethod
    def from_lab_units(
        cls,
        *,
        rated_voltage_kv: float | None = None,
        rated_energy_j: float | None = None,
        rated_peak_current_a: float | None = None,
        max_shots_per_hour: int | None = None,
        veracity: Veracity = Veracity.UNKNOWN,
    ) -> "CapacitorRatings":
        return cls(
            rated_voltage_v=None if rated_voltage_kv is None else kv_to_v(rated_voltage_kv),
            rated_energy_j=rated_energy_j,
            rated_peak_current_a=rated_peak_current_a,
            max_shots_per_hour=max_shots_per_hour,
            veracity=veracity,
        )


@dataclass(frozen=True)
class GeneratorRatings:
    """Fixed properties of the generator. Workbook ``B13``-``B15``.

    ``max_stages`` (15), ``max_stage_voltage_v`` (200 kV) and
    ``stage_capacitance_f`` (3 uF) come from the problem statement and are tagged
    ``SPECIFIED``. ``capacitors_per_stage`` (2) is recorded there too but no
    workbook formula uses it.
    """

    max_stages: int = 15
    min_stages: int = 2
    max_stage_voltage_v: float = kv_to_v(200.0)
    stage_capacitance_f: float = uf_to_f(3.0)
    capacitors_per_stage: int = 2
    intra_stage_topology: IntraStageTopology = IntraStageTopology.UNSPECIFIED
    capacitor_ratings: CapacitorRatings = field(default_factory=CapacitorRatings)
    veracity: Veracity = Veracity.SPECIFIED

    def __post_init__(self) -> None:
        _require("max_stages", self.max_stages, "stages", maximum=100)
        _require("min_stages", self.min_stages, "stages", maximum=100)
        if self.min_stages > self.max_stages:
            raise ValueError(
                f"min_stages ({self.min_stages}) cannot exceed max_stages ({self.max_stages})"
            )
        _require("max_stage_voltage_v", self.max_stage_voltage_v, "V", maximum=1e7)
        _require("stage_capacitance_f", self.stage_capacitance_f, "F", maximum=1.0)
        _require("capacitors_per_stage", self.capacitors_per_stage, "capacitors", maximum=100)

    @property
    def capacitor_capacitance_f(self) -> float | None:
        """Capacitance of one physical capacitor, or ``None`` if undeterminable.

        Requires the intra-stage topology. With ``k`` capacitors per stage forming a
        stage capacitance ``C_stage``: parallel gives ``C_stage / k`` each, series
        gives ``C_stage * k`` each. Without knowing which, this returns ``None``.
        """
        k = self.capacitors_per_stage
        if self.intra_stage_topology is IntraStageTopology.PARALLEL:
            return self.stage_capacitance_f / k
        if self.intra_stage_topology is IntraStageTopology.SERIES:
            return self.stage_capacitance_f * k
        return None

    @property
    def max_output_voltage_v(self) -> float:
        """Ceiling before efficiency: every stage charged to its rating."""
        return self.max_stages * self.max_stage_voltage_v

    @classmethod
    def from_lab_units(
        cls,
        *,
        max_stages: int = 15,
        min_stages: int = 2,
        max_stage_voltage_kv: float = 200.0,
        stage_capacitance_uf: float = 3.0,
        capacitors_per_stage: int = 2,
        intra_stage_topology: IntraStageTopology = IntraStageTopology.UNSPECIFIED,
        capacitor_ratings: CapacitorRatings | None = None,
        veracity: Veracity = Veracity.SPECIFIED,
    ) -> "GeneratorRatings":
        return cls(
            max_stages=max_stages,
            min_stages=min_stages,
            max_stage_voltage_v=kv_to_v(max_stage_voltage_kv),
            stage_capacitance_f=uf_to_f(stage_capacitance_uf),
            capacitors_per_stage=capacitors_per_stage,
            intra_stage_topology=intra_stage_topology,
            capacitor_ratings=capacitor_ratings or CapacitorRatings(),
            veracity=veracity,
        )


@dataclass(frozen=True)
class ResistorStock:
    """One resistor value and how many of it can be put in series on one stage."""

    ohms: float
    units_per_stage: int

    def __post_init__(self) -> None:
        _require("ohms", self.ohms, "ohm", maximum=1e9)
        _require("units_per_stage", self.units_per_stage, "units", maximum=100)

    def series_values(self) -> list[float]:
        """Per-stage resistances obtainable by stacking 1..n of these in series."""
        return [self.ohms * k for k in range(1, self.units_per_stage + 1)]


@dataclass(frozen=True)
class ResistorInventory:
    """What the laboratory physically has, from the workbook's inventory sheet.

    Worth knowing: no formula in the workbook reads that sheet. The workbook picks
    resistors by rounding the theoretical value to the nearest 5 ohm
    (``E14``/``E17``) regardless of whether that value exists. This package uses the
    inventory for real, which is why it can reach different conclusions.
    """

    front: tuple[ResistorStock, ...]
    tail: tuple[ResistorStock, ...]
    external_front_ohms: tuple[float, ...] = ()
    external_tail_ohms: tuple[float, ...] = ()
    veracity: Veracity = Veracity.SPECIFIED

    def __post_init__(self) -> None:
        if not self.front:
            raise ValueError("front resistor inventory is empty; nothing can be built")
        if not self.tail:
            raise ValueError("tail resistor inventory is empty; nothing can be built")
        for side, values in (
            ("external_front_ohms", self.external_front_ohms),
            ("external_tail_ohms", self.external_tail_ohms),
        ):
            for v in values:
                _require(f"{side} entry", v, "ohm", maximum=1e9)

    @classmethod
    def from_pairs(
        cls,
        front: Iterable[tuple[float, int]],
        tail: Iterable[tuple[float, int]],
        *,
        external_front_ohms: Sequence[float] = (),
        external_tail_ohms: Sequence[float] = (),
        veracity: Veracity = Veracity.SPECIFIED,
    ) -> "ResistorInventory":
        return cls(
            front=tuple(ResistorStock(o, u) for o, u in front),
            tail=tuple(ResistorStock(o, u) for o, u in tail),
            external_front_ohms=tuple(external_front_ohms),
            external_tail_ohms=tuple(external_tail_ohms),
            veracity=veracity,
        )


@dataclass(frozen=True)
class TestCircuit:
    """The part of the circuit that changes with every test setup.

    Workbook ``B8``-``B12``. These are the workbook's single worked example, so they
    are ``ILLUSTRATIVE`` unless a laboratory measures its own.

    ``efficiency`` is the generator's voltage efficiency. The workbook takes it as an
    *input* (``B12`` = 0.82) rather than deriving it from the circuit, so the model
    cannot predict it; it is an assumption the operator supplies.
    """

    load_capacitance_f: float
    divider_capacitance_f: float
    stray_capacitance_f: float
    stray_inductance_h: float
    efficiency: float
    veracity: Veracity = Veracity.ILLUSTRATIVE

    def __post_init__(self) -> None:
        _require("load_capacitance_f", self.load_capacitance_f, "F", maximum=1e-3)
        _require("divider_capacitance_f", self.divider_capacitance_f, "F", maximum=1e-3)
        _require("stray_capacitance_f", self.stray_capacitance_f, "F", maximum=1e-3)
        _require(
            "stray_inductance_h",
            self.stray_inductance_h,
            "H",
            minimum=0.0,
            maximum=1.0,
            allow_equal_min=True,
        )
        _require("efficiency", self.efficiency, "fraction", maximum=1.0)

    @property
    def total_load_capacitance_f(self) -> float:
        """Workbook ``E7``: the three load capacitances in parallel."""
        return (
            self.load_capacitance_f
            + self.divider_capacitance_f
            + self.stray_capacitance_f
        )

    @classmethod
    def from_lab_units(
        cls,
        *,
        load_capacitance_pf: float,
        divider_capacitance_pf: float,
        stray_capacitance_pf: float,
        stray_inductance_uh: float,
        efficiency: float,
        veracity: Veracity = Veracity.ILLUSTRATIVE,
    ) -> "TestCircuit":
        return cls(
            load_capacitance_f=pf_to_f(load_capacitance_pf),
            divider_capacitance_f=pf_to_f(divider_capacitance_pf),
            stray_capacitance_f=pf_to_f(stray_capacitance_pf),
            stray_inductance_h=uh_to_h(stray_inductance_uh),
            efficiency=efficiency,
            veracity=veracity,
        )


@dataclass(frozen=True)
class GeneratorConfiguration:
    """A fully determined generator setting: what an engineer would physically build.

    ``front_resistance_per_stage_ohm`` and ``tail_resistance_per_stage_ohm`` are
    per-stage values, matching the workbook's ``E14``/``E17`` and the dataset's
    ``Front_R_Stage``/``Tail_R_Stage`` columns. Totals are ``n`` times those, which
    is what the waveform equations use.
    """

    stages: int
    charging_voltage_per_stage_v: float
    front_resistance_per_stage_ohm: float
    tail_resistance_per_stage_ohm: float
    impulse_type: ImpulseType
    circuit: TestCircuit
    ratings: GeneratorRatings

    def __post_init__(self) -> None:
        _require("stages", self.stages, "stages", maximum=100)
        _require(
            "charging_voltage_per_stage_v",
            self.charging_voltage_per_stage_v,
            "V",
            maximum=1e7,
        )
        _require(
            "front_resistance_per_stage_ohm",
            self.front_resistance_per_stage_ohm,
            "ohm",
            maximum=1e9,
        )
        _require(
            "tail_resistance_per_stage_ohm",
            self.tail_resistance_per_stage_ohm,
            "ohm",
            maximum=1e9,
        )

    @property
    def front_resistance_total_ohm(self) -> float:
        return self.front_resistance_per_stage_ohm * self.stages

    @property
    def tail_resistance_total_ohm(self) -> float:
        return self.tail_resistance_per_stage_ohm * self.stages

    @property
    def generator_capacitance_f(self) -> float:
        """Workbook ``E11``: the stage capacitors in series once the gaps fire."""
        return self.ratings.stage_capacitance_f / self.stages

    @property
    def target(self) -> ImpulseTarget:
        return IMPULSE_TARGETS[self.impulse_type]


# The workbook's one worked example, kept here so tests and demos have a single
# agreed starting point. Cells B8-B12 of 'Hybrid Calculator'.
WORKBOOK_EXAMPLE_CIRCUIT = TestCircuit.from_lab_units(
    load_capacitance_pf=850.0,
    divider_capacitance_pf=500.0,
    stray_capacitance_pf=150.0,
    stray_inductance_uh=18.5,
    efficiency=0.82,
)

WORKBOOK_EXAMPLE_RATINGS = GeneratorRatings.from_lab_units()

# Cells A4:B13 and E4:F15 of 'Resistor Inventory'.
WORKBOOK_EXAMPLE_INVENTORY = ResistorInventory.from_pairs(
    front=[
        (10.0, 4), (20.0, 4), (30.0, 4), (50.0, 4), (75.0, 4),
        (100.0, 4), (150.0, 4), (200.0, 4), (300.0, 4), (500.0, 4),
    ],
    tail=[
        (5.0, 4), (10.0, 4), (15.0, 4), (20.0, 4), (25.0, 4), (30.0, 4),
        (40.0, 4), (50.0, 4), (75.0, 4), (100.0, 4), (150.0, 4), (200.0, 4),
    ],
)
