"""Single source of truth for every engineering constant used by the application.

Unit conventions
----------------
The API and this configuration use laboratory-facing units, because that is what an
engineer reads off a datasheet or types into a form:

    voltage            kV
    capacitance        pF  (test-circuit side)  /  uF  (generator stage side)
    inductance         uH
    resistance         ohm
    time               us

Conversion to SI happens in exactly one place, ``backend.units``, and the physics
engine will work in SI only. Nothing in this file is converted.

Every constant below is wrapped in a provenance tag. If you add a number here and
cannot say where it came from, it belongs in ``TEST_OBJECT_CATALOGUE`` as a
placeholder, not in the generator or tolerance sections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend.provenance import (
    Assumption,
    AssumptionRegister,
    Provenance,
    Source,
    conflicting,
    from_dataset,
    from_problem_statement,
    placeholder,
    supplied,
    unverified,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

WORKBOOK = "Hybrid_Physics_ML_Impulse_Generator_Optimiser.xlsx"
PROBLEM_STATEMENT = "HV IG Problem Statement.pdf"


# ---------------------------------------------------------------------------
# Impulse types
# ---------------------------------------------------------------------------

LIGHTNING = "lightning"
SWITCHING = "switching"
IMPULSE_TYPES = (LIGHTNING, SWITCHING)


# ---------------------------------------------------------------------------
# Generator ratings and operating limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorLimits:
    """Fixed ratings of the impulse generator, as distinct from test-circuit values.

    The problem statement draws exactly this distinction: generator ratings are fixed
    hardware properties, while load, divider, stray capacitance and stray inductance
    change with every test setup.
    """

    max_stages: int
    min_active_stages: int
    max_stage_voltage_kv: float
    stage_capacitance_uf: float
    capacitors_per_stage: int
    min_test_voltage_kv: float
    max_test_voltage_kv: float

    @property
    def max_charge_voltage_kv(self) -> float:
        """Total charging voltage with every stage at its rated voltage."""
        return self.max_stages * self.max_stage_voltage_kv

    def generator_capacitance_uf(self, stages: int) -> float:
        """Series capacitance of ``stages`` stages, per workbook cell E11.

        Stage capacitors in series divide: C1 = C_stage / n.
        """
        if stages <= 0:
            raise ValueError("stages must be positive")
        return self.stage_capacitance_uf / stages


GENERATOR = GeneratorLimits(
    max_stages=15,
    min_active_stages=2,
    max_stage_voltage_kv=200.0,
    stage_capacitance_uf=3.0,
    capacitors_per_stage=2,
    min_test_voltage_kv=50.0,
    max_test_voltage_kv=2400.0,
)


# ---------------------------------------------------------------------------
# Waveform definitions and tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """An absolute window around a nominal waveform parameter."""

    nominal: float
    relative: float
    """Permitted fractional deviation, e.g. 0.30 for +/-30%."""

    unit: str
    provenance: Provenance

    @property
    def minimum(self) -> float:
        return self.nominal * (1.0 - self.relative)

    @property
    def maximum(self) -> float:
        return self.nominal * (1.0 + self.relative)

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum

    def to_dict(self) -> dict:
        return {
            "nominal": self.nominal,
            "relative": self.relative,
            "min": self.minimum,
            "max": self.maximum,
            "unit": self.unit,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class ImpulseSpec:
    """Target waveform and acceptance window for one impulse type."""

    impulse_type: str
    label: str
    designation: str
    front_time: Tolerance
    half_value_time: Tolerance
    peak_relative_tolerance: float
    peak_tolerance_provenance: Provenance
    front_time_definition: str
    front_time_definition_provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "impulse_type": self.impulse_type,
            "label": self.label,
            "designation": self.designation,
            "front_time": self.front_time.to_dict(),
            "half_value_time": self.half_value_time.to_dict(),
            "peak": {
                "relative": self.peak_relative_tolerance,
                "provenance": self.peak_tolerance_provenance.to_dict(),
            },
            "front_time_definition": {
                "text": self.front_time_definition,
                "provenance": self.front_time_definition_provenance.to_dict(),
            },
        }


# The workbook encodes its acceptance gate in cell H13 of the Hybrid Calculator:
#   lightning front  0.84 .. 1.56 us   -> 1.2 us +/-30%
#   switching front  200  .. 300  us   -> 250 us +/-20%
#   tail (both)      nominal +/-20%
#   crest (both)     within +/-3% of the requested test voltage
#
# The lightning numbers agree with the tolerances commonly cited for IEC 60060-1
# lightning impulses, but "commonly cited" is not the standard text, so they are
# tagged NEEDS_VERIFICATION rather than SUPPLIED. The switching half-value window is
# tagged CONFLICT: the standard is generally understood to allow a far wider band on
# the time to half-value for switching impulses than the +/-20% the workbook applies.

_LIGHTNING_FRONT_NOTE = (
    "Workbook gate H13 uses 0.84-1.56 us, i.e. 1.2 us +/-30%. This matches the "
    "tolerance usually quoted for IEC 60060-1 lightning impulse front time. Confirm "
    "against the standard text and record the edition before relying on it."
)

_HALF_VALUE_NOTE = (
    "Workbook gate H13 applies +/-20% to the time to half-value for both impulse "
    "types. For lightning this matches the usually quoted IEC 60060-1 figure."
)

_SWITCHING_TAIL_CONFLICT = (
    "Workbook gate H13 applies +/-20% to the switching time to half-value (2000-3000 "
    "us). IEC 60060-1 is generally understood to permit a substantially wider band for "
    "switching impulses, so this window is probably too strict and will reject "
    "configurations a real laboratory would accept. Do not present compliance results "
    "for switching impulses as standard-conformant until this is checked."
)

_PEAK_NOTE = (
    "Workbook gate H13 requires the predicted crest within +/-3% of the requested test "
    "voltage. This matches the tolerance usually quoted for the peak value in "
    "IEC 60060-1. Confirm against the standard text."
)

_LIGHTNING_FRONT_DEFINITION = (
    "Front time T1 = 1.67 x (t90 - t30), i.e. 1.67 times the interval between 30% and "
    "90% of the peak value on the rising edge. The workbook's physics formula (cell "
    "E18) applies this 1.67 factor explicitly."
)

_SWITCHING_FRONT_DEFINITION = (
    "Time to peak Tp is the actual interval from the virtual origin to the crest, NOT "
    "a 30-90% extrapolation. The workbook reuses the lightning 1.67 x formula in cell "
    "E18 for switching impulses as well, which is dimensionally inconsistent with the "
    "time-to-peak definition. The physics engine must treat the two definitions "
    "separately."
)

STANDARD_ID = "IEC_60060-1"
STANDARD_LABEL = "IEC 60060-1 - High-voltage test techniques, general definitions"
STANDARD_EDITION_NOTE = (
    "Edition not supplied with the project material. The applicable edition must be "
    "confirmed, since tolerance values and the definition of the front time have "
    "changed between editions."
)

IMPULSE_SPECS: dict[str, ImpulseSpec] = {
    LIGHTNING: ImpulseSpec(
        impulse_type=LIGHTNING,
        label="Lightning impulse",
        designation="1.2/50 us",
        front_time=Tolerance(
            nominal=1.2,
            relative=0.30,
            unit="us",
            provenance=unverified(
                Source.WORKBOOK, f"{WORKBOOK} / Hybrid Calculator!H13", _LIGHTNING_FRONT_NOTE
            ),
        ),
        half_value_time=Tolerance(
            nominal=50.0,
            relative=0.20,
            unit="us",
            provenance=unverified(
                Source.WORKBOOK, f"{WORKBOOK} / Hybrid Calculator!H13", _HALF_VALUE_NOTE
            ),
        ),
        peak_relative_tolerance=0.03,
        peak_tolerance_provenance=unverified(
            Source.WORKBOOK, f"{WORKBOOK} / Hybrid Calculator!H13", _PEAK_NOTE
        ),
        front_time_definition=_LIGHTNING_FRONT_DEFINITION,
        front_time_definition_provenance=supplied(
            f"{WORKBOOK} / Hybrid Calculator!E18",
            "The 1.67 scaling factor appears directly in the workbook formula.",
        ),
    ),
    SWITCHING: ImpulseSpec(
        impulse_type=SWITCHING,
        label="Switching impulse",
        designation="250/2500 us",
        front_time=Tolerance(
            nominal=250.0,
            relative=0.20,
            unit="us",
            provenance=unverified(
                Source.WORKBOOK,
                f"{WORKBOOK} / Hybrid Calculator!H13",
                "Workbook gate H13 uses 200-300 us, i.e. 250 us +/-20%. Confirm against "
                "the standard text.",
            ),
        ),
        half_value_time=Tolerance(
            nominal=2500.0,
            relative=0.20,
            unit="us",
            provenance=conflicting(
                Source.WORKBOOK,
                f"{WORKBOOK} / Hybrid Calculator!H13",
                _SWITCHING_TAIL_CONFLICT,
            ),
        ),
        peak_relative_tolerance=0.03,
        peak_tolerance_provenance=unverified(
            Source.WORKBOOK, f"{WORKBOOK} / Hybrid Calculator!H13", _PEAK_NOTE
        ),
        front_time_definition=_SWITCHING_FRONT_DEFINITION,
        front_time_definition_provenance=conflicting(
            Source.WORKBOOK,
            f"{WORKBOOK} / Hybrid Calculator!E18",
            "Workbook applies the lightning 30-90% extrapolation to switching impulses.",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Resistor inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResistorStock:
    ohms: float
    units_per_stage: int

    def to_dict(self) -> dict:
        return {"ohms": self.ohms, "units_per_stage": self.units_per_stage}


_INVENTORY_LOCATOR = f"{WORKBOOK} / Resistor Inventory"

FRONT_RESISTORS: tuple[ResistorStock, ...] = tuple(
    ResistorStock(ohms, 4)
    for ohms in (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0)
)

TAIL_RESISTORS: tuple[ResistorStock, ...] = tuple(
    ResistorStock(ohms, 4)
    for ohms in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0)
)


def max_series_resistance_per_stage(stock: tuple[ResistorStock, ...]) -> float:
    """Largest per-stage resistance reachable by putting stocked units in series."""
    return max(item.ohms * item.units_per_stage for item in stock)


# ---------------------------------------------------------------------------
# Test object catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestObjectType:
    """A selectable test object.

    The supplied material contains no per-object capacitance data whatsoever. The only
    figures available are the single worked example in the workbook (850 pF) and the
    600-1250 pF span the synthetic dataset sweeps. Every default below is therefore a
    placeholder chosen to make the form usable, and each one is reported as such.
    """

    id: str
    label: str
    description: str
    default_load_capacitance_pf: float
    typical_min_pf: float
    typical_max_pf: float
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "default_load_capacitance_pf": self.default_load_capacitance_pf,
            "typical_min_pf": self.typical_min_pf,
            "typical_max_pf": self.typical_max_pf,
            "provenance": self.provenance.to_dict(),
        }


def _placeholder_object(
    object_id: str,
    label: str,
    description: str,
    default_pf: float,
    low_pf: float,
    high_pf: float,
) -> TestObjectType:
    return TestObjectType(
        id=object_id,
        label=label,
        description=description,
        default_load_capacitance_pf=default_pf,
        typical_min_pf=low_pf,
        typical_max_pf=high_pf,
        provenance=placeholder(
            "not present in supplied material",
            "Placeholder capacitance with no measured basis. The supplied material "
            "gives only one worked value (850 pF, workbook cell B8) and the synthetic "
            "dataset's 600-1250 pF sweep. Replace with measured values from the test "
            "laboratory before this field is used for a real recommendation.",
        ),
    )


TEST_OBJECT_CATALOGUE: tuple[TestObjectType, ...] = (
    TestObjectType(
        id="workbook_example",
        label="Workbook worked example",
        description=(
            "The single test case present in the supplied workbook: 850 pF load, 500 pF "
            "divider, 150 pF stray, 18.5 uH stray inductance, 1425 kV lightning impulse."
        ),
        default_load_capacitance_pf=850.0,
        typical_min_pf=850.0,
        typical_max_pf=850.0,
        provenance=supplied(
            f"{WORKBOOK} / Hybrid Calculator!B8",
            "The only load capacitance stated anywhere in the supplied material.",
        ),
    ),
    _placeholder_object(
        "cap_and_pin_string",
        "400 kV cap-and-pin insulator string",
        "Suspension string of cap-and-pin discs.",
        700.0,
        600.0,
        900.0,
    ),
    _placeholder_object(
        "long_rod_composite",
        "400 kV composite long-rod insulator",
        "Polymeric long-rod insulator.",
        650.0,
        600.0,
        800.0,
    ),
    _placeholder_object(
        "station_post",
        "400 kV station post insulator",
        "Rigid post insulator for substation support.",
        900.0,
        700.0,
        1100.0,
    ),
    _placeholder_object(
        "bushing",
        "400 kV transformer bushing",
        "Condenser-graded bushing. Appreciably more capacitive than a plain insulator.",
        1100.0,
        900.0,
        1250.0,
    ),
    TestObjectType(
        id="custom",
        label="Other / enter values manually",
        description="No default applied. Enter the measured capacitance directly.",
        default_load_capacitance_pf=850.0,
        typical_min_pf=1.0,
        typical_max_pf=100_000.0,
        provenance=placeholder(
            "n/a",
            "No defaults are applied for this option; the engineer supplies all values.",
        ),
    ),
)

TEST_OBJECTS_BY_ID = {item.id: item for item in TEST_OBJECT_CATALOGUE}


# ---------------------------------------------------------------------------
# Default test-circuit values for the form
# ---------------------------------------------------------------------------

DEFAULT_DIVIDER_CAPACITANCE_PF = 500.0
DEFAULT_STRAY_CAPACITANCE_PF = 150.0
DEFAULT_STRAY_INDUCTANCE_UH = 18.5
DEFAULT_EFFICIENCY = 0.82
DEFAULT_TEST_VOLTAGE_KV = 1425.0


# ---------------------------------------------------------------------------
# ML training domain
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def ml_domain() -> dict:
    """Training-data ranges, regenerated by ``scripts/extract_ml_domain.py``."""
    path = DATA_DIR / "ml_domain.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/extract_ml_domain.py"
        )
    return json.loads(path.read_text())


def domain_for(impulse_type: str) -> dict:
    domain = ml_domain()["impulse_types"]
    if impulse_type not in domain:
        raise KeyError(f"no training data for impulse type {impulse_type!r}")
    return domain[impulse_type]


# ---------------------------------------------------------------------------
# Assumption register
# ---------------------------------------------------------------------------


def build_assumption_register() -> AssumptionRegister:
    register = AssumptionRegister()

    register.add(
        Assumption(
            key="generator.ratings",
            subject="Generator ratings",
            value=(
                f"{GENERATOR.max_stages} stages max, "
                f"{GENERATOR.max_stage_voltage_kv:g} kV/stage max, "
                f"{GENERATOR.stage_capacitance_uf:g} uF/stage"
            ),
            provenance=supplied(
                f"{WORKBOOK} / Hybrid Calculator!B13:B15",
                "Stated as user inputs in the workbook rather than quoted from a "
                "generator datasheet. No datasheet was supplied.",
            ),
            impact="Sets the hard ceiling on achievable test voltage.",
        )
    )
    register.add(
        Assumption(
            key="generator.test_voltage_range",
            subject="Supported test voltage range",
            value=(
                f"{GENERATOR.min_test_voltage_kv:g}-{GENERATOR.max_test_voltage_kv:g} kV, "
                f"minimum {GENERATOR.min_active_stages} active stages"
            ),
            provenance=from_problem_statement(
                f"{PROBLEM_STATEMENT}, Additional Guidance",
                "Stated as approximate in the problem statement.",
            ),
            impact="Defines which requests are accepted at all.",
        )
    )
    register.add(
        Assumption(
            key="generator.capacitors_per_stage",
            subject="Capacitors per stage",
            value=f"{GENERATOR.capacitors_per_stage} per stage",
            provenance=unverified(
                Source.PROBLEM_STATEMENT,
                "project discussion",
                "A 15-stage generator with two capacitors per stage was described in "
                "discussion. How the two capacitors combine into the 3 uF stage figure "
                "(series, parallel, or 3 uF each) is not stated and is not currently "
                "used by any calculation.",
            ),
            impact="None yet. Will matter if stored energy per stage is computed.",
        )
    )
    register.add(
        Assumption(
            key="efficiency",
            subject="Generator voltage efficiency",
            value=f"user input, default {DEFAULT_EFFICIENCY:g}",
            provenance=unverified(
                Source.WORKBOOK,
                f"{WORKBOOK} / Hybrid Calculator!B12",
                "Efficiency is an input to the workbook, not a calculated result. The "
                "workbook contains no model relating efficiency to the circuit, so the "
                "engineer's estimate propagates directly into the stage count and "
                "charging voltage.",
            ),
            impact="Directly scales required stages and charging voltage per stage.",
        )
    )
    register.add(
        Assumption(
            key="physics.crest_voltage",
            subject="Crest voltage model",
            value="none - predicted crest equals the requested test voltage",
            provenance=conflicting(
                Source.WORKBOOK,
                f"{WORKBOOK} / Hybrid Calculator!E20 and Synthetic Dataset!P",
                "Physics_Crest_kV equals Test_kV exactly in all 2000 dataset rows, and "
                "the workbook cell is literally '=B7'. There is no physics model for "
                "crest voltage, so the ML 'crest residual' is synthetic scatter around "
                "a tautology and its reported accuracy is not meaningful.",
            ),
            impact=(
                "Predicted crest voltage carries no engineering information until a real "
                "voltage-efficiency model is implemented."
            ),
        )
    )
    register.add(
        Assumption(
            key="standard.edition",
            subject="Applicable standard edition",
            value=f"{STANDARD_LABEL} - edition unknown",
            provenance=unverified(
                Source.STANDARD, STANDARD_ID, STANDARD_EDITION_NOTE
            ),
            impact="All tolerance windows and the front-time definition depend on it.",
        )
    )
    for spec in IMPULSE_SPECS.values():
        for name, prov, value in (
            ("front_time", spec.front_time.provenance, spec.front_time.to_dict()),
            (
                "half_value_time",
                spec.half_value_time.provenance,
                spec.half_value_time.to_dict(),
            ),
            (
                "peak",
                spec.peak_tolerance_provenance,
                {"relative": spec.peak_relative_tolerance},
            ),
            (
                "front_time_definition",
                spec.front_time_definition_provenance,
                {"text": spec.front_time_definition},
            ),
        ):
            if prov.is_trustworthy:
                continue
            if name == "front_time_definition":
                rendered = spec.front_time_definition
            elif name == "peak":
                rendered = f"+/-{spec.peak_relative_tolerance * 100:g}%"
            else:
                rendered = (
                    f"{value['min']:g}-{value['max']:g} {value['unit']} "
                    f"({value['nominal']:g} +/-{value['relative'] * 100:g}%)"
                )
            register.add(
                Assumption(
                    key=f"tolerance.{spec.impulse_type}.{name}",
                    subject=f"{spec.label} - {name.replace('_', ' ')}",
                    value=rendered,
                    provenance=prov,
                    impact="Determines whether a candidate is reported as compliant.",
                )
            )

    register.add(
        Assumption(
            key="inventory.values",
            subject="Resistor inventory",
            value=(
                f"front {min(r.ohms for r in FRONT_RESISTORS):g}-"
                f"{max(r.ohms for r in FRONT_RESISTORS):g} ohm, "
                f"tail {min(r.ohms for r in TAIL_RESISTORS):g}-"
                f"{max(r.ohms for r in TAIL_RESISTORS):g} ohm, 4 units/stage"
            ),
            provenance=supplied(
                _INVENTORY_LOCATOR,
                "Listed on its own sheet, but no workbook formula references that "
                "sheet. The calculator instead rounds the theoretical resistance to "
                "the nearest 5 ohm (cells E14/E17), producing values such as 35 or 55 "
                "ohm that are not stocked.",
            ),
            impact="Determines which candidate configurations are physically buildable.",
        )
    )
    register.add(
        Assumption(
            key="inventory.switching_shortfall",
            subject="Switching impulse resistance shortfall",
            value=(
                f"inventory reaches {max_series_resistance_per_stage(FRONT_RESISTORS):g} "
                f"ohm/stage front and "
                f"{max_series_resistance_per_stage(TAIL_RESISTORS):g} ohm/stage tail"
            ),
            provenance=conflicting(
                Source.DATASET,
                "Synthetic Dataset!L:M, switching rows",
                "Switching rows in the dataset use 9370-21810 ohm/stage front and "
                "~1200 ohm/stage tail, far above what the stocked inventory can reach "
                "even with all four units in series. The problem statement mentions "
                "external front/tail resistors, which is the likely resolution, but no "
                "external resistor values were supplied.",
            ),
            impact=(
                "Switching impulses cannot be realised from stocked parts alone. The "
                "engineer must enter external resistor values."
            ),
        )
    )
    register.add(
        Assumption(
            key="ml.training_domain",
            subject="ML training domain",
            value=_render_domain_summary(),
            provenance=from_dataset(
                "Synthetic Dataset, Train split",
                "1400 training rows, 700 per impulse type.",
            ),
            impact=(
                "Inputs outside these ranges make the residual correction an "
                "extrapolation from unrelated neighbours."
            ),
        )
    )
    register.add(
        Assumption(
            key="ml.validation_status",
            subject="Model validation status",
            value="synthetic only - no laboratory measurement",
            provenance=supplied(
                f"{WORKBOOK} / Model Card!B11 and Model Validation!A16",
                "The workbook states that no laboratory validation has been completed "
                "and that measured waveforms are pending.",
            ),
            impact="Predicted compliance is not evidence that a physical test will pass.",
        )
    )
    register.add(
        Assumption(
            key="test_objects.capacitance",
            subject="Test object capacitance defaults",
            value="placeholder values for all catalogue entries except the workbook example",
            provenance=placeholder(
                "not present in supplied material",
                "No per-test-object capacitance data exists in the supplied material.",
            ),
            impact=(
                "Selecting a test object type pre-fills a capacitance with no measured "
                "basis. Always override with a measured value."
            ),
        )
    )
    return register


def _render_domain_summary() -> str:
    parts = []
    for impulse_type in IMPULSE_TYPES:
        try:
            block = domain_for(impulse_type)
        except (FileNotFoundError, KeyError):
            continue
        voltage = block["inputs"]["Test_kV"]
        parts.append(
            f"{impulse_type} {voltage['min']:.0f}-{voltage['max']:.0f} kV "
            f"({block['training_records']} rows)"
        )
    return "; ".join(parts) if parts else "unavailable"


ASSUMPTIONS = build_assumption_register()
