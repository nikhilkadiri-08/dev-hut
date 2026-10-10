"""Configuration and component-limit validation.

Two kinds of constraint live here and they are deliberately not mixed:

**Hard safety constraints** are things that must not be violated: charging a stage
above its rating, using more stages than exist, specifying a resistance the
laboratory cannot physically build. These are never traded against waveform
quality, and a violation is fatal to a candidate rather than a penalty on it.

**Waveform-quality objectives** are the tolerance windows on front time, tail time
and peak voltage. Missing one of these means the shot would not be a valid test, not
that anything would be damaged. These are the quantities the optimiser ranks on.

The third state
---------------
A check whose data was never supplied is ``INDETERMINATE``, not safe. The verdict
type has three values for that reason: the supplied material contains no capacitor
ratings at all, so the honest answer for a real configuration today is "cannot
confirm", and a two-valued verdict would be forced to lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from backend.units import s_to_us, v_to_kv
from physics_model.calculations import (
    Caveat,
    PhysicsPrediction,
    Severity,
    front_time_floor,
    front_time_is_reachable,
)
from physics_model.capacitor_stress import (
    CapacitorStressReport,
    CheckStatus,
    assess_capacitor_stress,
)
from physics_model.inputs import (
    GeneratorConfiguration,
    ImpulseTarget,
    ResistorInventory,
    ResistorStock,
    Veracity,
)

__all__ = [
    "Verdict",
    "SafetyConstraint",
    "QualityObjective",
    "ValidationReport",
    "achievable_per_stage_values",
    "is_realisable",
    "validate_configuration",
]


class Verdict(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    INDETERMINATE = "indeterminate"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"


@dataclass(frozen=True)
class SafetyConstraint:
    """A hard limit. ``satisfied is None`` means it could not be evaluated."""

    code: str
    description: str
    satisfied: bool | None
    detail: str

    @property
    def violated(self) -> bool:
        return self.satisfied is False

    @property
    def indeterminate(self) -> bool:
        return self.satisfied is None


@dataclass(frozen=True)
class QualityObjective:
    """A tolerance window, with how centred the prediction sits inside it.

    ``margin`` is 1.0 at the exact centre of the window and 0.0 at an edge;
    negative values are outside. Centring matters operationally: a configuration
    sitting at the edge of the window will fall out of it under a small model error
    and send somebody back up the generator to change resistors, whereas a centred
    one absorbs the same error and still passes.

    ``vacuous`` marks an objective that cannot fail by construction, so passing it
    means nothing.
    """

    code: str
    description: str
    value: float
    minimum: float
    maximum: float
    unit: str
    vacuous: bool = False

    @property
    def satisfied(self) -> bool:
        return self.minimum <= self.value <= self.maximum

    @property
    def margin(self) -> float:
        centre = 0.5 * (self.minimum + self.maximum)
        half_width = 0.5 * (self.maximum - self.minimum)
        if half_width <= 0.0:
            return 0.0
        return 1.0 - abs(self.value - centre) / half_width


@dataclass
class ValidationReport:
    """The full verdict on one configuration."""

    configuration: GeneratorConfiguration
    safety: list[SafetyConstraint] = field(default_factory=list)
    quality: list[QualityObjective] = field(default_factory=list)
    stress: CapacitorStressReport | None = None
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def safety_verdict(self) -> Verdict:
        if any(c.violated for c in self.safety):
            return Verdict.UNSAFE
        if any(c.indeterminate for c in self.safety):
            return Verdict.INDETERMINATE
        return Verdict.SAFE

    @property
    def quality_verdict(self) -> Verdict:
        return (
            Verdict.COMPLIANT
            if all(o.satisfied for o in self.quality)
            else Verdict.NON_COMPLIANT
        )

    @property
    def buildable(self) -> bool:
        """Whether no hard constraint is actually violated.

        Distinct from ``safety_verdict is SAFE``: a configuration with unevaluated
        capacitor ratings is buildable but not confirmed safe. The optimiser keeps
        buildable candidates and reports the unevaluated checks alongside them.
        """
        return not any(c.violated for c in self.safety)

    @property
    def worst_margin(self) -> float:
        """Smallest margin across the non-vacuous objectives.

        This is the quantity to maximise if the goal is to pass on the first shot.
        Vacuous objectives are excluded so the crest tautology cannot inflate it.
        """
        real = [o.margin for o in self.quality if not o.vacuous]
        return min(real) if real else 0.0

    def violations(self) -> list[SafetyConstraint]:
        return [c for c in self.safety if c.violated]

    def unevaluated(self) -> list[SafetyConstraint]:
        return [c for c in self.safety if c.indeterminate]

    def summary(self) -> str:
        return (
            f"safety={self.safety_verdict.value}, "
            f"quality={self.quality_verdict.value}, "
            f"worst margin={self.worst_margin:+.3f}"
        )


def achievable_per_stage_values(stock: tuple[ResistorStock, ...]) -> list[float]:
    """Every per-stage resistance obtainable from one side of the inventory.

    Each stocked value can be stacked 1..``units_per_stage`` deep in series, so the
    reachable set is ``{k * r}``. Returned sorted and de-duplicated.

    Cost: ``O(m log m)`` in the number of combinations, computed once per
    inventory rather than once per candidate.
    """
    values = {
        round(value, 9)
        for item in stock
        for value in item.series_values()
    }
    return sorted(values)


def is_realisable(
    ohms_per_stage: float, stock: tuple[ResistorStock, ...], *, tolerance: float = 1e-6
) -> bool:
    """Whether the laboratory can actually build this per-stage resistance."""
    return any(
        abs(ohms_per_stage - candidate) <= tolerance
        for candidate in achievable_per_stage_values(stock)
    )


def _safety_constraints(
    configuration: GeneratorConfiguration,
    prediction: PhysicsPrediction,
    inventory: ResistorInventory | None,
    stress: CapacitorStressReport,
) -> list[SafetyConstraint]:
    ratings = configuration.ratings
    circuit = configuration.circuit
    stages = configuration.stages
    stage_v = configuration.charging_voltage_per_stage_v
    constraints: list[SafetyConstraint] = []

    # Workbook H12, first half.
    constraints.append(
        SafetyConstraint(
            code="STAGE_COUNT_WITHIN_RATING",
            description="Stages used must not exceed the stages the generator has",
            satisfied=stages <= ratings.max_stages,
            detail=f"{stages} of {ratings.max_stages} stages",
        )
    )
    constraints.append(
        SafetyConstraint(
            code="STAGE_COUNT_AT_LEAST_MINIMUM",
            description="Stage count must not fall below the stated minimum",
            satisfied=stages >= ratings.min_stages,
            detail=f"{stages} against a minimum of {ratings.min_stages}",
        )
    )

    # Workbook H12, second half. Over-charging a stage damages it; this is the one
    # constraint in the whole model that is unambiguously a safety limit.
    constraints.append(
        SafetyConstraint(
            code="STAGE_VOLTAGE_WITHIN_RATING",
            description="Per-stage charging voltage must not exceed the stage rating",
            satisfied=stage_v <= ratings.max_stage_voltage_v,
            detail=(
                f"{v_to_kv(stage_v):.4g} kV of "
                f"{v_to_kv(ratings.max_stage_voltage_v):.4g} kV per stage "
                f"({prediction.stage_utilisation:.1%} utilisation)"
            ),
        )
    )

    # Reachability. The workbook clamps this away; see front_time_floor.
    reachable = front_time_is_reachable(
        configuration.target.front_time_s,
        prediction.total_load_capacitance_f,
        circuit.stray_inductance_h,
    )
    floor_us = s_to_us(
        front_time_floor(
            prediction.total_load_capacitance_f, circuit.stray_inductance_h
        )
    )
    constraints.append(
        SafetyConstraint(
            code="FRONT_TIME_PHYSICALLY_REACHABLE",
            description=(
                "The target front time must be above the floor set by stray "
                "inductance and load capacitance alone"
            ),
            satisfied=reachable,
            detail=(
                f"target {s_to_us(configuration.target.front_time_s):.4g} us "
                f"against a floor of {floor_us:.4g} us"
            ),
        )
    )

    if inventory is not None:
        front_ok = is_realisable(
            configuration.front_resistance_per_stage_ohm, inventory.front
        )
        tail_ok = is_realisable(
            configuration.tail_resistance_per_stage_ohm, inventory.tail
        )
        front_reach = max(achievable_per_stage_values(inventory.front))
        tail_reach = max(achievable_per_stage_values(inventory.tail))
        constraints.append(
            SafetyConstraint(
                code="FRONT_RESISTOR_IN_STOCK",
                description=(
                    "The front resistance must be buildable from stocked resistors"
                ),
                satisfied=front_ok,
                detail=(
                    f"{configuration.front_resistance_per_stage_ohm:.4g} ohm/stage; "
                    f"stock reaches {front_reach:.4g} ohm/stage in series"
                ),
            )
        )
        constraints.append(
            SafetyConstraint(
                code="TAIL_RESISTOR_IN_STOCK",
                description=(
                    "The tail resistance must be buildable from stocked resistors"
                ),
                satisfied=tail_ok,
                detail=(
                    f"{configuration.tail_resistance_per_stage_ohm:.4g} ohm/stage; "
                    f"stock reaches {tail_reach:.4g} ohm/stage in series"
                ),
            )
        )

    # Component-level stress. FAIL is a violation; UNKNOWN stays indeterminate so
    # that a missing datasheet can never read as a pass.
    for check in stress.checks:
        if check.name == "Stage charging voltage":
            continue  # already covered above, from the same rating
        constraints.append(
            SafetyConstraint(
                code="STRESS_" + check.name.upper().replace(" ", "_"),
                description=check.name,
                satisfied=(
                    True
                    if check.status is CheckStatus.PASS
                    else False
                    if check.status is CheckStatus.FAIL
                    else None
                ),
                detail=check.reason,
            )
        )

    return constraints


def _quality_objectives(
    prediction: PhysicsPrediction, target: ImpulseTarget
) -> list[QualityObjective]:
    crest_min, crest_max = target.crest_window_v(prediction.requested_voltage_v)
    return [
        QualityObjective(
            code="FRONT_TIME_IN_WINDOW",
            description="Front or peak time within the acceptance window",
            value=prediction.front_time_us,
            minimum=s_to_us(target.front_min_s),
            maximum=s_to_us(target.front_max_s),
            unit="us",
        ),
        QualityObjective(
            code="TAIL_TIME_IN_WINDOW",
            description="Time to half value within the acceptance window",
            value=prediction.tail_time_us,
            minimum=s_to_us(target.tail_min_s),
            maximum=s_to_us(target.tail_max_s),
            unit="us",
        ),
        QualityObjective(
            code="CREST_VOLTAGE_IN_WINDOW",
            description="Peak voltage within tolerance of the requested voltage",
            value=prediction.crest_voltage_kv,
            minimum=v_to_kv(crest_min),
            maximum=v_to_kv(crest_max),
            unit="kV",
            # Workbook E20 returns the requested voltage, so the predicted crest
            # equals the window centre exactly and this can never fail. Passing it
            # carries no information until a voltage-efficiency model exists.
            vacuous=True,
        ),
    ]


def validate_configuration(
    configuration: GeneratorConfiguration,
    prediction: PhysicsPrediction,
    *,
    inventory: ResistorInventory | None = None,
    shots_per_hour: float | None = None,
) -> ValidationReport:
    """Validate one configuration against hard limits and quality windows.

    Args:
        configuration: the setting to check.
        prediction: the physics result for it, from
            :func:`physics_model.calculations.predict`.
        inventory: if given, the front and tail resistances are checked for being
            physically buildable. Without it that check is skipped entirely rather
            than assumed to pass.
        shots_per_hour: intended repetition rate for the stress assessment.
    """
    stress = assess_capacitor_stress(configuration, shots_per_hour=shots_per_hour)
    safety = _safety_constraints(configuration, prediction, inventory, stress)
    quality = _quality_objectives(prediction, configuration.target)

    caveats = list(prediction.caveats) + list(stress.caveats)
    caveats.append(
        Caveat(
            "PREDICTED_COMPLIANCE_IS_NOT_COMPLIANCE",
            "Satisfying these windows means the model expects a compliant shot. It "
            "is not a statement that the generator will produce one, and it does "
            "not substitute for a verification test witnessed by a qualified "
            "engineer.",
            Severity.NOTICE,
        )
    )
    if configuration.target.tolerance_veracity is Veracity.UNKNOWN:
        caveats.append(
            Caveat(
                "TOLERANCE_EDITION_UNCONFIRMED",
                "The acceptance windows were taken from the workbook, which did not "
                "record which edition of IEC 60060-1 it followed. Confirm the "
                "clause text before relying on the pass/fail boundary.",
                Severity.WARNING,
            )
        )

    return ValidationReport(
        configuration=configuration,
        safety=safety,
        quality=quality,
        stress=stress,
        caveats=caveats,
    )
