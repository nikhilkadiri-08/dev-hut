"""Energy and stress accounting for the generator's capacitors.

The governing rule in this module: a check whose input data was never supplied
reports ``UNKNOWN``. It does not report ``PASS``. An absent datasheet is not
evidence of safety, and a prototype that renders missing ratings as green ticks is
worse than one that renders nothing.

On inferring individual capacitor voltage
-----------------------------------------
The per-stage DC charging voltage *is* determinable: the stages are charged in
parallel from the DC supply, so each sits at workbook ``E9``. That is a circuit
fact, not a division of the output voltage.

Going from stage voltage to *individual capacitor* voltage is not determinable from
the supplied material. The problem statement records 2 capacitors per stage and a
3 uF stage capacitance, which is equally consistent with:

* two 1.5 uF units in parallel, each seeing the full stage voltage, or
* two 6 uF units in series, each seeing half of it.

Those differ by a factor of two in voltage stress. So unless the laboratory states
:class:`~physics_model.inputs.IntraStageTopology`, every per-capacitor result here
is ``UNKNOWN``. Dividing the total output voltage by the number of capacitors would
produce a plausible number with no circuit justification, which is exactly the
failure this module is built to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from physics_model.calculations import Caveat, Severity, stored_energy
from physics_model.inputs import (
    GeneratorConfiguration,
    IntraStageTopology,
)

__all__ = [
    "CheckStatus",
    "StressCheck",
    "CapacitorStressReport",
    "individual_capacitor_voltage",
    "assess_capacitor_stress",
]


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StressCheck:
    """One limit comparison, with the margin when there is a limit to compare to."""

    name: str
    status: CheckStatus
    reason: str
    actual: float | None = None
    limit: float | None = None
    unit: str = ""
    applicable: bool = True
    """False if this limit does not apply to the present request (e.g. no shot rate)."""

    @property
    def margin_fraction(self) -> float | None:
        """Headroom as a fraction of the limit. ``None`` when either side is absent.

        Positive means the limit is not reached; -0.1 means 10% over.
        """
        if self.actual is None or self.limit is None or self.limit == 0.0:
            return None
        return (self.limit - self.actual) / self.limit

    def __str__(self) -> str:
        if self.actual is None or self.limit is None:
            return f"{self.name}: {self.status.value.upper()} - {self.reason}"
        margin = self.margin_fraction
        return (
            f"{self.name}: {self.status.value.upper()} "
            f"({self.actual:.4g} of {self.limit:.4g} {self.unit}, "
            f"margin {margin:+.1%})"
        )


@dataclass
class CapacitorStressReport:
    """Stress accounting for one configuration."""

    stage_voltage_v: float
    generator_output_voltage_v: float
    total_stored_energy_j: float
    energy_per_stage_j: float
    capacitor_voltage_v: float | None
    capacitor_capacitance_f: float | None
    energy_per_capacitor_j: float | None
    peak_current_upper_bound_a: float | None
    checks: list[StressCheck] = field(default_factory=list)
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def failed(self) -> list[StressCheck]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def unknown(self) -> list[StressCheck]:
        return [c for c in self.checks if c.status is CheckStatus.UNKNOWN]

    @property
    def all_clear(self) -> bool:
        """True only if every check actively passed.

        Deliberately strict: one ``UNKNOWN`` is enough to make this False, because
        an unevaluated limit is not a satisfied limit.
        """
        return bool(self.checks) and all(
            c.status is CheckStatus.PASS for c in self.checks if c.applicable
        )


def individual_capacitor_voltage(
    configuration: GeneratorConfiguration,
) -> tuple[float | None, str]:
    """Voltage across one physical capacitor, or ``None`` with the reason why not.

    Returns ``(volts, justification)``. ``volts`` is ``None`` when the intra-stage
    topology is unspecified, in which case the second element explains what is
    missing rather than offering a guess.
    """
    stage_v = configuration.charging_voltage_per_stage_v
    ratings = configuration.ratings
    k = ratings.capacitors_per_stage

    if k == 1:
        return stage_v, (
            "One capacitor per stage, so it carries the full stage charging "
            "voltage (workbook E9). No topology assumption needed."
        )
    if ratings.intra_stage_topology is IntraStageTopology.PARALLEL:
        return stage_v, (
            f"{k} capacitors declared in parallel within the stage, so each carries "
            f"the full stage charging voltage (workbook E9)."
        )
    if ratings.intra_stage_topology is IntraStageTopology.SERIES:
        return stage_v / k, (
            f"{k} capacitors declared in series within the stage, so each carries "
            f"1/{k} of the stage charging voltage. This assumes matched capacitors "
            f"and effective voltage grading; an ungraded series string divides "
            f"unequally and the worst-case unit sees more than this."
        )
    return None, (
        f"Intra-stage topology is unspecified. With {k} capacitors per stage and a "
        f"{ratings.stage_capacitance_f * 1e6:.3g} uF stage capacitance, both a "
        f"parallel arrangement (each unit at the full stage voltage) and a series "
        f"arrangement (each at half) fit the supplied data, and they differ by a "
        f"factor of {k}. Declare the topology to enable per-capacitor checks."
    )


def assess_capacitor_stress(
    configuration: GeneratorConfiguration,
    *,
    shots_per_hour: float | None = None,
) -> CapacitorStressReport:
    """Compute stored energy and run every stress check the data supports.

    Args:
        configuration: the generator setting to assess.
        shots_per_hour: intended repetition rate, if the operator has one in mind.
            Without it the repetitive-operation check reports ``UNKNOWN``.
    """
    ratings = configuration.ratings
    cap_ratings = ratings.capacitor_ratings
    stage_v = configuration.charging_voltage_per_stage_v
    stages = configuration.stages

    # Charge side: every stage capacitor sits at the stage voltage.
    energy_per_stage = stored_energy(ratings.stage_capacitance_f, stage_v)
    total_energy = stages * energy_per_stage
    output_v = stages * stage_v

    cap_v, cap_v_reason = individual_capacitor_voltage(configuration)
    cap_c = ratings.capacitor_capacitance_f
    energy_per_capacitor = (
        stored_energy(cap_c, cap_v) if (cap_c is not None and cap_v is not None) else None
    )

    checks: list[StressCheck] = []
    caveats: list[Caveat] = []

    # --- stage voltage against the stage rating (this one we can always do) ---
    checks.append(
        StressCheck(
            name="Stage charging voltage",
            status=(
                CheckStatus.PASS
                if stage_v <= ratings.max_stage_voltage_v
                else CheckStatus.FAIL
            ),
            reason=(
                "Compared against the 'maximum stage voltage' rating from the "
                "problem statement (workbook B14)."
            ),
            actual=stage_v,
            limit=ratings.max_stage_voltage_v,
            unit="V",
        )
    )

    # --- per-capacitor voltage ---
    if cap_v is None:
        checks.append(
            StressCheck(
                name="Individual capacitor voltage",
                status=CheckStatus.UNKNOWN,
                reason=cap_v_reason,
            )
        )
        caveats.append(
            Caveat("CAPACITOR_TOPOLOGY_UNSPECIFIED", cap_v_reason, Severity.WARNING)
        )
    elif cap_ratings.rated_voltage_v is None:
        checks.append(
            StressCheck(
                name="Individual capacitor voltage",
                status=CheckStatus.UNKNOWN,
                reason=(
                    f"Capacitor voltage is determinable ({cap_v:.4g} V: "
                    f"{cap_v_reason}) but no capacitor voltage rating was supplied, "
                    f"so there is nothing to compare it against."
                ),
                actual=cap_v,
                unit="V",
            )
        )
    else:
        checks.append(
            StressCheck(
                name="Individual capacitor voltage",
                status=(
                    CheckStatus.PASS
                    if cap_v <= cap_ratings.rated_voltage_v
                    else CheckStatus.FAIL
                ),
                reason=cap_v_reason,
                actual=cap_v,
                limit=cap_ratings.rated_voltage_v,
                unit="V",
            )
        )

    # --- per-capacitor energy ---
    if energy_per_capacitor is None:
        checks.append(
            StressCheck(
                name="Individual capacitor energy",
                status=CheckStatus.UNKNOWN,
                reason=(
                    "Needs both the individual capacitor voltage and its "
                    "capacitance, and at least one is undetermined. "
                    + cap_v_reason
                ),
            )
        )
    elif cap_ratings.rated_energy_j is None:
        checks.append(
            StressCheck(
                name="Individual capacitor energy",
                status=CheckStatus.UNKNOWN,
                reason=(
                    f"Energy per capacitor is {energy_per_capacitor:.4g} J by "
                    f"0.5*C*V^2, but no capacitor energy rating was supplied."
                ),
                actual=energy_per_capacitor,
                unit="J",
            )
        )
    else:
        checks.append(
            StressCheck(
                name="Individual capacitor energy",
                status=(
                    CheckStatus.PASS
                    if energy_per_capacitor <= cap_ratings.rated_energy_j
                    else CheckStatus.FAIL
                ),
                reason="0.5*C*V^2 against the supplied capacitor energy rating.",
                actual=energy_per_capacitor,
                limit=cap_ratings.rated_energy_j,
                unit="J",
            )
        )

    # --- peak discharge current ---
    # At the instant the gaps fire, the generator capacitance is charged to the
    # full series voltage and the load capacitance is not. Ignoring inductance,
    # the initial current is set by the resistances alone. Stray inductance can
    # only slow the rise, so this is an upper bound rather than an estimate.
    rf = configuration.front_resistance_total_ohm
    rt = configuration.tail_resistance_total_ohm
    peak_current = output_v * (1.0 / rf + 1.0 / rt) if rf > 0 and rt > 0 else None
    if peak_current is None:
        checks.append(
            StressCheck(
                name="Peak discharge current",
                status=CheckStatus.UNKNOWN,
                reason="Front or tail resistance is zero, so no bound can be formed.",
            )
        )
    elif cap_ratings.rated_peak_current_a is None:
        checks.append(
            StressCheck(
                name="Peak discharge current",
                status=CheckStatus.UNKNOWN,
                reason=(
                    f"Upper bound is {peak_current:.4g} A, from the series voltage "
                    f"across the front and tail branches with inductance neglected. "
                    f"No capacitor current rating was supplied to compare against."
                ),
                actual=peak_current,
                unit="A",
            )
        )
    else:
        checks.append(
            StressCheck(
                name="Peak discharge current",
                status=(
                    CheckStatus.PASS
                    if peak_current <= cap_ratings.rated_peak_current_a
                    else CheckStatus.FAIL
                ),
                reason=(
                    "Upper bound with inductance neglected, against the supplied "
                    "peak current rating. Conservative by construction."
                ),
                actual=peak_current,
                limit=cap_ratings.rated_peak_current_a,
                unit="A",
            )
        )

    # --- repetitive operation ---
    if shots_per_hour is None:
        checks.append(
            StressCheck(
                name="Repetitive operation",
                status=CheckStatus.UNKNOWN,
                reason="No shot rate was given, so a repetition-limit check does not apply.",
                applicable=False,
            )
        )
    elif cap_ratings.max_shots_per_hour is None:
        checks.append(
            StressCheck(
                name="Repetitive operation",
                status=CheckStatus.UNKNOWN,
                reason=(
                    f"Intended rate is {shots_per_hour:g} shots/h, but no capacitor "
                    "repetition rating was supplied."
                ),
                actual=shots_per_hour,
                unit="shots/h",
            )
        )
    else:
        checks.append(
            StressCheck(
                name="Repetitive operation",
                status=(
                    CheckStatus.PASS
                    if shots_per_hour <= cap_ratings.max_shots_per_hour
                    else CheckStatus.FAIL
                ),
                reason="Intended shot rate against the supplied repetition rating.",
                actual=shots_per_hour,
                limit=float(cap_ratings.max_shots_per_hour),
                unit="shots/h",
            )
        )

    resistor_limit = ratings.resistor_energy_rating_j
    if resistor_limit is None:
        checks.append(
            StressCheck(
                name="Resistor energy duty",
                status=CheckStatus.UNKNOWN,
                reason=(
                    f"The front and tail resistors absorb essentially all of the "
                    f"{total_energy / 1000.0:.4g} kJ stored per shot, but no resistor "
                    f"energy rating was supplied. Unevaluated, so this configuration "
                    f"cannot be presented as safe."
                ),
                actual=total_energy,
                unit="J",
            )
        )
    else:
        checks.append(
            StressCheck(
                name="Resistor energy duty",
                status=(
                    CheckStatus.PASS
                    if total_energy <= resistor_limit
                    else CheckStatus.FAIL
                ),
                reason="Stored energy per shot against the supplied resistor energy rating.",
                actual=total_energy,
                limit=resistor_limit,
                unit="J",
            )
        )

    if not cap_ratings.any_known:
        caveats.append(
            Caveat(
                "NO_CAPACITOR_RATINGS_SUPPLIED",
                "No capacitor voltage, energy, current or repetition ratings were "
                "supplied with the project material, so the component-level stress "
                "checks are unevaluated rather than satisfied. A qualified engineer "
                "must confirm these against the actual capacitor datasheets.",
                Severity.WARNING,
            )
        )

    # Workbook modelling C is 3 uF. At the 200 kV stage rating that is 60 kJ.
    # A 2.5 kJ (~0.125 uF at 200 kV) figure is not present as a sourced file here.
    rated_energy = stored_energy(ratings.stage_capacitance_f, ratings.max_stage_voltage_v)
    caveats.append(
        Caveat(
            "MODELLING_C_ENERGY_UNVERIFIED",
            f"Energy is computed from the workbook modelling capacitance "
            f"({ratings.stage_capacitance_f * 1e6:.4g} uF). At the "
            f"{ratings.max_stage_voltage_v / 1000.0:.4g} kV stage rating that is "
            f"{rated_energy / 1000.0:.4g} kJ/stage. A 2.5 kJ rating "
            f"(≈0.125 uF at 200 kV) is not sourced from any file in this "
            f"repository. The two cannot describe the same capacitor. This "
            f"package keeps the workbook 3 uF for E11 and does not treat either "
            f"energy as verified.",
            Severity.WARNING,
        )
    )

    return CapacitorStressReport(
        stage_voltage_v=stage_v,
        generator_output_voltage_v=output_v,
        total_stored_energy_j=total_energy,
        energy_per_stage_j=energy_per_stage,
        capacitor_voltage_v=cap_v,
        capacitor_capacitance_f=cap_c,
        energy_per_capacitor_j=energy_per_capacitor,
        peak_current_upper_bound_a=peak_current,
        checks=checks,
        caveats=caveats,
    )
