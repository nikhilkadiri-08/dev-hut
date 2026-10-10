"""Configurable impulse-test sequences. Not a hardcoded IEC procedure.

A sequence is an ordered list of shots the operator (or a cited procedure) asked
for. This module does not invent a universal IEC sequence. When a source, edition
or clause is not in the repository the citation is ``unresolved`` and the
sequence remains user-configured.

Voltage *levels* are fractions of a reference test voltage (0.5, 1.0, …). That
is not the tail 50% crossing used to define T2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from backend.units import v_to_kv
from physics_model.calculations import Caveat, PhysicsPrediction, Severity, predict
from physics_model.inputs import (
    GeneratorConfiguration,
    ImpulseType,
)
from physics_model.targets import timing_definition
from physics_model.validation import ValidationReport, validate_configuration
from physics_model.waveform import (
    ChoppedLightning,
    build_chopped_lightning,
    fit_double_exponential,
)

__all__ = [
    "WaveformKind",
    "CitationStatus",
    "ConstraintStatus",
    "StandardCitation",
    "ShotSpec",
    "TestSequence",
    "ShotEvaluation",
    "SequenceEvaluation",
    "impulse_type_for",
    "evaluate_shot",
    "evaluate_sequence",
]


class WaveformKind(str, Enum):
    LIGHTNING_FULL = "lightning_full"
    SWITCHING = "switching"
    CHOPPED_LIGHTNING = "chopped_lightning"


class CitationStatus(str, Enum):
    KNOWN = "known"
    UNRESOLVED = "unresolved"


class ConstraintStatus(str, Enum):
    SATISFIED = "satisfied"
    FAILED = "failed"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class StandardCitation:
    """Where a sequence came from, if we actually have the text."""

    source: str | None = None
    edition: str | None = None
    clause: str | None = None
    status: CitationStatus = CitationStatus.UNRESOLVED
    note: str = (
        "No IEC edition or clause text is stored in this repository. "
        "This sequence is user-configured."
    )

    def __post_init__(self) -> None:
        known = self.source and self.edition and self.clause
        if self.status is CitationStatus.KNOWN and not known:
            raise ValueError(
                "A 'known' citation needs source, edition and clause. "
                "Otherwise leave status unresolved."
            )


@dataclass(frozen=True)
class ShotSpec:
    """One configured shot. Missing chop parameters are errors, not defaults."""

    waveform_kind: WaveformKind
    level_fraction: float
    chopping_time_s: float | None = None
    collapse_duration_s: float | None = None
    collapse_shape: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.level_fraction <= 0.0:
            raise ValueError(
                f"level_fraction must be positive, got {self.level_fraction}"
            )
        if self.waveform_kind is WaveformKind.CHOPPED_LIGHTNING:
            if self.chopping_time_s is None or self.collapse_duration_s is None:
                raise ValueError(
                    "Chopped lightning requires chopping_time_s and "
                    "collapse_duration_s. They are not defaulted."
                )
            if self.collapse_shape is None:
                raise ValueError(
                    "Chopped lightning requires collapse_shape "
                    "('linear' or 'exponential'). It is not defaulted."
                )


@dataclass(frozen=True)
class TestSequence:
    """An ordered, configuration-driven list of shots."""

    reference_voltage_v: float
    shots: tuple[ShotSpec, ...]
    name: str = "user_configured"
    citation: StandardCitation = field(default_factory=StandardCitation)

    def __post_init__(self) -> None:
        if self.reference_voltage_v <= 0.0:
            raise ValueError(
                f"reference_voltage_v must be positive, got {self.reference_voltage_v}"
            )
        if not self.shots:
            raise ValueError("a sequence must contain at least one shot")

    def requested_voltage_v(self, spec: ShotSpec) -> float:
        return self.reference_voltage_v * spec.level_fraction


def impulse_type_for(kind: WaveformKind) -> ImpulseType:
    """Workbook impulse type used for E18/E19. Chopped uses the lightning scalars."""
    if kind is WaveformKind.SWITCHING:
        return ImpulseType.SWITCHING
    return ImpulseType.LIGHTNING


def _constraint_status(report: ValidationReport) -> ConstraintStatus:
    if not report.buildable or report.quality_verdict.value != "compliant":
        return ConstraintStatus.FAILED
    if not report.confirmed_safe:
        return ConstraintStatus.UNVERIFIED
    return ConstraintStatus.SATISFIED


@dataclass
class ShotEvaluation:
    spec: ShotSpec
    requested_voltage_v: float
    prediction: PhysicsPrediction
    report: ValidationReport
    constraint_status: ConstraintStatus
    tail_half_crossing_v: float
    chopped: ChoppedLightning | None = None
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def requested_voltage_kv(self) -> float:
        return v_to_kv(self.requested_voltage_v)

    @property
    def predicted_crest_kv(self) -> float:
        return self.prediction.crest_voltage_kv

    @property
    def tail_half_crossing_kv(self) -> float:
        return v_to_kv(self.tail_half_crossing_v)


@dataclass
class SequenceEvaluation:
    sequence: TestSequence
    shots: list[ShotEvaluation]

    @property
    def all_confirmed_safe(self) -> bool:
        return all(shot.report.confirmed_safe for shot in self.shots)

    @property
    def any_failed(self) -> bool:
        return any(shot.constraint_status is ConstraintStatus.FAILED for shot in self.shots)


def evaluate_shot(
    spec: ShotSpec,
    configuration: GeneratorConfiguration,
    requested_voltage_v: float,
    *,
    inventory=None,
    shots_per_hour: float | None = None,
) -> ShotEvaluation:
    """Predict one shot. Charging voltage is not silently scaled to the request."""
    kind_type = impulse_type_for(spec.waveform_kind)
    if configuration.impulse_type is not kind_type:
        configuration = GeneratorConfiguration(
            stages=configuration.stages,
            charging_voltage_per_stage_v=configuration.charging_voltage_per_stage_v,
            front_resistance_per_stage_ohm=configuration.front_resistance_per_stage_ohm,
            tail_resistance_per_stage_ohm=configuration.tail_resistance_per_stage_ohm,
            impulse_type=kind_type,
            circuit=configuration.circuit,
            ratings=configuration.ratings,
        )

    prediction = predict(configuration, requested_voltage_v)
    report = validate_configuration(
        configuration, prediction, inventory=inventory, shots_per_hour=shots_per_hour
    )
    caveats = list(prediction.caveats)
    timing = timing_definition(kind_type)
    caveats.append(
        Caveat(
            "LEVEL_IS_NOT_TAIL_HALF",
            f"This shot's requested level is {spec.level_fraction:.4g} of the "
            "reference test voltage. That is not the 50% tail crossing used for T2.",
            Severity.NOTICE,
        )
    )
    if spec.waveform_kind is WaveformKind.SWITCHING:
        caveats.append(
            Caveat(
                "SWITCHING_E18_IS_NOT_TP",
                "Workbook E18 still uses the lightning 1.67 construction for "
                "switching. The waveform extractor uses time-to-peak Tp. "
                + timing.front_or_peak,
                Severity.WARNING,
            )
        )

    chopped = None
    if spec.waveform_kind is WaveformKind.CHOPPED_LIGHTNING:
        fitted = fit_double_exponential(
            prediction.front_time_s,
            prediction.tail_time_s,
            prediction.crest_voltage_v,
            ImpulseType.LIGHTNING,
        )
        chopped = build_chopped_lightning(
            fitted.shape,
            chopping_time_s=spec.chopping_time_s,  # type: ignore[arg-type]
            collapse_duration_s=spec.collapse_duration_s,  # type: ignore[arg-type]
            collapse_shape=spec.collapse_shape,  # type: ignore[arg-type]
        )
        caveats.extend(chopped.caveats)

    # Tail 50% crossing of *this* predicted crest — not the requested 50% level.
    tail_half_v = 0.5 * prediction.crest_voltage_v

    return ShotEvaluation(
        spec=spec,
        requested_voltage_v=requested_voltage_v,
        prediction=prediction,
        report=report,
        constraint_status=_constraint_status(report),
        tail_half_crossing_v=tail_half_v,
        chopped=chopped,
        caveats=caveats,
    )


def evaluate_sequence(
    sequence: TestSequence,
    configuration: GeneratorConfiguration,
    *,
    inventory=None,
    shots_per_hour: float | None = None,
) -> SequenceEvaluation:
    """Evaluate every shot in order against one physical configuration."""
    shots = [
        evaluate_shot(
            spec,
            configuration,
            sequence.requested_voltage_v(spec),
            inventory=inventory,
            shots_per_hour=shots_per_hour,
        )
        for spec in sequence.shots
    ]
    return SequenceEvaluation(sequence=sequence, shots=shots)


