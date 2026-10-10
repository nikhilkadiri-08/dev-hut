"""Request and response contracts.

These are the interfaces the five workstreams agree on. Field names carry their unit
as a suffix so that no caller has to guess, and every numeric field is plain ``float``
with no pydantic range constraint. Range checking lives in ``backend.validation`` so
that the engineer gets an explanatory engineering message instead of a schema error.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend import config

ImpulseType = Literal["lightning", "switching"]
WaveformKindIn = Literal["lightning_full", "switching", "chopped_lightning"]
Polarity = Literal["positive", "negative"]
Severity = Literal["error", "warning", "notice"]


class ResistorStockIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ohms: float
    units_per_stage: int = 1


class TestCircuitIn(BaseModel):
    """Test-circuit parameters. These change with every test setup.

    The problem statement is explicit that these are variable and
    configuration-dependent, in contrast to the fixed generator ratings.
    """

    model_config = ConfigDict(extra="forbid")

    load_capacitance_pf: float = Field(
        description="Capacitance of the test object itself."
    )
    divider_capacitance_pf: float = Field(
        default=config.DEFAULT_DIVIDER_CAPACITANCE_PF,
        description="Capacitance of the measuring system / voltage divider.",
    )
    stray_capacitance_pf: float = Field(
        default=config.DEFAULT_STRAY_CAPACITANCE_PF,
        description="Estimated stray capacitance of leads, layout and earth return.",
    )
    stray_inductance_uh: float = Field(
        default=config.DEFAULT_STRAY_INDUCTANCE_UH,
        description="Estimated stray/connection inductance of the discharge loop.",
    )
    expected_efficiency: float = Field(
        default=config.DEFAULT_EFFICIENCY,
        description=(
            "Expected generator voltage efficiency as a fraction. This is an engineer's "
            "estimate, not a calculated quantity; the supplied material contains no "
            "model for it."
        ),
    )

    @property
    def total_load_capacitance_pf(self) -> float:
        """Load seen by the generator, per workbook cell E7 (B8 + B9 + B10)."""
        return (
            self.load_capacitance_pf
            + self.divider_capacitance_pf
            + self.stray_capacitance_pf
        )


class GeneratorConfigIn(BaseModel):
    """Generator ratings and stage constraints, overridable per request."""

    model_config = ConfigDict(extra="forbid")

    max_stages: int = config.GENERATOR.max_stages
    min_active_stages: int = config.GENERATOR.min_active_stages
    max_stage_voltage_kv: float = config.GENERATOR.max_stage_voltage_kv
    stage_capacitance_uf: float = config.GENERATOR.stage_capacitance_uf


class InventoryIn(BaseModel):
    """Available resistors.

    ``external_*_ohms`` exists because the stocked inventory cannot reach the
    resistances a switching impulse requires. The problem statement lists external
    front/tail resistors among the generator parameters, but supplies no values, so the
    engineer must enter the ones actually available in the laboratory.
    """

    model_config = ConfigDict(extra="forbid")

    front: list[ResistorStockIn] = Field(
        default_factory=lambda: [
            ResistorStockIn(ohms=item.ohms, units_per_stage=item.units_per_stage)
            for item in config.FRONT_RESISTORS
        ]
    )
    tail: list[ResistorStockIn] = Field(
        default_factory=lambda: [
            ResistorStockIn(ohms=item.ohms, units_per_stage=item.units_per_stage)
            for item in config.TAIL_RESISTORS
        ]
    )
    external_front_ohms: list[float] = Field(default_factory=list)
    external_tail_ohms: list[float] = Field(default_factory=list)


class GeneratorProfile(BaseModel):
    """The laboratory's standing generator setup, owned by the admin interface.

    These are properties of the equipment and the resistor store, not of any one test.
    They are entered once by whoever maintains the generator, persisted, and then
    reused by every test request so the operator never retypes them.
    """

    model_config = ConfigDict(extra="forbid")

    max_stages: int = config.GENERATOR.max_stages
    min_active_stages: int = config.GENERATOR.min_active_stages
    max_stage_voltage_kv: float = config.GENERATOR.max_stage_voltage_kv
    stage_capacitance_uf: float = config.GENERATOR.stage_capacitance_uf
    capacitors_per_stage: int = config.GENERATOR.capacitors_per_stage
    front_resistors: list[ResistorStockIn] = Field(
        default_factory=lambda: [
            ResistorStockIn(ohms=item.ohms, units_per_stage=item.units_per_stage)
            for item in config.FRONT_RESISTORS
        ]
    )
    tail_resistors: list[ResistorStockIn] = Field(
        default_factory=lambda: [
            ResistorStockIn(ohms=item.ohms, units_per_stage=item.units_per_stage)
            for item in config.TAIL_RESISTORS
        ]
    )
    external_front_ohms: list[float] = Field(default_factory=list)
    external_tail_ohms: list[float] = Field(default_factory=list)
    default_stray_inductance_uh: float = config.DEFAULT_STRAY_INDUCTANCE_UH
    default_efficiency: float = config.DEFAULT_EFFICIENCY


class StoredProfile(BaseModel):
    """A generator profile plus the provenance of the saved record itself."""

    profile: GeneratorProfile
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None
    is_factory_default: bool = True
    """True while no admin has saved anything and the supplied-material values apply."""


class ProfileUpdateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: GeneratorProfile
    updated_by: str = Field(
        default="", description="Who made the change. Recorded for traceability only."
    )


class OperatorRequestIn(BaseModel):
    """What the test engineer specifies for one shot.

    Deliberately omits every generator and inventory field: those come from the stored
    admin profile. ``stray_inductance_uh`` and ``expected_efficiency`` are optional
    overrides that fall back to the laboratory defaults the admin set.
    """

    model_config = ConfigDict(extra="forbid")

    test_object_type_id: str = "workbook_example"
    test_object_label: Optional[str] = None
    standard_id: str = config.STANDARD_ID
    impulse_type: ImpulseType = "lightning"
    polarity: Polarity = "positive"
    required_peak_kv: float = config.DEFAULT_TEST_VOLTAGE_KV
    load_capacitance_pf: float
    divider_capacitance_pf: float = config.DEFAULT_DIVIDER_CAPACITANCE_PF
    stray_capacitance_pf: float = config.DEFAULT_STRAY_CAPACITANCE_PF
    stray_inductance_uh: Optional[float] = None
    expected_efficiency: Optional[float] = None
    sequence: Optional["TestSequenceIn"] = None


class TestRequestIn(BaseModel):
    """Everything the engineer specifies for one optimisation run.

    The fully resolved form, after an operator request has been merged with the stored
    generator profile. Downstream modules work with this.
    """

    model_config = ConfigDict(extra="forbid")

    test_object_type_id: str = Field(
        default="workbook_example",
        description="Identifier from the test object catalogue.",
    )
    test_object_label: Optional[str] = Field(
        default=None, description="Free-text description, e.g. a serial number."
    )
    standard_id: str = config.STANDARD_ID
    impulse_type: ImpulseType = "lightning"
    polarity: Polarity = Field(
        default="positive",
        description=(
            "Recorded for traceability only. No supplied calculation or dataset column "
            "depends on polarity."
        ),
    )
    required_peak_kv: float = config.DEFAULT_TEST_VOLTAGE_KV
    circuit: TestCircuitIn
    generator: GeneratorConfigIn = Field(default_factory=GeneratorConfigIn)
    inventory: InventoryIn = Field(default_factory=InventoryIn)
    sequence: Optional["TestSequenceIn"] = None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class IssueOut(BaseModel):
    severity: Severity
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


class DomainCheckOut(BaseModel):
    """Whether one input lies inside the range the residual model was fitted on."""

    field: str
    label: str
    value: float
    training_min: float
    training_max: float
    in_domain: bool
    unit: str


class AcceptanceWindowOut(BaseModel):
    nominal: float
    min: float
    max: float
    unit: str
    provenance_status: str
    provenance_note: str


class DerivedQuantitiesOut(BaseModel):
    """Feasibility arithmetic reproduced from the workbook.

    These are capacity figures, not waveform predictions. Nothing here says anything
    about the shape of the impulse; that is the physics engine's job.
    """

    total_load_capacitance_pf: float
    stages_required: int
    charging_voltage_per_stage_kv: float
    stage_utilisation: float
    generator_capacitance_uf: float
    max_achievable_peak_kv: float
    workbook_reference: str = "Hybrid Calculator!E7:E11"


class NormalizedRequestOut(BaseModel):
    test_object_id: str
    test_object_label: Optional[str]
    standard_id: str
    standard_label: str
    impulse_type: ImpulseType
    impulse_designation: str
    polarity: Polarity
    required_peak_kv: float
    target_front_time_us: float
    target_half_value_time_us: float
    front_time_definition: str
    front_time_window: AcceptanceWindowOut
    half_value_time_window: AcceptanceWindowOut
    peak_relative_tolerance: float
    derived: DerivedQuantitiesOut


class ValidationResponse(BaseModel):
    ok: bool
    """True when there are no blocking errors. Warnings do not block."""

    errors: list[IssueOut] = Field(default_factory=list)
    warnings: list[IssueOut] = Field(default_factory=list)
    notices: list[IssueOut] = Field(default_factory=list)
    normalized: Optional[NormalizedRequestOut] = None
    domain_checks: list[DomainCheckOut] = Field(default_factory=list)
    relevant_assumptions: list[dict] = Field(default_factory=list)


class ProfileReachOut(BaseModel):
    """What the saved generator setup can physically do.

    Shown in the admin interface so the effect of a change is immediately visible,
    and summarised read-only in the operator interface.
    """

    max_front_ohms_per_stage: float
    max_tail_ohms_per_stage: float
    max_charge_voltage_kv: float
    max_achievable_peak_kv: float
    front_values: int
    tail_values: int
    has_external_front: bool
    has_external_tail: bool


class ProfileResponse(BaseModel):
    stored: StoredProfile
    ok: bool
    errors: list[IssueOut] = Field(default_factory=list)
    warnings: list[IssueOut] = Field(default_factory=list)
    notices: list[IssueOut] = Field(default_factory=list)
    reach: Optional[ProfileReachOut] = None


class PredictRatingsIn(BaseModel):
    """Generator ratings for a one-shot physics prediction.

    All fields are required when this object is present. Omitting the whole
    object means the stored laboratory profile is used instead — that is not
    a silent numeric default.
    """

    model_config = ConfigDict(extra="forbid")

    max_stages: int
    min_active_stages: int
    max_stage_voltage_kv: float
    stage_capacitance_uf: float
    capacitors_per_stage: int


class PhysicsPredictIn(BaseModel):
    """One fully specified configuration for ``POST /api/physics/predict``.

    Every numeric field that the physics functions need is required. There are
    no laboratory-unit defaults on this model.
    """

    model_config = ConfigDict(extra="forbid")

    impulse_type: ImpulseType
    requested_peak_kv: float
    stages: int
    charging_voltage_per_stage_kv: float
    front_resistance_per_stage_ohm: float
    tail_resistance_per_stage_ohm: float
    load_capacitance_pf: float
    divider_capacitance_pf: float
    stray_capacitance_pf: float
    stray_inductance_uh: float
    expected_efficiency: float
    ratings: Optional[PredictRatingsIn] = None
    waveform_kind: Optional[WaveformKindIn] = None
    chopping_time_us: Optional[float] = None
    collapse_duration_us: Optional[float] = None
    collapse_shape: Optional[Literal["linear", "exponential"]] = None
    sequence: Optional["TestSequenceIn"] = None

    @field_validator(
        "requested_peak_kv",
        "charging_voltage_per_stage_kv",
        "front_resistance_per_stage_ohm",
        "tail_resistance_per_stage_ohm",
        "load_capacitance_pf",
        "divider_capacitance_pf",
        "stray_capacitance_pf",
        "stray_inductance_uh",
        "expected_efficiency",
    )
    @classmethod
    def _finite_float(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Enter a number.")
        if not math.isfinite(float(value)):
            raise ValueError("Enter a finite number.")
        return float(value)


class ShotSpecIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    waveform_kind: WaveformKindIn
    level_fraction: float
    chopping_time_us: Optional[float] = None
    collapse_duration_us: Optional[float] = None
    collapse_shape: Optional[Literal["linear", "exponential"]] = None
    label: Optional[str] = None


class TestSequenceIn(BaseModel):
    """User-configured shot list. Not a stored IEC procedure."""

    model_config = ConfigDict(extra="forbid")

    reference_peak_kv: float
    shots: list[ShotSpecIn]
    name: Optional[str] = None
    source: Optional[str] = None
    edition: Optional[str] = None
    clause: Optional[str] = None


class ApiErrorBody(BaseModel):
    code: str
    message: str
    details: list[IssueOut] = Field(default_factory=list)


class PhysicsApiMetadata(BaseModel):
    physics_model_used: bool = True
    ml_model_used: bool = False


def merge_operator_request(
    operator: OperatorRequestIn, profile: GeneratorProfile
) -> TestRequestIn:
    """Combine a test request with the laboratory's stored generator setup.

    The two optional operator fields fall back to the admin's laboratory defaults, so
    an operator who leaves them alone inherits whatever the admin configured.
    """
    return TestRequestIn(
        test_object_type_id=operator.test_object_type_id,
        test_object_label=operator.test_object_label,
        standard_id=operator.standard_id,
        impulse_type=operator.impulse_type,
        polarity=operator.polarity,
        required_peak_kv=operator.required_peak_kv,
        circuit=TestCircuitIn(
            load_capacitance_pf=operator.load_capacitance_pf,
            divider_capacitance_pf=operator.divider_capacitance_pf,
            stray_capacitance_pf=operator.stray_capacitance_pf,
            stray_inductance_uh=(
                operator.stray_inductance_uh
                if operator.stray_inductance_uh is not None
                else profile.default_stray_inductance_uh
            ),
            expected_efficiency=(
                operator.expected_efficiency
                if operator.expected_efficiency is not None
                else profile.default_efficiency
            ),
        ),
        generator=GeneratorConfigIn(
            max_stages=profile.max_stages,
            min_active_stages=profile.min_active_stages,
            max_stage_voltage_kv=profile.max_stage_voltage_kv,
            stage_capacitance_uf=profile.stage_capacitance_uf,
        ),
        inventory=InventoryIn(
            front=list(profile.front_resistors),
            tail=list(profile.tail_resistors),
            external_front_ohms=list(profile.external_front_ohms),
            external_tail_ohms=list(profile.external_tail_ohms),
        ),
        sequence=operator.sequence,
    )
