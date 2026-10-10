"""Request and response contracts.

These are the interfaces the five workstreams agree on. Field names carry their unit
as a suffix so that no caller has to guess, and every numeric field is plain ``float``
with no pydantic range constraint. Range checking lives in ``backend.validation`` so
that the engineer gets an explanatory engineering message instead of a schema error.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend import config

ImpulseType = Literal["lightning", "switching"]
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


class TestRequestIn(BaseModel):
    """Everything the engineer specifies for one optimisation run."""

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
