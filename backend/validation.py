"""Input validation and normalisation.

Three kinds of finding are reported, and the distinction matters:

``error``    The request cannot be processed. Something is missing, physically
             impossible, or outside a verified generator limit.
``warning``  The request will be processed but the result is less trustworthy than it
             looks, most often because an input sits outside the ML training domain or
             relies on a value the supplied material never verified.
``notice``   Context the engineer should read once, such as how a quantity was derived.

Nothing in this module predicts a waveform. The only arithmetic performed is the
capacity calculation reproduced from the supplied workbook (cells E7 to E11), which is
needed to answer "can this generator reach the requested voltage at all".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from backend import config
from backend.schemas import (
    AcceptanceWindowOut,
    DerivedQuantitiesOut,
    DomainCheckOut,
    GeneratorProfile,
    IssueOut,
    NormalizedRequestOut,
    ProfileReachOut,
    ResistorStockIn,
    TestRequestIn,
    ValidationResponse,
)

# Mapping from request field to the dataset column that bounds it, used for the
# out-of-domain check. Keys are dotted paths into the request.
DOMAIN_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("required_peak_kv", "Test_kV", "Required peak voltage", "kV"),
    ("circuit.load_capacitance_pf", "Load_C_pF", "Test object capacitance", "pF"),
    ("circuit.divider_capacitance_pf", "Divider_C_pF", "Divider capacitance", "pF"),
    ("circuit.stray_capacitance_pf", "Stray_C_pF", "Stray capacitance", "pF"),
    ("circuit.stray_inductance_uh", "L_uH", "Stray inductance", "uH"),
    ("circuit.expected_efficiency", "Efficiency", "Expected efficiency", ""),
)


@dataclass
class _Findings:
    errors: list[IssueOut] = field(default_factory=list)
    warnings: list[IssueOut] = field(default_factory=list)
    notices: list[IssueOut] = field(default_factory=list)

    def error(self, code: str, message: str, *, field_: str | None = None, suggestion: str | None = None) -> None:
        self.errors.append(
            IssueOut(severity="error", code=code, message=message, field=field_, suggestion=suggestion)
        )

    def warn(self, code: str, message: str, *, field_: str | None = None, suggestion: str | None = None) -> None:
        self.warnings.append(
            IssueOut(severity="warning", code=code, message=message, field=field_, suggestion=suggestion)
        )

    def notice(self, code: str, message: str, *, field_: str | None = None, suggestion: str | None = None) -> None:
        self.notices.append(
            IssueOut(severity="notice", code=code, message=message, field=field_, suggestion=suggestion)
        )


def _resolve(request: TestRequestIn, dotted: str) -> float:
    target: object = request
    for part in dotted.split("."):
        target = getattr(target, part)
    return float(target)  # type: ignore[arg-type]


def validate_request(request: TestRequestIn) -> ValidationResponse:
    """Check one test request and normalise it for the downstream pipeline."""
    findings = _Findings()

    _check_catalogue(request, findings)
    _check_voltage(request, findings)
    _check_circuit(request, findings)
    _check_generator(request, findings)
    _check_inventory(request, findings)

    domain_checks = _check_training_domain(request, findings)
    _add_standing_caveats(request, findings)

    ok = not findings.errors
    normalized = _normalise(request) if ok else None

    return ValidationResponse(
        ok=ok,
        errors=findings.errors,
        warnings=findings.warnings,
        notices=findings.notices,
        normalized=normalized,
        domain_checks=domain_checks,
        relevant_assumptions=_relevant_assumptions(request),
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_catalogue(request: TestRequestIn, findings: _Findings) -> None:
    if request.test_object_type_id not in config.TEST_OBJECTS_BY_ID:
        available = ", ".join(sorted(config.TEST_OBJECTS_BY_ID))
        findings.error(
            "UNKNOWN_TEST_OBJECT",
            f"Test object type {request.test_object_type_id!r} is not in the catalogue.",
            field_="test_object_type_id",
            suggestion=f"Choose one of: {available}.",
        )
    if request.standard_id != config.STANDARD_ID:
        findings.error(
            "UNKNOWN_STANDARD",
            f"Standard {request.standard_id!r} is not configured. Only "
            f"{config.STANDARD_ID} tolerance windows are implemented.",
            field_="standard_id",
            suggestion=f"Use {config.STANDARD_ID!r}.",
        )


def _check_voltage(request: TestRequestIn, findings: _Findings) -> None:
    voltage = request.required_peak_kv
    limits = config.GENERATOR

    if not math.isfinite(voltage):
        findings.error(
            "PEAK_VOLTAGE_NOT_A_NUMBER",
            "Required peak voltage must be a finite number.",
            field_="required_peak_kv",
        )
        return
    if voltage <= 0:
        findings.error(
            "PEAK_VOLTAGE_NOT_POSITIVE",
            f"Required peak voltage must be greater than zero, received {voltage:g} kV.",
            field_="required_peak_kv",
        )
        return
    if voltage < limits.min_test_voltage_kv:
        findings.error(
            "PEAK_VOLTAGE_BELOW_RANGE",
            f"Required peak voltage {voltage:g} kV is below the supported range. The "
            f"generator is specified for approximately "
            f"{limits.min_test_voltage_kv:g}-{limits.max_test_voltage_kv:g} kV.",
            field_="required_peak_kv",
            suggestion=f"Enter at least {limits.min_test_voltage_kv:g} kV.",
        )
    if voltage > limits.max_test_voltage_kv:
        findings.error(
            "PEAK_VOLTAGE_ABOVE_RANGE",
            f"Required peak voltage {voltage:g} kV exceeds the supported range of "
            f"approximately {limits.max_test_voltage_kv:g} kV stated in the problem "
            f"statement.",
            field_="required_peak_kv",
            suggestion=f"Reduce the test level to {limits.max_test_voltage_kv:g} kV or below.",
        )


def _check_circuit(request: TestRequestIn, findings: _Findings) -> None:
    circuit = request.circuit

    for name, value in (
        ("load_capacitance_pf", circuit.load_capacitance_pf),
        ("divider_capacitance_pf", circuit.divider_capacitance_pf),
        ("stray_capacitance_pf", circuit.stray_capacitance_pf),
        ("stray_inductance_uh", circuit.stray_inductance_uh),
    ):
        if not math.isfinite(value):
            findings.error(
                "CIRCUIT_VALUE_NOT_A_NUMBER",
                f"{name} must be a finite number.",
                field_=f"circuit.{name}",
            )
        elif value < 0:
            findings.error(
                "CIRCUIT_VALUE_NEGATIVE",
                f"{name} cannot be negative, received {value:g}.",
                field_=f"circuit.{name}",
            )

    if math.isfinite(circuit.load_capacitance_pf) and circuit.load_capacitance_pf <= 0:
        findings.error(
            "LOAD_CAPACITANCE_NOT_POSITIVE",
            "Test object capacitance must be greater than zero. The impulse shape is "
            "set by the resistors working against this capacitance, so a zero load has "
            "no physical meaning here.",
            field_="circuit.load_capacitance_pf",
        )

    if circuit.divider_capacitance_pf == 0:
        findings.warn(
            "DIVIDER_CAPACITANCE_ZERO",
            "Divider capacitance is zero. A real measuring system always contributes "
            "some capacitance, and ignoring it will make the predicted front time "
            "shorter than the measured one.",
            field_="circuit.divider_capacitance_pf",
            suggestion=f"Typical entry for this setup is "
            f"{config.DEFAULT_DIVIDER_CAPACITANCE_PF:g} pF.",
        )

    if circuit.stray_inductance_uh == 0:
        findings.warn(
            "STRAY_INDUCTANCE_ZERO",
            "Stray inductance is zero. The front-time calculation includes an "
            "inductive term, so setting it to zero removes the effect of lead length "
            "and layout on the rising edge.",
            field_="circuit.stray_inductance_uh",
        )

    efficiency = circuit.expected_efficiency
    if not math.isfinite(efficiency):
        findings.error(
            "EFFICIENCY_NOT_A_NUMBER",
            "Expected efficiency must be a finite number.",
            field_="circuit.expected_efficiency",
        )
    elif not 0 < efficiency <= 1:
        findings.error(
            "EFFICIENCY_OUT_OF_BOUNDS",
            f"Expected efficiency must be a fraction greater than 0 and at most 1, "
            f"received {efficiency:g}.",
            field_="circuit.expected_efficiency",
            suggestion="Enter 0.82 for 82%, not 82.",
        )


def _check_generator(request: TestRequestIn, findings: _Findings) -> None:
    generator = request.generator
    limits = config.GENERATOR

    if generator.max_stages < 1:
        findings.error(
            "MAX_STAGES_INVALID",
            f"Maximum stages must be at least 1, received {generator.max_stages}.",
            field_="generator.max_stages",
        )
    if generator.max_stages > limits.max_stages:
        findings.error(
            "MAX_STAGES_ABOVE_RATING",
            f"The generator has {limits.max_stages} stages; {generator.max_stages} were "
            f"requested.",
            field_="generator.max_stages",
            suggestion=f"Use {limits.max_stages} or fewer.",
        )
    if generator.min_active_stages < 1:
        findings.error(
            "MIN_STAGES_INVALID",
            f"Minimum active stages must be at least 1, received "
            f"{generator.min_active_stages}.",
            field_="generator.min_active_stages",
        )
    if generator.min_active_stages > generator.max_stages:
        findings.error(
            "STAGE_RANGE_INVERTED",
            f"Minimum active stages ({generator.min_active_stages}) exceeds maximum "
            f"stages ({generator.max_stages}).",
            field_="generator.min_active_stages",
        )
    if generator.max_stage_voltage_kv <= 0:
        findings.error(
            "STAGE_VOLTAGE_INVALID",
            f"Maximum stage voltage must be greater than zero, received "
            f"{generator.max_stage_voltage_kv:g} kV.",
            field_="generator.max_stage_voltage_kv",
        )
    if generator.max_stage_voltage_kv > limits.max_stage_voltage_kv:
        findings.error(
            "STAGE_VOLTAGE_ABOVE_RATING",
            f"Requested stage voltage {generator.max_stage_voltage_kv:g} kV exceeds the "
            f"rated {limits.max_stage_voltage_kv:g} kV per stage.",
            field_="generator.max_stage_voltage_kv",
        )
    if generator.stage_capacitance_uf <= 0:
        findings.error(
            "STAGE_CAPACITANCE_INVALID",
            f"Stage capacitance must be greater than zero, received "
            f"{generator.stage_capacitance_uf:g} uF.",
            field_="generator.stage_capacitance_uf",
        )

    # Generator ceiling. Only meaningful once the inputs above are usable.
    efficiency = request.circuit.expected_efficiency
    if (
        generator.max_stages >= 1
        and generator.max_stage_voltage_kv > 0
        and math.isfinite(efficiency)
        and 0 < efficiency <= 1
        and math.isfinite(request.required_peak_kv)
        and request.required_peak_kv > 0
    ):
        ceiling = generator.max_stages * generator.max_stage_voltage_kv * efficiency
        if request.required_peak_kv > ceiling:
            findings.error(
                "EXCEEDS_GENERATOR_CEILING",
                f"{request.required_peak_kv:g} kV cannot be reached. With "
                f"{generator.max_stages} stages at {generator.max_stage_voltage_kv:g} kV "
                f"and {efficiency:.0%} efficiency the generator tops out at "
                f"{ceiling:.0f} kV.",
                field_="required_peak_kv",
                suggestion=(
                    f"Reduce the test level to {ceiling:.0f} kV or below, or revise the "
                    f"efficiency estimate if there is evidence the generator performs "
                    f"better than {efficiency:.0%}."
                ),
            )
        else:
            stages_required = math.ceil(
                request.required_peak_kv / (generator.max_stage_voltage_kv * efficiency)
            )
            if stages_required < generator.min_active_stages:
                findings.notice(
                    "STAGES_RAISED_TO_MINIMUM",
                    f"Only {stages_required} stage(s) would be needed electrically, but "
                    f"the generator requires at least {generator.min_active_stages} "
                    f"active stages, so {generator.min_active_stages} will be used at a "
                    f"lower charging voltage.",
                    field_="generator.min_active_stages",
                )


def _check_resistor_stock(
    label: str,
    stock: Sequence[ResistorStockIn],
    findings: _Findings,
    *,
    field_base: str,
) -> None:
    """Shared between a test request's inventory and an admin generator profile."""
    if not stock:
        findings.error(
            f"{label.upper()}_INVENTORY_EMPTY",
            f"No {label} resistors are available, so no candidate configuration can "
            f"be built.",
            field_=field_base,
        )
    for index, item in enumerate(stock):
        if not math.isfinite(item.ohms) or item.ohms <= 0:
            findings.error(
                "RESISTOR_VALUE_INVALID",
                f"{label.title()} resistor #{index + 1} has a non-positive value "
                f"({item.ohms:g} ohm).",
                field_=f"{field_base}[{index}].ohms",
            )
        if item.units_per_stage < 1:
            findings.error(
                "RESISTOR_UNITS_INVALID",
                f"{label.title()} resistor {item.ohms:g} ohm has "
                f"{item.units_per_stage} units per stage. At least 1 is needed for "
                f"the value to be usable.",
                field_=f"{field_base}[{index}].units_per_stage",
            )

    duplicates = _duplicate_values([item.ohms for item in stock])
    if duplicates:
        findings.warn(
            "RESISTOR_VALUE_DUPLICATED",
            f"{label.title()} resistor value(s) "
            f"{', '.join(f'{value:g}' for value in duplicates)} ohm appear more than "
            f"once. Combine them into a single entry with the total units per stage, "
            f"otherwise the candidate generator will treat them as separate options.",
            field_=field_base,
        )


def _check_external_resistors(
    name: str, values: Sequence[float], findings: _Findings, *, field_base: str
) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value) or value <= 0:
            findings.error(
                "EXTERNAL_RESISTOR_INVALID",
                f"External resistor #{index + 1} in {name} has a non-positive "
                f"value ({value:g} ohm).",
                field_=f"{field_base}[{index}]",
            )


def _duplicate_values(values: Sequence[float]) -> list[float]:
    seen: set[float] = set()
    duplicates: list[float] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _check_inventory(request: TestRequestIn, findings: _Findings) -> None:
    inventory = request.inventory

    _check_resistor_stock("front", inventory.front, findings, field_base="inventory.front")
    _check_resistor_stock("tail", inventory.tail, findings, field_base="inventory.tail")
    _check_external_resistors(
        "external_front_ohms",
        inventory.external_front_ohms,
        findings,
        field_base="inventory.external_front_ohms",
    )
    _check_external_resistors(
        "external_tail_ohms",
        inventory.external_tail_ohms,
        findings,
        field_base="inventory.external_tail_ohms",
    )

    if request.impulse_type == config.SWITCHING:
        _check_switching_resistance_reach(request, findings)


def _check_switching_resistance_reach(request: TestRequestIn, findings: _Findings) -> None:
    """Warn that stocked resistors cannot produce a switching impulse.

    A switching impulse needs roughly two orders of magnitude more front resistance
    than a lightning impulse, because the front time is 250 us instead of 1.2 us. The
    dataset's switching rows bear this out, and the stocked inventory cannot get close.
    """
    inventory = request.inventory
    front_stock = [item for item in inventory.front if item.ohms > 0 and item.units_per_stage >= 1]
    tail_stock = [item for item in inventory.tail if item.ohms > 0 and item.units_per_stage >= 1]
    if not front_stock or not tail_stock:
        return

    front_reach = max(item.ohms * item.units_per_stage for item in front_stock)
    tail_reach = max(item.ohms * item.units_per_stage for item in tail_stock)

    try:
        domain = config.domain_for(config.SWITCHING)
    except (FileNotFoundError, KeyError):
        return
    needed_front = domain["inputs"]["Front_R_Stage"]
    needed_tail = domain["inputs"]["Tail_R_Stage"]

    front_short = front_reach < needed_front["min"] and not inventory.external_front_ohms
    tail_short = tail_reach < needed_tail["min"] and not inventory.external_tail_ohms

    if front_short or tail_short:
        shortfalls = []
        if front_short:
            shortfalls.append(
                f"front resistance reaches {front_reach:g} ohm/stage but the training "
                f"data uses {needed_front['min']:.0f}-{needed_front['max']:.0f} ohm/stage"
            )
        if tail_short:
            shortfalls.append(
                f"tail resistance reaches {tail_reach:g} ohm/stage but the training data "
                f"uses {needed_tail['min']:.0f}-{needed_tail['max']:.0f} ohm/stage"
            )
        findings.warn(
            "SWITCHING_NEEDS_EXTERNAL_RESISTORS",
            "The stocked resistor inventory cannot produce a 250/2500 us switching "
            "impulse: " + "; and ".join(shortfalls) + ".",
            field_="inventory.external_front_ohms",
            suggestion=(
                "Enter the external front and tail resistor values actually available "
                "in the laboratory. The problem statement lists external resistors as "
                "part of the generator, but supplies no values, so they cannot be "
                "assumed."
            ),
        )


def _check_training_domain(request: TestRequestIn, findings: _Findings) -> list[DomainCheckOut]:
    """Compare each input against the range the residual model was fitted on."""
    try:
        domain = config.domain_for(request.impulse_type)
    except (FileNotFoundError, KeyError) as exc:
        findings.warn(
            "TRAINING_DOMAIN_UNAVAILABLE",
            f"Could not load the ML training domain, so no out-of-domain check was "
            f"performed ({exc}).",
            suggestion="Run: python scripts/extract_ml_domain.py",
        )
        return []

    inputs = domain["inputs"]
    checks: list[DomainCheckOut] = []
    outside: list[str] = []

    for dotted, column, label, unit in DOMAIN_FIELDS:
        try:
            value = _resolve(request, dotted)
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        bounds = inputs[column]
        in_domain = bounds["min"] <= value <= bounds["max"]
        checks.append(
            DomainCheckOut(
                field=dotted,
                label=label,
                value=value,
                training_min=bounds["min"],
                training_max=bounds["max"],
                in_domain=in_domain,
                unit=unit,
            )
        )
        if not in_domain:
            rendered_unit = f" {unit}" if unit else ""
            outside.append(
                f"{label} {value:g}{rendered_unit} (trained on "
                f"{bounds['min']:.4g}-{bounds['max']:.4g}{rendered_unit})"
            )

    if outside:
        findings.warn(
            "OUTSIDE_ML_TRAINING_DOMAIN",
            f"{len(outside)} input(s) fall outside the range the residual model was "
            f"trained on: " + "; ".join(outside) + ". The correction will be an "
            "extrapolation from unrelated neighbours, so treat the corrected prediction "
            "as no better than the physics-only prediction.",
            suggestion=(
                "The synthetic dataset only covers "
                f"{inputs['Test_kV']['min']:.0f}-{inputs['Test_kV']['max']:.0f} kV for "
                f"{request.impulse_type} impulses, whereas the generator is specified "
                f"for {config.GENERATOR.min_test_voltage_kv:g}-"
                f"{config.GENERATOR.max_test_voltage_kv:g} kV. Extending the dataset is "
                "the only real fix."
            ),
        )

    return checks


def _add_standing_caveats(request: TestRequestIn, findings: _Findings) -> None:
    """Caveats that apply to every run, surfaced so they cannot be overlooked."""
    findings.notice(
        "NO_LABORATORY_VALIDATION",
        "No laboratory validation has been completed. Every prediction this tool "
        "produces comes from a simplified circuit model corrected by a model fitted to "
        "synthetic data. A predicted pass is not evidence that the physical test will "
        "pass, and all recommendations remain subject to review by a qualified engineer "
        "and the laboratory's approved test procedure.",
    )
    # Deliberately not attached to a field: it is a property of the model, not of
    # anything the engineer typed, and attaching it would duplicate the message inline.
    findings.warn(
        "CREST_VOLTAGE_NOT_MODELLED",
        "The supplied material contains no physics model for crest voltage: the "
        "predicted crest is simply the requested test voltage. Peak-voltage accuracy "
        "figures therefore carry no engineering information until a voltage-efficiency "
        "model is implemented.",
    )

    spec = config.IMPULSE_SPECS[request.impulse_type]
    if spec.half_value_time.provenance.status.value == "conflict":
        findings.warn(
            "TOLERANCE_WINDOW_DISPUTED",
            f"The time-to-half-value window used for {spec.label.lower()}s "
            f"({spec.half_value_time.minimum:g}-{spec.half_value_time.maximum:g} us) "
            f"comes from the supplied workbook and is probably stricter than the "
            f"standard requires. {spec.half_value_time.provenance.note}",
            suggestion="Confirm the window against the IEC 60060-1 text before relying "
            "on a compliance verdict for switching impulses.",
        )

    test_object = config.TEST_OBJECTS_BY_ID.get(request.test_object_type_id)
    if test_object is not None and test_object.provenance.status.value == "placeholder":
        findings.warn(
            "TEST_OBJECT_CAPACITANCE_IS_PLACEHOLDER",
            f"{test_object.label!r} has no measured capacitance in the supplied "
            f"material. The pre-filled value is a placeholder with no engineering "
            f"basis.",
            field_="circuit.load_capacitance_pf",
            suggestion="Replace it with a measured or datasheet capacitance for the "
            "actual test object.",
        )

    if request.polarity != "positive":
        findings.notice(
            "POLARITY_NOT_MODELLED",
            "Polarity is recorded for traceability but affects no calculation. Neither "
            "the supplied workbook nor the dataset contains a polarity column.",
            field_="polarity",
        )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _window(tolerance: config.Tolerance) -> AcceptanceWindowOut:
    return AcceptanceWindowOut(
        nominal=tolerance.nominal,
        min=tolerance.minimum,
        max=tolerance.maximum,
        unit=tolerance.unit,
        provenance_status=tolerance.provenance.status.value,
        provenance_note=tolerance.provenance.note,
    )


def _normalise(request: TestRequestIn) -> NormalizedRequestOut:
    spec = config.IMPULSE_SPECS[request.impulse_type]
    generator = request.generator
    efficiency = request.circuit.expected_efficiency

    stages_required = math.ceil(
        request.required_peak_kv / (generator.max_stage_voltage_kv * efficiency)
    )
    stages_used = max(stages_required, generator.min_active_stages)
    charge_per_stage_kv = request.required_peak_kv / (efficiency * stages_used)

    return NormalizedRequestOut(
        test_object_id=request.test_object_type_id,
        test_object_label=request.test_object_label,
        standard_id=request.standard_id,
        standard_label=config.STANDARD_LABEL,
        impulse_type=request.impulse_type,
        impulse_designation=spec.designation,
        polarity=request.polarity,
        required_peak_kv=request.required_peak_kv,
        target_front_time_us=spec.front_time.nominal,
        target_half_value_time_us=spec.half_value_time.nominal,
        front_time_definition=spec.front_time_definition,
        front_time_window=_window(spec.front_time),
        half_value_time_window=_window(spec.half_value_time),
        peak_relative_tolerance=spec.peak_relative_tolerance,
        derived=DerivedQuantitiesOut(
            total_load_capacitance_pf=request.circuit.total_load_capacitance_pf,
            stages_required=stages_used,
            charging_voltage_per_stage_kv=charge_per_stage_kv,
            stage_utilisation=charge_per_stage_kv / generator.max_stage_voltage_kv,
            generator_capacitance_uf=generator.stage_capacitance_uf / stages_used,
            max_achievable_peak_kv=(
                generator.max_stages * generator.max_stage_voltage_kv * efficiency
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Admin generator profile
# ---------------------------------------------------------------------------


@dataclass
class ProfileCheck:
    """Outcome of validating a laboratory generator profile."""

    errors: list[IssueOut] = field(default_factory=list)
    warnings: list[IssueOut] = field(default_factory=list)
    notices: list[IssueOut] = field(default_factory=list)
    reach: ProfileReachOut | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_profile(profile: GeneratorProfile) -> ProfileCheck:
    """Check a generator profile submitted from the admin interface.

    Structural only: it says nothing about any particular test, because a profile is
    saved before anyone knows what will be tested on it. The hard ratings in
    ``config.GENERATOR`` act as a ceiling, since those come from the supplied material
    and describe physical equipment the admin cannot reconfigure by typing.
    """
    findings = _Findings()
    rating = config.GENERATOR

    if profile.max_stages < 1:
        findings.error(
            "MAX_STAGES_INVALID",
            f"Stages available must be at least 1, received {profile.max_stages}.",
            field_="profile.max_stages",
        )
    elif profile.max_stages > rating.max_stages:
        findings.error(
            "MAX_STAGES_ABOVE_RATING",
            f"The generator physically has {rating.max_stages} stages; "
            f"{profile.max_stages} were entered.",
            field_="profile.max_stages",
            suggestion=f"Enter {rating.max_stages} or fewer.",
        )

    if profile.min_active_stages < 1:
        findings.error(
            "MIN_STAGES_INVALID",
            f"Minimum active stages must be at least 1, received "
            f"{profile.min_active_stages}.",
            field_="profile.min_active_stages",
        )
    elif profile.min_active_stages > profile.max_stages:
        findings.error(
            "STAGE_RANGE_INVERTED",
            f"Minimum active stages ({profile.min_active_stages}) exceeds stages "
            f"available ({profile.max_stages}).",
            field_="profile.min_active_stages",
        )
    elif profile.min_active_stages < rating.min_active_stages:
        findings.warn(
            "BELOW_STATED_MINIMUM_STAGES",
            f"The problem statement specifies a minimum of "
            f"{rating.min_active_stages} active stages; {profile.min_active_stages} "
            f"was entered.",
            field_="profile.min_active_stages",
        )

    if profile.max_stage_voltage_kv <= 0:
        findings.error(
            "STAGE_VOLTAGE_INVALID",
            f"Stage voltage must be greater than zero, received "
            f"{profile.max_stage_voltage_kv:g} kV.",
            field_="profile.max_stage_voltage_kv",
        )
    elif profile.max_stage_voltage_kv > rating.max_stage_voltage_kv:
        findings.error(
            "STAGE_VOLTAGE_ABOVE_RATING",
            f"Stage voltage {profile.max_stage_voltage_kv:g} kV exceeds the rated "
            f"{rating.max_stage_voltage_kv:g} kV per stage. Charging a stage above its "
            f"rating is a damage and safety risk, not a configuration choice.",
            field_="profile.max_stage_voltage_kv",
        )

    if profile.stage_capacitance_uf <= 0:
        findings.error(
            "STAGE_CAPACITANCE_INVALID",
            f"Stage capacitance must be greater than zero, received "
            f"{profile.stage_capacitance_uf:g} uF.",
            field_="profile.stage_capacitance_uf",
        )

    if profile.capacitors_per_stage < 1:
        findings.error(
            "CAPACITORS_PER_STAGE_INVALID",
            f"Capacitors per stage must be at least 1, received "
            f"{profile.capacitors_per_stage}.",
            field_="profile.capacitors_per_stage",
        )

    if not 0 < profile.default_efficiency <= 1:
        findings.error(
            "EFFICIENCY_OUT_OF_BOUNDS",
            f"Default efficiency must be a fraction greater than 0 and at most 1, "
            f"received {profile.default_efficiency:g}.",
            field_="profile.default_efficiency",
            suggestion="Enter 0.82 for 82%, not 82.",
        )

    if profile.default_stray_inductance_uh < 0:
        findings.error(
            "CIRCUIT_VALUE_NEGATIVE",
            f"Default stray inductance cannot be negative, received "
            f"{profile.default_stray_inductance_uh:g} uH.",
            field_="profile.default_stray_inductance_uh",
        )

    _check_resistor_stock(
        "front", profile.front_resistors, findings, field_base="profile.front_resistors"
    )
    _check_resistor_stock(
        "tail", profile.tail_resistors, findings, field_base="profile.tail_resistors"
    )
    _check_external_resistors(
        "external front resistors",
        profile.external_front_ohms,
        findings,
        field_base="profile.external_front_ohms",
    )
    _check_external_resistors(
        "external tail resistors",
        profile.external_tail_ohms,
        findings,
        field_base="profile.external_tail_ohms",
    )

    reach = _profile_reach(profile) if not findings.errors else None
    if reach is not None:
        _check_profile_coverage(profile, reach, findings)

    return ProfileCheck(
        errors=findings.errors,
        warnings=findings.warnings,
        notices=findings.notices,
        reach=reach,
    )


def _profile_reach(profile: GeneratorProfile) -> ProfileReachOut:
    front_reach = max(
        (item.ohms * item.units_per_stage for item in profile.front_resistors), default=0.0
    )
    tail_reach = max(
        (item.ohms * item.units_per_stage for item in profile.tail_resistors), default=0.0
    )
    charge_ceiling = profile.max_stages * profile.max_stage_voltage_kv
    return ProfileReachOut(
        max_front_ohms_per_stage=front_reach,
        max_tail_ohms_per_stage=tail_reach,
        max_charge_voltage_kv=charge_ceiling,
        max_achievable_peak_kv=charge_ceiling * profile.default_efficiency,
        front_values=len(profile.front_resistors),
        tail_values=len(profile.tail_resistors),
        has_external_front=bool(profile.external_front_ohms),
        has_external_tail=bool(profile.external_tail_ohms),
    )


def _check_profile_coverage(
    profile: GeneratorProfile, reach: ProfileReachOut, findings: _Findings
) -> None:
    """Report what this profile will and will not be able to do."""
    rating = config.GENERATOR

    if reach.max_achievable_peak_kv < rating.max_test_voltage_kv:
        findings.notice(
            "PROFILE_BELOW_STATED_RANGE",
            f"This setup reaches {reach.max_achievable_peak_kv:.0f} kV, below the "
            f"{rating.max_test_voltage_kv:g} kV top of the range stated in the problem "
            f"statement. Test requests above "
            f"{reach.max_achievable_peak_kv:.0f} kV will be rejected.",
            field_="profile.max_stages",
        )

    try:
        needed = config.domain_for(config.SWITCHING)["inputs"]
    except (FileNotFoundError, KeyError):
        return

    front_short = (
        reach.max_front_ohms_per_stage < needed["Front_R_Stage"]["min"]
        and not profile.external_front_ohms
    )
    tail_short = (
        reach.max_tail_ohms_per_stage < needed["Tail_R_Stage"]["min"]
        and not profile.external_tail_ohms
    )
    if front_short or tail_short:
        findings.warn(
            "PROFILE_CANNOT_DO_SWITCHING",
            f"This setup cannot produce a 250/2500 us switching impulse. In series it "
            f"reaches {reach.max_front_ohms_per_stage:g} ohm/stage front and "
            f"{reach.max_tail_ohms_per_stage:g} ohm/stage tail, whereas switching needs "
            f"roughly {needed['Front_R_Stage']['min']:.0f} ohm/stage front and "
            f"{needed['Tail_R_Stage']['min']:.0f} ohm/stage tail. Operators will be "
            f"able to request switching impulses but will see a warning on every one.",
            field_="profile.external_front_ohms",
            suggestion=(
                "Add the external front and tail resistor values the laboratory "
                "actually has. No values were supplied with the project material, so "
                "they cannot be assumed."
            ),
        )


def _relevant_assumptions(request: TestRequestIn) -> list[dict]:
    """Assumptions worth showing for this particular request."""
    wanted = {
        "generator.ratings",
        "generator.test_voltage_range",
        "efficiency",
        "physics.crest_voltage",
        "standard.edition",
        "ml.training_domain",
        "ml.validation_status",
        f"tolerance.{request.impulse_type}.front_time",
        f"tolerance.{request.impulse_type}.half_value_time",
        f"tolerance.{request.impulse_type}.peak",
        f"tolerance.{request.impulse_type}.front_time_definition",
    }
    if request.impulse_type == config.SWITCHING:
        wanted.add("inventory.switching_shortfall")
    test_object = config.TEST_OBJECTS_BY_ID.get(request.test_object_type_id)
    if test_object is not None and test_object.provenance.status.value == "placeholder":
        wanted.add("test_objects.capacitance")

    return [entry.to_dict() for entry in config.ASSUMPTIONS.entries if entry.key in wanted]
