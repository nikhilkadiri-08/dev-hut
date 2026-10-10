"""Tests for the input layer.

Every assertion here checks either a value stated in the supplied material or plain
arithmetic reproduced from the supplied workbook. None of them assert an invented
physical relationship.
"""

from __future__ import annotations

import math

import pytest

from backend import config
from backend.schemas import (
    GeneratorConfigIn,
    InventoryIn,
    ResistorStockIn,
    TestCircuitIn,
    TestRequestIn,
)
from backend.validation import validate_request


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def workbook_request(**overrides) -> TestRequestIn:
    """The worked example from the supplied workbook, with optional overrides.

    Hybrid Calculator: 1425 kV lightning, 850 pF load, 500 pF divider, 150 pF stray,
    18.5 uH stray inductance, 0.82 efficiency.
    """
    circuit_overrides = overrides.pop("circuit", {})
    payload = {
        "test_object_type_id": "workbook_example",
        "impulse_type": "lightning",
        "required_peak_kv": 1425.0,
        "circuit": {
            "load_capacitance_pf": 850.0,
            "divider_capacitance_pf": 500.0,
            "stray_capacitance_pf": 150.0,
            "stray_inductance_uh": 18.5,
            "expected_efficiency": 0.82,
            **circuit_overrides,
        },
    }
    payload.update(overrides)
    return TestRequestIn(**payload)


# ---------------------------------------------------------------------------
# A valid, feasible input
# ---------------------------------------------------------------------------


def test_workbook_example_is_accepted():
    report = validate_request(workbook_request())
    assert report.ok, codes(report.errors)
    assert report.normalized is not None


def test_derived_quantities_match_the_workbook():
    """Reproduce Hybrid Calculator cells E7 to E11 for the worked example."""
    report = validate_request(workbook_request())
    derived = report.normalized.derived

    # E7: total load = 850 + 500 + 150
    assert derived.total_load_capacitance_pf == pytest.approx(1500.0)
    # E8: CEILING(1425 / (200 * 0.82)) = CEILING(8.69) = 9
    assert derived.stages_required == 9
    # E9: 1425 / (0.82 * 9)
    assert derived.charging_voltage_per_stage_kv == pytest.approx(193.0894308943, rel=1e-9)
    # E10: 193.089 / 200
    assert derived.stage_utilisation == pytest.approx(0.9654471545, rel=1e-9)
    # E11: 3 uF / 9 stages
    assert derived.generator_capacitance_uf == pytest.approx(1 / 3, rel=1e-9)


def test_target_waveform_follows_the_selected_impulse_type():
    """Workbook cells E5 and E6 switch the targets on the impulse type."""
    lightning = validate_request(workbook_request()).normalized
    assert lightning.target_front_time_us == pytest.approx(1.2)
    assert lightning.target_half_value_time_us == pytest.approx(50.0)

    switching = validate_request(
        workbook_request(impulse_type="switching", required_peak_kv=1025.0)
    ).normalized
    assert switching.target_front_time_us == pytest.approx(250.0)
    assert switching.target_half_value_time_us == pytest.approx(2500.0)


def test_lightning_acceptance_window_matches_the_workbook_gate():
    """Workbook gate H13 uses 0.84-1.56 us front and 40-60 us half-value."""
    normalized = validate_request(workbook_request()).normalized
    assert normalized.front_time_window.min == pytest.approx(0.84)
    assert normalized.front_time_window.max == pytest.approx(1.56)
    assert normalized.half_value_time_window.min == pytest.approx(40.0)
    assert normalized.half_value_time_window.max == pytest.approx(60.0)
    assert normalized.peak_relative_tolerance == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Missing or invalid inputs
# ---------------------------------------------------------------------------


def test_load_capacitance_is_required():
    with pytest.raises(Exception):
        TestRequestIn(impulse_type="lightning", required_peak_kv=1425.0, circuit={})


@pytest.mark.parametrize(
    "overrides,expected_code",
    [
        ({"load_capacitance_pf": 0.0}, "LOAD_CAPACITANCE_NOT_POSITIVE"),
        ({"load_capacitance_pf": -10.0}, "CIRCUIT_VALUE_NEGATIVE"),
        ({"stray_capacitance_pf": -1.0}, "CIRCUIT_VALUE_NEGATIVE"),
        ({"expected_efficiency": 0.0}, "EFFICIENCY_OUT_OF_BOUNDS"),
        ({"expected_efficiency": 82.0}, "EFFICIENCY_OUT_OF_BOUNDS"),
        ({"expected_efficiency": math.nan}, "EFFICIENCY_NOT_A_NUMBER"),
    ],
)
def test_invalid_circuit_values_are_rejected(overrides, expected_code):
    report = validate_request(workbook_request(circuit=overrides))
    assert not report.ok
    assert expected_code in codes(report.errors)


def test_percentage_efficiency_gets_a_helpful_suggestion():
    report = validate_request(workbook_request(circuit={"expected_efficiency": 82.0}))
    issue = next(i for i in report.errors if i.code == "EFFICIENCY_OUT_OF_BOUNDS")
    assert "0.82" in issue.suggestion


def test_unknown_test_object_lists_the_alternatives():
    report = validate_request(workbook_request(test_object_type_id="not_a_real_object"))
    issue = next(i for i in report.errors if i.code == "UNKNOWN_TEST_OBJECT")
    assert "workbook_example" in issue.suggestion


def test_unknown_standard_is_rejected():
    report = validate_request(workbook_request(standard_id="IEC_61000"))
    assert "UNKNOWN_STANDARD" in codes(report.errors)


def test_zero_divider_capacitance_warns_but_does_not_block():
    report = validate_request(workbook_request(circuit={"divider_capacitance_pf": 0.0}))
    assert report.ok
    assert "DIVIDER_CAPACITANCE_ZERO" in codes(report.warnings)


# ---------------------------------------------------------------------------
# Requested voltage beyond verified generator limits
# ---------------------------------------------------------------------------


def test_voltage_above_the_stated_range_is_rejected():
    report = validate_request(workbook_request(required_peak_kv=3000.0))
    assert not report.ok
    assert "PEAK_VOLTAGE_ABOVE_RANGE" in codes(report.errors)


def test_voltage_below_the_stated_range_is_rejected():
    report = validate_request(workbook_request(required_peak_kv=10.0))
    assert not report.ok
    assert "PEAK_VOLTAGE_BELOW_RANGE" in codes(report.errors)


def test_generator_ceiling_is_reported_with_the_achievable_voltage():
    """15 stages x 200 kV x 0.82 efficiency = 2460 kV, so 2400 kV fits but 2000 kV
    does not once the available stages are reduced to 12."""
    report = validate_request(
        workbook_request(
            required_peak_kv=2000.0,
            generator=GeneratorConfigIn(max_stages=12),
        )
    )
    assert not report.ok
    issue = next(i for i in report.errors if i.code == "EXCEEDS_GENERATOR_CEILING")
    # 12 * 200 * 0.82 = 1968 kV
    assert "1968" in issue.message


def test_the_full_generator_reaches_the_stated_upper_test_level():
    report = validate_request(workbook_request(required_peak_kv=2400.0))
    assert "EXCEEDS_GENERATOR_CEILING" not in codes(report.errors)
    assert report.normalized.derived.max_achievable_peak_kv == pytest.approx(2460.0)


def test_requesting_more_stages_than_the_generator_has_is_rejected():
    report = validate_request(workbook_request(generator=GeneratorConfigIn(max_stages=20)))
    assert "MAX_STAGES_ABOVE_RATING" in codes(report.errors)


def test_stage_count_is_raised_to_the_minimum_active_stages():
    """At a low test level the arithmetic wants one stage, but the problem statement
    requires at least two active stages."""
    report = validate_request(workbook_request(required_peak_kv=100.0))
    assert report.ok
    assert "STAGES_RAISED_TO_MINIMUM" in codes(report.notices)
    assert report.normalized.derived.stages_required == config.GENERATOR.min_active_stages


# ---------------------------------------------------------------------------
# Unavailable or unusable resistor values
# ---------------------------------------------------------------------------


def test_empty_front_inventory_is_rejected():
    report = validate_request(
        workbook_request(inventory=InventoryIn(front=[], tail=list(InventoryIn().tail)))
    )
    assert not report.ok
    assert "FRONT_INVENTORY_EMPTY" in codes(report.errors)


def test_non_positive_resistor_value_is_rejected():
    inventory = InventoryIn(front=[ResistorStockIn(ohms=0.0, units_per_stage=4)])
    report = validate_request(workbook_request(inventory=inventory))
    assert "RESISTOR_VALUE_INVALID" in codes(report.errors)


def test_resistor_with_no_available_units_is_rejected():
    inventory = InventoryIn(front=[ResistorStockIn(ohms=50.0, units_per_stage=0)])
    report = validate_request(workbook_request(inventory=inventory))
    assert "RESISTOR_UNITS_INVALID" in codes(report.errors)


def test_switching_impulse_warns_that_stocked_resistors_cannot_reach_the_target():
    """The stocked front resistors reach 4 x 500 = 2000 ohm/stage, but the switching
    training data uses 9370 ohm/stage and above."""
    report = validate_request(
        workbook_request(impulse_type="switching", required_peak_kv=1025.0)
    )
    assert report.ok
    issue = next(
        i for i in report.warnings if i.code == "SWITCHING_NEEDS_EXTERNAL_RESISTORS"
    )
    assert "2000" in issue.message
    assert "external" in issue.suggestion.lower()


def test_supplying_external_resistors_clears_the_switching_warning():
    inventory = InventoryIn(external_front_ohms=[12000.0], external_tail_ohms=[1200.0])
    report = validate_request(
        workbook_request(
            impulse_type="switching", required_peak_kv=1025.0, inventory=inventory
        )
    )
    assert "SWITCHING_NEEDS_EXTERNAL_RESISTORS" not in codes(report.warnings)


def test_lightning_impulse_does_not_raise_the_switching_shortfall():
    report = validate_request(workbook_request())
    assert "SWITCHING_NEEDS_EXTERNAL_RESISTORS" not in codes(report.warnings)


def test_negative_external_resistor_is_rejected():
    inventory = InventoryIn(external_front_ohms=[-100.0])
    report = validate_request(workbook_request(inventory=inventory))
    assert "EXTERNAL_RESISTOR_INVALID" in codes(report.errors)


# ---------------------------------------------------------------------------
# Out-of-domain detection
# ---------------------------------------------------------------------------


def test_workbook_example_sits_inside_the_training_domain():
    report = validate_request(workbook_request())
    assert "OUTSIDE_ML_TRAINING_DOMAIN" not in codes(report.warnings)
    assert all(check.in_domain for check in report.domain_checks)


def test_voltage_inside_generator_limits_but_outside_training_data_warns():
    """500 kV is well within the 50-2400 kV generator range but far below the
    1350-1500 kV span the lightning training rows cover."""
    report = validate_request(workbook_request(required_peak_kv=500.0))
    assert report.ok
    issue = next(i for i in report.warnings if i.code == "OUTSIDE_ML_TRAINING_DOMAIN")
    assert "Required peak voltage" in issue.message

    check = next(c for c in report.domain_checks if c.field == "required_peak_kv")
    assert not check.in_domain
    assert check.training_min == pytest.approx(1350.0, abs=1.0)


def test_domain_bounds_depend_on_the_impulse_type():
    """1425 kV is in domain for lightning and out of domain for switching, because the
    switching training rows only span 950-1100 kV."""
    lightning = validate_request(workbook_request())
    switching = validate_request(workbook_request(impulse_type="switching"))

    lightning_check = next(c for c in lightning.domain_checks if c.field == "required_peak_kv")
    switching_check = next(c for c in switching.domain_checks if c.field == "required_peak_kv")
    assert lightning_check.in_domain
    assert not switching_check.in_domain


def test_every_domain_field_is_reported():
    report = validate_request(workbook_request())
    assert len(report.domain_checks) == 6


# ---------------------------------------------------------------------------
# Honesty caveats
# ---------------------------------------------------------------------------


def test_every_run_states_that_there_is_no_laboratory_validation():
    report = validate_request(workbook_request())
    assert "NO_LABORATORY_VALIDATION" in codes(report.notices)


def test_every_run_states_that_crest_voltage_is_not_modelled():
    report = validate_request(workbook_request())
    assert "CREST_VOLTAGE_NOT_MODELLED" in codes(report.warnings)


def test_switching_run_flags_the_disputed_half_value_window():
    report = validate_request(
        workbook_request(impulse_type="switching", required_peak_kv=1025.0)
    )
    assert "TOLERANCE_WINDOW_DISPUTED" in codes(report.warnings)


def test_lightning_run_does_not_flag_the_half_value_window():
    report = validate_request(workbook_request())
    assert "TOLERANCE_WINDOW_DISPUTED" not in codes(report.warnings)


def test_placeholder_test_object_is_flagged():
    report = validate_request(workbook_request(test_object_type_id="station_post"))
    assert "TEST_OBJECT_CAPACITANCE_IS_PLACEHOLDER" in codes(report.warnings)


def test_workbook_test_object_is_not_flagged_as_placeholder():
    report = validate_request(workbook_request())
    assert "TEST_OBJECT_CAPACITANCE_IS_PLACEHOLDER" not in codes(report.warnings)


def test_relevant_assumptions_are_attached():
    report = validate_request(workbook_request())
    keys = {entry["key"] for entry in report.relevant_assumptions}
    assert "physics.crest_voltage" in keys
    assert "ml.validation_status" in keys
    assert "inventory.switching_shortfall" not in keys


# ---------------------------------------------------------------------------
# Input changes produce logically consistent changes
# ---------------------------------------------------------------------------


def test_higher_test_voltage_never_needs_fewer_stages():
    previous = 0
    for voltage in (200.0, 500.0, 900.0, 1425.0, 2000.0, 2400.0):
        derived = validate_request(workbook_request(required_peak_kv=voltage)).normalized.derived
        assert derived.stages_required >= previous
        previous = derived.stages_required


def test_lower_efficiency_never_needs_fewer_stages():
    previous = 0
    for efficiency in (0.95, 0.85, 0.82, 0.70, 0.50):
        derived = validate_request(
            workbook_request(circuit={"expected_efficiency": efficiency})
        ).normalized.derived
        assert derived.stages_required >= previous
        previous = derived.stages_required


def test_total_load_capacitance_is_the_sum_of_the_three_contributions():
    derived = validate_request(
        workbook_request(
            circuit={
                "load_capacitance_pf": 1000.0,
                "divider_capacitance_pf": 600.0,
                "stray_capacitance_pf": 200.0,
            }
        )
    ).normalized.derived
    assert derived.total_load_capacitance_pf == pytest.approx(1800.0)


def test_more_stages_means_lower_generator_capacitance():
    """Stage capacitors in series divide, per workbook cell E11."""
    low = validate_request(workbook_request(required_peak_kv=500.0)).normalized.derived
    high = validate_request(workbook_request(required_peak_kv=2400.0)).normalized.derived
    assert high.stages_required > low.stages_required
    assert high.generator_capacitance_uf < low.generator_capacitance_uf


def test_stage_utilisation_never_exceeds_one_for_a_feasible_request():
    for voltage in (100.0, 800.0, 1425.0, 2400.0):
        derived = validate_request(workbook_request(required_peak_kv=voltage)).normalized.derived
        assert derived.stage_utilisation <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Configuration integrity
# ---------------------------------------------------------------------------


def test_generator_ceiling_covers_the_stated_test_range():
    """15 stages x 200 kV x 0.82 = 2460 kV, consistent with the stated 2400 kV limit."""
    ceiling = (
        config.GENERATOR.max_charge_voltage_kv * config.DEFAULT_EFFICIENCY
    )
    assert ceiling >= config.GENERATOR.max_test_voltage_kv


def test_disputed_and_unverified_assumptions_are_registered():
    statuses = {e.key: e.provenance.status.value for e in config.ASSUMPTIONS.entries}
    assert statuses["physics.crest_voltage"] == "conflict"
    assert statuses["inventory.switching_shortfall"] == "conflict"
    assert statuses["tolerance.switching.half_value_time"] == "conflict"
    assert statuses["standard.edition"] == "needs_verification"
    assert statuses["efficiency"] == "needs_verification"


def test_no_assumption_is_silently_trusted():
    """Any entry that is not SUPPLIED must carry an explanatory note."""
    for entry in config.ASSUMPTIONS.requiring_attention():
        assert entry.provenance.note, f"{entry.key} has no note explaining the risk"
        assert entry.impact, f"{entry.key} does not say what it affects"
