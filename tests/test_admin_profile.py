"""Tests for the admin generator setup and its effect on the operator interface."""

from __future__ import annotations

import pytest

from backend import config, store
from backend.schemas import (
    GeneratorProfile,
    OperatorRequestIn,
    ResistorStockIn,
    merge_operator_request,
)
from backend.validation import validate_profile, validate_request


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_missing_file_falls_back_to_supplied_material(tmp_path):
    stored = store.load_profile(tmp_path / "absent.json")
    assert stored.is_factory_default
    assert stored.profile.max_stages == config.GENERATOR.max_stages
    assert stored.profile.max_stage_voltage_kv == config.GENERATOR.max_stage_voltage_kv
    assert stored.updated_at is None


def test_saved_profile_round_trips(tmp_path):
    path = tmp_path / "profile.json"
    profile = GeneratorProfile(max_stages=10, external_front_ohms=[12000.0])

    saved = store.save_profile(profile, updated_by="hv lab", path=path)
    assert not saved.is_factory_default
    assert saved.updated_by == "hv lab"
    assert saved.updated_at is not None

    reloaded = store.load_profile(path)
    assert reloaded.profile.max_stages == 10
    assert reloaded.profile.external_front_ohms == [12000.0]
    assert reloaded.updated_by == "hv lab"


def test_unattributed_save_is_recorded_as_unidentified(tmp_path):
    saved = store.save_profile(GeneratorProfile(), updated_by="   ", path=tmp_path / "p.json")
    assert saved.updated_by == "unidentified"


def test_corrupt_file_falls_back_rather_than_crashing(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{ this is not valid json")

    stored = store.load_profile(path)
    assert stored.is_factory_default
    assert stored.profile.max_stages == config.GENERATOR.max_stages


def test_schema_invalid_file_falls_back(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text('{"profile": {"max_stages": "not a number"}}')
    assert store.load_profile(path).is_factory_default


def test_reset_removes_the_saved_file(tmp_path):
    path = tmp_path / "profile.json"
    store.save_profile(GeneratorProfile(max_stages=7), path=path)
    assert path.exists()

    stored = store.reset_profile(path)
    assert not path.exists()
    assert stored.is_factory_default
    assert stored.profile.max_stages == config.GENERATOR.max_stages


def test_reset_on_a_missing_file_is_harmless(tmp_path):
    assert store.reset_profile(tmp_path / "never_existed.json").is_factory_default


# ---------------------------------------------------------------------------
# Profile validation
# ---------------------------------------------------------------------------


def test_supplied_material_profile_is_valid():
    check = validate_profile(GeneratorProfile())
    assert check.ok, codes(check.errors)
    assert check.reach is not None


def test_reach_is_computed_from_the_stocked_values():
    """Four units of the largest stocked value in series: 4 x 500 and 4 x 200."""
    reach = validate_profile(GeneratorProfile()).reach
    assert reach.max_front_ohms_per_stage == pytest.approx(2000.0)
    assert reach.max_tail_ohms_per_stage == pytest.approx(800.0)
    assert reach.max_charge_voltage_kv == pytest.approx(3000.0)
    assert reach.max_achievable_peak_kv == pytest.approx(2460.0)
    assert reach.front_values == 10
    assert reach.tail_values == 12


def test_stage_voltage_above_the_rating_is_refused():
    check = validate_profile(GeneratorProfile(max_stage_voltage_kv=250.0))
    assert not check.ok
    assert "STAGE_VOLTAGE_ABOVE_RATING" in codes(check.errors)


def test_more_stages_than_the_generator_has_is_refused():
    check = validate_profile(GeneratorProfile(max_stages=20))
    assert not check.ok
    assert "MAX_STAGES_ABOVE_RATING" in codes(check.errors)


def test_inverted_stage_range_is_refused():
    check = validate_profile(GeneratorProfile(max_stages=4, min_active_stages=6))
    assert "STAGE_RANGE_INVERTED" in codes(check.errors)


def test_percentage_efficiency_is_refused_with_a_suggestion():
    check = validate_profile(GeneratorProfile(default_efficiency=82.0))
    issue = next(i for i in check.errors if i.code == "EFFICIENCY_OUT_OF_BOUNDS")
    assert "0.82" in issue.suggestion


def test_empty_front_stock_is_refused():
    check = validate_profile(GeneratorProfile(front_resistors=[]))
    assert "FRONT_INVENTORY_EMPTY" in codes(check.errors)


def test_duplicate_resistor_values_warn():
    profile = GeneratorProfile(
        front_resistors=[
            ResistorStockIn(ohms=50.0, units_per_stage=2),
            ResistorStockIn(ohms=50.0, units_per_stage=2),
        ]
    )
    check = validate_profile(profile)
    assert check.ok
    assert "RESISTOR_VALUE_DUPLICATED" in codes(check.warnings)


def test_reduced_setup_reports_that_it_cannot_reach_the_stated_range():
    """8 stages x 200 kV x 0.82 = 1312 kV, short of the stated 2400 kV."""
    check = validate_profile(GeneratorProfile(max_stages=8))
    assert check.ok
    notice = next(i for i in check.notices if i.code == "PROFILE_BELOW_STATED_RANGE")
    assert "1312" in notice.message


def test_profile_without_external_resistors_cannot_do_switching():
    check = validate_profile(GeneratorProfile())
    assert check.ok
    assert "PROFILE_CANNOT_DO_SWITCHING" in codes(check.warnings)


def test_adding_external_resistors_clears_the_switching_limitation():
    profile = GeneratorProfile(
        external_front_ohms=[12000.0], external_tail_ohms=[1200.0]
    )
    check = validate_profile(profile)
    assert "PROFILE_CANNOT_DO_SWITCHING" not in codes(check.warnings)
    assert check.reach.has_external_front
    assert check.reach.has_external_tail


# ---------------------------------------------------------------------------
# Merging a profile into an operator request
# ---------------------------------------------------------------------------


def test_operator_inherits_the_laboratory_defaults():
    profile = GeneratorProfile(
        max_stages=12, default_efficiency=0.78, default_stray_inductance_uh=25.0
    )
    operator = OperatorRequestIn(load_capacitance_pf=850.0, required_peak_kv=1425.0)

    resolved = merge_operator_request(operator, profile)
    assert resolved.generator.max_stages == 12
    assert resolved.circuit.expected_efficiency == pytest.approx(0.78)
    assert resolved.circuit.stray_inductance_uh == pytest.approx(25.0)


def test_operator_can_override_the_laboratory_defaults():
    profile = GeneratorProfile(default_efficiency=0.78, default_stray_inductance_uh=25.0)
    operator = OperatorRequestIn(
        load_capacitance_pf=850.0, expected_efficiency=0.85, stray_inductance_uh=10.0
    )

    resolved = merge_operator_request(operator, profile)
    assert resolved.circuit.expected_efficiency == pytest.approx(0.85)
    assert resolved.circuit.stray_inductance_uh == pytest.approx(10.0)


def test_operator_request_carries_no_generator_fields():
    """The split is the point: an operator cannot describe hardware at all."""
    assert "generator" not in OperatorRequestIn.model_fields
    assert "inventory" not in OperatorRequestIn.model_fields
    with pytest.raises(Exception):
        OperatorRequestIn(load_capacitance_pf=850.0, max_stages=15)


def test_the_admin_profile_decides_what_the_operator_can_request():
    """The same request is feasible on the full generator and refused on a reduced one."""
    operator = OperatorRequestIn(load_capacitance_pf=850.0, required_peak_kv=2000.0)

    full = validate_request(merge_operator_request(operator, GeneratorProfile()))
    assert full.ok

    reduced = validate_request(
        merge_operator_request(operator, GeneratorProfile(max_stages=10))
    )
    assert not reduced.ok
    issue = next(i for i in reduced.errors if i.code == "EXCEEDS_GENERATOR_CEILING")
    # 10 * 200 * 0.82 = 1640 kV
    assert "1640" in issue.message


def test_narrowing_the_inventory_reaches_the_operator():
    profile = GeneratorProfile(
        front_resistors=[ResistorStockIn(ohms=50.0, units_per_stage=0)]
    )
    operator = OperatorRequestIn(load_capacitance_pf=850.0)

    report = validate_request(merge_operator_request(operator, profile))
    assert not report.ok
    assert "RESISTOR_UNITS_INVALID" in codes(report.errors)
