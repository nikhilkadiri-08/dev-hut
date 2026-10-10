"""Tests for the ported physics model.

Where the expected values come from
-----------------------------------
No test in this file builds its expectation by calling the function it is testing.
Expectations come from one of four independent places:

1. **The workbook's own cached cell values.** ``WORKED_EXAMPLE_RESULTS`` is a
   transcription of what Excel computed, so comparing against it checks the port
   against software this project did not write.
2. **The 2000-row synthetic dataset**, generated from the same spreadsheet
   formulas before this package existed. Reproducing all 2000 rows is the broadest
   check available.
3. **Mathematical properties** that must hold regardless of the implementation:
   monotonicity, exact round-trips between a function and its algebraic inverse,
   limiting behaviour, and energy conservation across two different routes to the
   same quantity.
4. **Brute force**, for the optimiser. The fast search is checked against an
   exhaustive enumeration of the entire configuration space.

What is deliberately not tested
-------------------------------
Agreement with the physical generator. No measured waveform data was supplied, so
no test here is evidence of real-world accuracy, and the validation figures on the
workbook's 'Model Validation' sheet are not used as such: the "observed" values in
that dataset were synthesised by perturbing the physics baseline, so an R-squared
computed against them measures self-consistency, not accuracy.
"""

from __future__ import annotations

import math

import pytest

from backend.units import kv_to_v, pf_to_f, uf_to_f, uh_to_h, us_to_s
from physics_model import (
    IMPULSE_TARGETS,
    WORKBOOK_EXAMPLE_CIRCUIT,
    WORKBOOK_EXAMPLE_INVENTORY,
    WORKBOOK_EXAMPLE_RATINGS,
    CapacitorRatings,
    CheckStatus,
    DoubleExponential,
    GeneratorConfiguration,
    GeneratorRatings,
    ImpulseType,
    InstalledSetup,
    IntraStageTopology,
    OptimisationRequest,
    RankingWeights,
    ResidualCorrection,
    ResistorInventory,
    TestCircuit,
    Verdict,
    achievable_per_stage_values,
    apply_residuals,
    assess_capacitor_stress,
    bracketing_values,
    build_handoff,
    buildable_resistances,
    charging_voltage_per_stage,
    charging_voltage_trim,
    crest_voltage,
    ideal_marx_voltage,
    loaded_crest_estimate,
    extract_front_time,
    extract_tail_time,
    extract_time_to_peak,
    fit_double_exponential,
    front_time,
    front_time_floor,
    front_time_is_reachable,
    generator_capacitance,
    ideal_output_voltage,
    individual_capacitor_voltage,
    is_realisable,
    knn_distance,
    optimise,
    predict,
    required_stages,
    series_equivalent_capacitance,
    snap_to_grid,
    stage_utilisation,
    stored_energy,
    tail_time,
    theoretical_total_front_resistance,
    theoretical_total_tail_resistance,
    total_load_capacitance,
    training_domain,
    validate_configuration,
    workbook_echo_crest,
)
from physics_model.ml_interface import DATASET_FEATURE_COLUMNS, MLFeatures
from physics_model.targets import (
    WORKBOOK_BASELINE,
    timing_definition,
)
from physics_model.validation import _quality_objectives
from physics_model.workbook_reference import (
    WORKED_EXAMPLE_RESULTS,
    read_calculator_cells,
    read_dataset,
)

# Shorthand for the worked example, in SI.
EXAMPLE_V = kv_to_v(1425.0)
EXAMPLE_C2 = pf_to_f(1500.0)
EXAMPLE_L = uh_to_h(18.5)
EXAMPLE_STAGE_C = uf_to_f(3.0)
EXAMPLE_ETA = 0.82
EXAMPLE_STAGE_V_MAX = kv_to_v(200.0)


def example_configuration(
    stages: int = 9,
    front_ohms_per_stage: float = 50.0,
    tail_ohms_per_stage: float = 25.0,
    impulse_type: ImpulseType = ImpulseType.LIGHTNING,
    ratings: GeneratorRatings | None = None,
) -> GeneratorConfiguration:
    return GeneratorConfiguration(
        stages=stages,
        charging_voltage_per_stage_v=charging_voltage_per_stage(
            EXAMPLE_V, EXAMPLE_ETA, stages
        ),
        front_resistance_per_stage_ohm=front_ohms_per_stage,
        tail_resistance_per_stage_ohm=tail_ohms_per_stage,
        impulse_type=impulse_type,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=ratings or WORKBOOK_EXAMPLE_RATINGS,
    )


# --------------------------------------------------------------------------- #
# 1. Against the workbook's own cached cell values
# --------------------------------------------------------------------------- #


def test_transcribed_results_match_the_workbook_file():
    """Guard the literal oracle against drift in the workbook itself."""
    cells = read_calculator_cells()
    if cells is None:
        pytest.skip("reference workbook not present")
    for coordinate, expected in WORKED_EXAMPLE_RESULTS.items():
        assert cells[coordinate] == pytest.approx(expected, rel=1e-12), (
            f"{coordinate} in the workbook is {cells[coordinate]}, but this test "
            f"suite was written against {expected}"
        )


def test_total_load_capacitance_matches_cell_e7():
    result = total_load_capacitance(
        pf_to_f(850.0), pf_to_f(500.0), pf_to_f(150.0)
    )
    assert result * 1e12 == pytest.approx(WORKED_EXAMPLE_RESULTS["E7"], rel=1e-12)


def test_required_stages_matches_cell_e8():
    assert required_stages(EXAMPLE_V, EXAMPLE_STAGE_V_MAX, EXAMPLE_ETA) == int(
        WORKED_EXAMPLE_RESULTS["E8"]
    )


def test_charging_voltage_per_stage_matches_cell_e9():
    volts = charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, 9)
    assert volts / 1000.0 == pytest.approx(WORKED_EXAMPLE_RESULTS["E9"], rel=1e-12)


def test_stage_utilisation_matches_cell_e10():
    volts = charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, 9)
    assert stage_utilisation(volts, EXAMPLE_STAGE_V_MAX) == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E10"], rel=1e-12
    )


def test_generator_capacitance_matches_cell_e11():
    result = generator_capacitance(EXAMPLE_STAGE_C, 9)
    assert result * 1e6 == pytest.approx(WORKED_EXAMPLE_RESULTS["E11"], rel=1e-12)


def test_theoretical_front_resistance_matches_cells_e12_and_e13():
    total = theoretical_total_front_resistance(us_to_s(1.2), EXAMPLE_C2, EXAMPLE_L)
    assert total == pytest.approx(WORKED_EXAMPLE_RESULTS["E12"], rel=1e-12)
    assert total / 9 == pytest.approx(WORKED_EXAMPLE_RESULTS["E13"], rel=1e-12)


def test_theoretical_tail_resistance_matches_cells_e15_and_e16():
    total = theoretical_total_tail_resistance(
        us_to_s(50.0), uf_to_f(1.0 / 3.0), EXAMPLE_C2
    )
    assert total == pytest.approx(WORKED_EXAMPLE_RESULTS["E15"], rel=1e-12)
    assert total / 9 == pytest.approx(WORKED_EXAMPLE_RESULTS["E16"], rel=1e-12)


def test_front_time_matches_cell_e18():
    result = front_time(50.0 * 9, EXAMPLE_C2, EXAMPLE_L)
    assert result * 1e6 == pytest.approx(WORKED_EXAMPLE_RESULTS["E18"], rel=1e-12)


def test_tail_time_matches_cell_e19():
    result = tail_time(25.0 * 9, uf_to_f(1.0 / 3.0), EXAMPLE_C2)
    assert result * 1e6 == pytest.approx(WORKED_EXAMPLE_RESULTS["E19"], rel=1e-12)


def test_snap_to_grid_matches_cells_e14_and_e17():
    assert snap_to_grid(WORKED_EXAMPLE_RESULTS["E13"]) == WORKED_EXAMPLE_RESULTS["E14"]
    assert snap_to_grid(WORKED_EXAMPLE_RESULTS["E16"]) == WORKED_EXAMPLE_RESULTS["E17"]


def test_snap_to_grid_rounds_halves_up_like_excel():
    """Excel's ROUND is half-away-from-zero; Python's round() is half-to-even."""
    assert snap_to_grid(22.5) == 25.0
    assert round(22.5 / 5) * 5 == 20.0  # what the naive implementation would give


def test_crest_voltage_is_the_requested_voltage_unchanged():
    """Cell E20 is '=B7'. Documenting the tautology rather than hiding it."""
    assert crest_voltage(EXAMPLE_V) == EXAMPLE_V
    assert workbook_echo_crest(EXAMPLE_V) == EXAMPLE_V
    assert crest_voltage(EXAMPLE_V) / 1000.0 == WORKED_EXAMPLE_RESULTS["E20"]


# --------------------------------------------------------------------------- #
# 2. Against the 2000-row dataset
# --------------------------------------------------------------------------- #


def _dataset_rows():
    frame = read_dataset()
    if frame is None:
        pytest.skip("synthetic dataset not present")
    return frame


def test_front_time_reproduces_every_dataset_row():
    frame = _dataset_rows()
    worst = 0.0
    for row in frame.itertuples(index=False):
        c2 = pf_to_f(row.Load_C_pF + row.Divider_C_pF + row.Stray_C_pF)
        predicted = front_time(
            row.Front_R_Stage * row.Stages, c2, uh_to_h(row.L_uH)
        )
        worst = max(worst, abs(predicted * 1e6 / row.Physics_FrontPeak_us - 1.0))
    assert worst < 1e-12, f"largest relative disagreement was {worst:.3e}"


def test_tail_time_reproduces_every_dataset_row():
    frame = _dataset_rows()
    worst = 0.0
    for row in frame.itertuples(index=False):
        c2 = pf_to_f(row.Load_C_pF + row.Divider_C_pF + row.Stray_C_pF)
        c1 = generator_capacitance(EXAMPLE_STAGE_C, row.Stages)
        predicted = tail_time(row.Tail_R_Stage * row.Stages, c1, c2)
        worst = max(worst, abs(predicted * 1e6 / row.Physics_Tail_us - 1.0))
    assert worst < 1e-12, f"largest relative disagreement was {worst:.3e}"


def test_required_stages_reproduces_every_dataset_row():
    frame = _dataset_rows()
    for row in frame.itertuples(index=False):
        assert (
            required_stages(
                kv_to_v(row.Test_kV), EXAMPLE_STAGE_V_MAX, row.Efficiency
            )
            == row.Stages
        )


def test_charging_voltage_reproduces_every_dataset_row():
    frame = _dataset_rows()
    for row in frame.itertuples(index=False):
        volts = charging_voltage_per_stage(
            kv_to_v(row.Test_kV), row.Efficiency, row.Stages
        )
        assert volts / 1000.0 == pytest.approx(row.Charge_kV_Stage, rel=1e-12)


def test_dataset_crest_column_is_the_test_voltage_echoed_back():
    """Confirms the tautology is in the data too, not just the calculator sheet."""
    frame = _dataset_rows()
    assert (frame["Physics_Crest_kV"] - frame["Test_kV"]).abs().max() == 0.0


# --------------------------------------------------------------------------- #
# 3. Units and explicit conversion
# --------------------------------------------------------------------------- #


def test_lab_unit_constructor_converts_to_si():
    circuit = TestCircuit.from_lab_units(
        load_capacitance_pf=850.0,
        divider_capacitance_pf=500.0,
        stray_capacitance_pf=150.0,
        stray_inductance_uh=18.5,
        efficiency=0.82,
    )
    assert circuit.load_capacitance_f == pytest.approx(8.5e-10)
    assert circuit.stray_inductance_h == pytest.approx(1.85e-5)
    assert circuit.total_load_capacitance_f == pytest.approx(1.5e-9)


def test_impulse_targets_are_stored_in_seconds():
    lightning = IMPULSE_TARGETS[ImpulseType.LIGHTNING]
    assert lightning.front_time_s == pytest.approx(1.2e-6)
    assert lightning.tail_time_s == pytest.approx(50e-6)
    switching = IMPULSE_TARGETS[ImpulseType.SWITCHING]
    assert switching.front_time_s == pytest.approx(250e-6)
    assert switching.tail_time_s == pytest.approx(2500e-6)


def test_front_time_is_dimensionally_consistent_under_rescaling():
    """Scaling R and C by reciprocal factors must leave the time unchanged.

    A genuine dimensional check: the resistive term depends on the product R*C, so
    doubling R and halving C cannot change it. Also exercises the inductive term,
    which must be rescaled as L*C to match.
    """
    base = front_time(450.0, 1.5e-9, 18.5e-6)
    rescaled = front_time(900.0, 0.75e-9, 37.0e-6)
    assert rescaled == pytest.approx(base, rel=1e-15)


# --------------------------------------------------------------------------- #
# 4. Invalid inputs and boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"load_capacitance_pf": 0.0}, "load_capacitance_f"),
        ({"load_capacitance_pf": -10.0}, "load_capacitance_f"),
        ({"divider_capacitance_pf": -1.0}, "divider_capacitance_f"),
        ({"stray_inductance_uh": -1.0}, "stray_inductance_h"),
        ({"efficiency": 0.0}, "efficiency"),
        ({"efficiency": 1.5}, "efficiency"),
    ],
)
def test_test_circuit_rejects_impossible_values(kwargs, message):
    defaults = dict(
        load_capacitance_pf=850.0,
        divider_capacitance_pf=500.0,
        stray_capacitance_pf=150.0,
        stray_inductance_uh=18.5,
        efficiency=0.82,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        TestCircuit.from_lab_units(**defaults)


def test_zero_stray_inductance_is_allowed():
    """Zero inductance is physically idealised but not invalid, unlike zero C."""
    circuit = TestCircuit.from_lab_units(
        load_capacitance_pf=850.0,
        divider_capacitance_pf=500.0,
        stray_capacitance_pf=150.0,
        stray_inductance_uh=0.0,
        efficiency=0.82,
    )
    assert circuit.stray_inductance_h == 0.0


def test_required_stages_rejects_non_positive_efficiency():
    with pytest.raises(ValueError, match="efficiency must be positive"):
        required_stages(EXAMPLE_V, EXAMPLE_STAGE_V_MAX, 0.0)


def test_charging_voltage_rejects_zero_stages():
    with pytest.raises(ValueError, match="stages must be positive"):
        charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, 0)


def test_generator_capacitance_rejects_zero_stages():
    with pytest.raises(ValueError, match="stages must be positive"):
        generator_capacitance(EXAMPLE_STAGE_C, 0)


def test_ratings_reject_inverted_stage_range():
    with pytest.raises(ValueError, match="cannot exceed"):
        GeneratorRatings.from_lab_units(max_stages=4, min_stages=9)


def test_empty_inventory_is_rejected():
    with pytest.raises(ValueError, match="front resistor inventory is empty"):
        ResistorInventory.from_pairs(front=[], tail=[(25.0, 4)])


def test_double_exponential_rejects_front_slower_than_tail():
    with pytest.raises(ValueError, match="must be smaller than tau_tail"):
        DoubleExponential(1.0, 50e-6, 1.2e-6)


def test_fit_rejects_front_time_longer_than_tail_time():
    with pytest.raises(ValueError, match="no double-exponential impulse"):
        fit_double_exponential(us_to_s(60.0), us_to_s(50.0), 1.0)


def test_required_stages_is_exactly_at_the_boundary():
    """A request needing exactly n stages must not round up to n+1."""
    exact = 9 * EXAMPLE_STAGE_V_MAX * EXAMPLE_ETA
    assert required_stages(exact, EXAMPLE_STAGE_V_MAX, EXAMPLE_ETA) == 9
    assert required_stages(exact + 1.0, EXAMPLE_STAGE_V_MAX, EXAMPLE_ETA) == 10


def test_stage_utilisation_is_exactly_one_at_the_rating():
    assert stage_utilisation(EXAMPLE_STAGE_V_MAX, EXAMPLE_STAGE_V_MAX) == 1.0


# --------------------------------------------------------------------------- #
# 5. Mathematical properties
# --------------------------------------------------------------------------- #


def test_front_resistance_inverse_round_trips_exactly():
    """E12 must undo E18. Checked across four decades of target front time."""
    for target_us in (0.9, 1.2, 1.56, 50.0, 250.0, 2500.0):
        target = us_to_s(target_us)
        resistance = theoretical_total_front_resistance(target, EXAMPLE_C2, EXAMPLE_L)
        assert front_time(resistance, EXAMPLE_C2, EXAMPLE_L) == pytest.approx(
            target, rel=1e-12
        )


def test_tail_resistance_inverse_round_trips_exactly():
    c1 = generator_capacitance(EXAMPLE_STAGE_C, 9)
    for target_us in (40.0, 50.0, 60.0, 2000.0, 2500.0):
        target = us_to_s(target_us)
        resistance = theoretical_total_tail_resistance(target, c1, EXAMPLE_C2)
        assert tail_time(resistance, c1, EXAMPLE_C2) == pytest.approx(
            target, rel=1e-12
        )


def test_front_time_is_strictly_increasing_in_front_resistance():
    """The monotonicity the bracketing search depends on."""
    times = [front_time(r, EXAMPLE_C2, EXAMPLE_L) for r in range(1, 2000, 7)]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_tail_time_is_strictly_increasing_in_tail_resistance():
    c1 = generator_capacitance(EXAMPLE_STAGE_C, 9)
    times = [tail_time(r, c1, EXAMPLE_C2) for r in range(1, 2000, 7)]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_required_stages_never_decreases_with_voltage():
    counts = [
        required_stages(kv_to_v(kv), EXAMPLE_STAGE_V_MAX, EXAMPLE_ETA)
        for kv in range(50, 2400, 25)
    ]
    assert all(b >= a for a, b in zip(counts, counts[1:]))


def test_front_time_tends_to_the_pure_rc_result_as_inductance_vanishes():
    """With L = 0 the formula must collapse to 1.67 * R * C exactly."""
    assert front_time(450.0, EXAMPLE_C2, 0.0) == pytest.approx(
        1.67 * 450.0 * EXAMPLE_C2, rel=1e-15
    )


def test_front_time_never_falls_below_the_inductive_floor():
    floor = front_time_floor(EXAMPLE_C2, EXAMPLE_L)
    for resistance in (0.0, 1.0, 10.0, 450.0):
        assert front_time(resistance, EXAMPLE_C2, EXAMPLE_L) >= floor
    assert front_time(0.0, EXAMPLE_C2, EXAMPLE_L) == pytest.approx(floor, rel=1e-15)


def test_unreachable_front_time_is_detected_not_clamped():
    """The workbook's MAX(1E-18,...) hides this; the port must not."""
    heavy_l = uh_to_h(500.0)
    floor = front_time_floor(EXAMPLE_C2, heavy_l)
    assert floor > us_to_s(1.2)
    assert not front_time_is_reachable(us_to_s(1.2), EXAMPLE_C2, heavy_l)

    configuration = GeneratorConfiguration(
        stages=9,
        charging_voltage_per_stage_v=charging_voltage_per_stage(
            EXAMPLE_V, EXAMPLE_ETA, 9
        ),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=TestCircuit.from_lab_units(
            load_capacitance_pf=850.0,
            divider_capacitance_pf=500.0,
            stray_capacitance_pf=150.0,
            stray_inductance_uh=500.0,
            efficiency=0.82,
        ),
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    prediction = predict(configuration, EXAMPLE_V)
    codes = {c.code for c in prediction.blocking}
    assert "FRONT_TIME_BELOW_INDUCTIVE_FLOOR" in codes


def test_ideal_output_voltage_inverts_charging_voltage():
    for stages in range(2, 16):
        volts = charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, stages)
        assert ideal_output_voltage(stages, volts, EXAMPLE_ETA) == pytest.approx(
            EXAMPLE_V, rel=1e-12
        )


def test_series_equivalent_capacitance_is_bounded_by_the_smaller_capacitance():
    c1 = generator_capacitance(EXAMPLE_STAGE_C, 9)
    equivalent = series_equivalent_capacitance(c1, EXAMPLE_C2)
    assert equivalent < min(c1, EXAMPLE_C2)
    # And it justifies the workbook's Rf*C2: within 0.5% when C1 >> C2.
    assert equivalent == pytest.approx(EXAMPLE_C2, rel=5e-3)


def test_acceptance_windows_are_centred_on_the_target():
    """The optimiser's proof of optimality depends on this.

    If a window were asymmetric about its target, minimising |achieved - target|
    would no longer maximise the margin, and the bracketing search would need to
    optimise a different quantity.
    """
    for target in IMPULSE_TARGETS.values():
        front_centre = 0.5 * (target.front_min_s + target.front_max_s)
        assert front_centre == pytest.approx(target.front_time_s, rel=1e-12)
        tail_centre = 0.5 * (target.tail_min_s + target.tail_max_s)
        assert tail_centre == pytest.approx(target.tail_time_s, rel=1e-12)


# --------------------------------------------------------------------------- #
# 6. Energy and capacitor stress
# --------------------------------------------------------------------------- #


def test_stored_energy_is_half_c_v_squared():
    assert stored_energy(2.0, 3.0) == pytest.approx(9.0)
    assert stored_energy(1e-6, 1000.0) == pytest.approx(0.5)


def test_stored_energy_scales_quadratically_with_voltage():
    base = stored_energy(uf_to_f(3.0), kv_to_v(100.0))
    doubled = stored_energy(uf_to_f(3.0), kv_to_v(200.0))
    assert doubled == pytest.approx(4.0 * base, rel=1e-15)


def test_stored_energy_rejects_negative_capacitance():
    with pytest.raises(ValueError, match="must not be negative"):
        stored_energy(-1.0, 100.0)


def test_marx_series_connection_conserves_stored_energy():
    """Two independent routes to the same energy must agree.

    On the charging side, n stage capacitors each hold 0.5*C_stage*Vc^2. On the
    discharge side, the series string is one capacitor of C_stage/n charged to
    n*Vc. These are different expressions; that they agree is a real property of
    the idealised series connection and a genuine check on both formulas.
    """
    for stages in range(2, 16):
        stage_v = charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, stages)
        charge_side = stages * stored_energy(EXAMPLE_STAGE_C, stage_v)
        discharge_side = stored_energy(
            generator_capacitance(EXAMPLE_STAGE_C, stages), stages * stage_v
        )
        assert charge_side == pytest.approx(discharge_side, rel=1e-12)


def test_capacitor_voltage_is_unknown_when_topology_is_unspecified():
    """The headline honesty requirement for this module."""
    voltage, reason = individual_capacitor_voltage(example_configuration())
    assert voltage is None
    assert "topology is unspecified" in reason


def test_capacitor_voltage_is_determined_once_topology_is_declared():
    stage_v = charging_voltage_per_stage(EXAMPLE_V, EXAMPLE_ETA, 9)

    parallel = example_configuration(
        ratings=GeneratorRatings.from_lab_units(
            intra_stage_topology=IntraStageTopology.PARALLEL
        )
    )
    voltage, _ = individual_capacitor_voltage(parallel)
    assert voltage == pytest.approx(stage_v)

    series = example_configuration(
        ratings=GeneratorRatings.from_lab_units(
            intra_stage_topology=IntraStageTopology.SERIES
        )
    )
    voltage, _ = individual_capacitor_voltage(series)
    assert voltage == pytest.approx(stage_v / 2.0)


def test_declared_topology_divides_capacitance_consistently():
    """Whichever topology is declared, the stage capacitance must come back to 3 uF."""
    parallel = GeneratorRatings.from_lab_units(
        intra_stage_topology=IntraStageTopology.PARALLEL
    )
    series = GeneratorRatings.from_lab_units(
        intra_stage_topology=IntraStageTopology.SERIES
    )
    assert parallel.capacitor_capacitance_f == pytest.approx(uf_to_f(1.5))
    assert series.capacitor_capacitance_f == pytest.approx(uf_to_f(6.0))
    # Two 1.5 uF in parallel and two 6 uF in series both give 3 uF, which is why
    # the supplied data cannot distinguish them.
    assert 2 * parallel.capacitor_capacitance_f == pytest.approx(uf_to_f(3.0))
    assert series.capacitor_capacitance_f / 2 == pytest.approx(uf_to_f(3.0))
    assert GeneratorRatings.from_lab_units().capacitor_capacitance_f is None


def test_missing_ratings_produce_unknown_not_pass():
    report = assess_capacitor_stress(example_configuration())
    unknown_names = {c.name for c in report.unknown}
    assert "Individual capacitor voltage" in unknown_names
    assert "Individual capacitor energy" in unknown_names
    assert "Peak discharge current" in unknown_names
    assert "Repetitive operation" in unknown_names
    assert not report.all_clear
    assert not report.failed


def test_supplied_ratings_enable_a_real_check_that_can_fail():
    """A check with data behind it must be able to fail, not just to pass."""

    def ratings_with(rated_voltage_kv: float) -> GeneratorRatings:
        return GeneratorRatings.from_lab_units(
            intra_stage_topology=IntraStageTopology.PARALLEL,
            capacitor_ratings=CapacitorRatings.from_lab_units(
                rated_voltage_kv=rated_voltage_kv
            ),
        )

    # Each capacitor sees 193 kV, so a 150 kV rating is exceeded.
    report = assess_capacitor_stress(example_configuration(ratings=ratings_with(150.0)))
    assert "Individual capacitor voltage" in {c.name for c in report.failed}

    report = assess_capacitor_stress(example_configuration(ratings=ratings_with(250.0)))
    passed = {c.name for c in report.checks if c.status is CheckStatus.PASS}
    assert "Individual capacitor voltage" in passed


def test_stress_report_energy_matches_the_workbook_worked_example():
    """Cross-check the energy against the workbook's own stage voltage."""
    report = assess_capacitor_stress(example_configuration())
    expected_stage_v = kv_to_v(WORKED_EXAMPLE_RESULTS["E9"])
    expected = 9 * 0.5 * uf_to_f(3.0) * expected_stage_v**2
    assert report.total_stored_energy_j == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# 7. Validation: safety separated from quality
# --------------------------------------------------------------------------- #


def test_workbook_example_passes_both_verdicts_except_for_unknown_ratings():
    configuration = example_configuration()
    report = validate_configuration(
        configuration, predict(configuration, EXAMPLE_V),
        inventory=WORKBOOK_EXAMPLE_INVENTORY,
    )
    assert report.quality_verdict is Verdict.COMPLIANT
    assert report.buildable
    assert not report.confirmed_safe
    assert report.safety_label == "buildable_unconfirmed"
    # Not SAFE, because the capacitor ratings were never supplied.
    assert report.safety_verdict is Verdict.INDETERMINATE
    assert report.unevaluated()
    assert any(c.code == "SAFETY_UNCONFIRMED" for c in report.caveats)


def test_overvoltage_configuration_is_unsafe_regardless_of_waveform_quality():
    """A hard constraint must not be redeemable by a good waveform."""
    configuration = GeneratorConfiguration(
        stages=2,
        charging_voltage_per_stage_v=kv_to_v(900.0),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    report = validate_configuration(
        configuration, predict(configuration, EXAMPLE_V)
    )
    assert report.safety_verdict is Verdict.UNSAFE
    assert not report.buildable
    assert any(c.code == "STAGE_VOLTAGE_WITHIN_RATING" for c in report.violations())


def test_crest_objective_is_the_loaded_estimate_not_a_vacuous_echo():
    configuration = example_configuration()
    prediction = predict(configuration, EXAMPLE_V)
    objectives = _quality_objectives(prediction, configuration.target)
    crest = next(o for o in objectives if o.code == "CREST_VOLTAGE_IN_WINDOW")
    assert not crest.vacuous
    assert crest.satisfied
    # Charging was sized as V_req/(eta*N), so the loaded estimate lands on the
    # request. That is computed, not tautological: a different V_stage misses.
    assert crest.margin == pytest.approx(1.0)

    report = validate_configuration(configuration, prediction)
    assert report.worst_margin < 1.0

    other = GeneratorConfiguration(
        stages=9,
        charging_voltage_per_stage_v=kv_to_v(100.0),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    other_prediction = predict(other, EXAMPLE_V)
    other_crest = next(
        o
        for o in _quality_objectives(other_prediction, other.target)
        if o.code == "CREST_VOLTAGE_IN_WINDOW"
    )
    assert other_prediction.crest_voltage_kv == pytest.approx(9 * 100.0 * EXAMPLE_ETA)
    assert other_prediction.workbook_echo_voltage_kv == pytest.approx(1425.0)
    assert not other_crest.satisfied


def test_worst_margin_is_computed_from_the_workbook_times():
    """Margin must follow from E18/E19 and the H13 windows, independently derived."""
    configuration = example_configuration()
    report = validate_configuration(
        configuration, predict(configuration, EXAMPLE_V)
    )
    front_margin = 1.0 - abs(WORKED_EXAMPLE_RESULTS["E18"] - 1.2) / 0.36
    tail_margin = 1.0 - abs(WORKED_EXAMPLE_RESULTS["E19"] - 50.0) / 10.0
    assert report.worst_margin == pytest.approx(
        min(front_margin, tail_margin), rel=1e-9
    )


def test_resistance_outside_the_inventory_is_rejected():
    configuration = example_configuration(front_ohms_per_stage=47.0)
    report = validate_configuration(
        configuration,
        predict(configuration, EXAMPLE_V),
        inventory=WORKBOOK_EXAMPLE_INVENTORY,
    )
    assert not report.buildable
    assert any(c.code == "FRONT_RESISTOR_IN_STOCK" for c in report.violations())


def test_achievable_values_are_series_multiples_and_deduplicated():
    values = achievable_per_stage_values(WORKBOOK_EXAMPLE_INVENTORY.front)
    assert values == sorted(set(values))
    assert 10.0 in values and 40.0 in values  # 1x and 4x the 10 ohm unit
    assert 2000.0 in values  # 4 x 500
    assert 45.0 not in values  # not reachable from the front stock
    assert is_realisable(40.0, WORKBOOK_EXAMPLE_INVENTORY.front)
    assert not is_realisable(45.0, WORKBOOK_EXAMPLE_INVENTORY.front)


# --------------------------------------------------------------------------- #
# 8. Waveform reconstruction
# --------------------------------------------------------------------------- #


def test_analytic_peak_time_matches_numerical_search():
    """The closed-form peak time must agree with a scan of the curve."""
    shape = DoubleExponential(1.0, 0.4e-6, 68.2e-6)
    coarse = max(
        (shape.voltage_at(t * 1e-9), t * 1e-9) for t in range(1, 5000)
    )[1]
    assert shape.time_to_peak_s == pytest.approx(coarse, abs=2e-9)


def test_fitted_waveform_reproduces_the_scalar_times_it_was_given():
    """Consistency between waveform.py and calculations.py.

    The fit targets the workbook's own front and tail times, and the extraction
    uses the IEC 30%/90% construction. Agreement shows the two modules share one
    definition of front time rather than drifting apart.
    """
    fitted = fit_double_exponential(
        us_to_s(WORKED_EXAMPLE_RESULTS["E18"]),
        us_to_s(WORKED_EXAMPLE_RESULTS["E19"]),
        EXAMPLE_V,
        ImpulseType.LIGHTNING,
    )
    assert fitted.converged
    assert extract_front_time(fitted.shape) * 1e6 == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E18"], rel=1e-8
    )
    assert extract_tail_time(fitted.shape) * 1e6 == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E19"], rel=1e-8
    )


def test_fitted_lightning_shape_is_near_the_classical_time_constants():
    """A 1.2/50 impulse classically has time constants near 0.4/68 us."""
    fitted = fit_double_exponential(
        us_to_s(1.2), us_to_s(50.0), 1.0, ImpulseType.LIGHTNING
    )
    assert fitted.shape.tau_front_s * 1e6 == pytest.approx(0.4, abs=0.15)
    assert fitted.shape.tau_tail_s * 1e6 == pytest.approx(68.0, abs=6.0)


def test_switching_fit_uses_time_to_peak_not_the_thirty_ninety_rule():
    fitted = fit_double_exponential(
        us_to_s(250.0), us_to_s(2500.0), EXAMPLE_V, ImpulseType.SWITCHING
    )
    assert fitted.converged
    assert extract_time_to_peak(fitted.shape) * 1e6 == pytest.approx(250.0, rel=1e-8)
    # The 30/90 construction gives a different number for the same curve, which is
    # exactly why the impulse type has to select the definition.
    assert extract_front_time(fitted.shape) * 1e6 != pytest.approx(250.0, rel=1e-3)


def test_waveform_peak_equals_the_supplied_peak():
    fitted = fit_double_exponential(
        us_to_s(1.2), us_to_s(50.0), EXAMPLE_V, ImpulseType.LIGHTNING
    )
    assert fitted.shape.peak_voltage_v == pytest.approx(EXAMPLE_V, rel=1e-10)


def test_waveform_carries_its_reconstruction_caveats():
    fitted = fit_double_exponential(us_to_s(1.2), us_to_s(50.0), EXAMPLE_V)
    codes = {c.code for c in fitted.caveats}
    assert "WAVEFORM_IS_RECONSTRUCTION" in codes
    assert "WAVEFORM_PEAK_IS_AN_INPUT" in codes


def test_waveform_sampling_returns_a_monotone_time_base_starting_at_zero():
    shape = DoubleExponential(1.0, 0.4e-6, 68.2e-6)
    t, v = shape.sample(200e-6, points=500)
    assert len(t) == len(v) == 500
    assert t[0] == 0.0
    assert all(b >= a for a, b in zip(t, t[1:]))
    assert v[0] == pytest.approx(0.0)
    assert v.max() == pytest.approx(shape.peak_voltage_v, rel=1e-3)


# --------------------------------------------------------------------------- #
# 9. The optimiser
# --------------------------------------------------------------------------- #


def example_request(**overrides) -> OptimisationRequest:
    base = dict(
        requested_voltage_v=EXAMPLE_V,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
        inventory=WORKBOOK_EXAMPLE_INVENTORY,
        weights=RankingWeights(
            margin=1.0,
            stage_change_penalty=0.0,
            front_swap_penalty=0.0,
            tail_swap_penalty=0.0,
            units_penalty=0.0,
        ),
    )
    base.update(overrides)
    return OptimisationRequest(**base)


def test_buildable_resistances_prefers_the_fewest_units():
    values, builds = buildable_resistances(WORKBOOK_EXAMPLE_INVENTORY.front)
    assert values == sorted(values)
    # 20 ohm is reachable as 2 x 10 or 1 x 20; the single unit must win.
    assert builds[20.0].units_in_series == 1
    assert builds[20.0].base_ohms == 20.0
    # 40 ohm is reachable as 4 x 10 or 2 x 20; the cheaper build must win.
    assert builds[40.0].units_in_series == 2
    assert builds[40.0].base_ohms == 20.0
    # 90 ohm is reachable only as 3 x 30, so there is nothing to choose.
    assert builds[90.0].units_in_series == 3
    assert builds[90.0].base_ohms == 30.0


def test_bracketing_returns_the_two_neighbours():
    values = [10.0, 20.0, 30.0, 40.0]
    assert bracketing_values(values, 25.0) == [20.0, 30.0]
    assert bracketing_values(values, 20.0) == [10.0, 20.0]
    assert bracketing_values(values, 5.0) == [10.0]
    assert bracketing_values(values, 100.0) == [40.0]
    assert bracketing_values([], 5.0) == []


def test_optimiser_reproduces_the_workbook_resistor_selection():
    """Independent agreement: the workbook got here by rounding to 5 ohm.

    This package searches the inventory instead, so arriving at the same 50/25
    ohm per stage is a meaningful cross-check rather than a restatement.
    Without component ratings the winner is buildable-unconfirmed, not recommended.
    """
    result = optimise(example_request())
    assert result.recommended is None
    best = result.best_unconfirmed
    assert best is not None
    assert best.presentation == "buildable_unconfirmed"
    assert best.stages == int(WORKED_EXAMPLE_RESULTS["E8"])
    assert best.configuration.front_resistance_per_stage_ohm == (
        WORKED_EXAMPLE_RESULTS["E14"]
    )
    assert best.configuration.tail_resistance_per_stage_ohm == (
        WORKED_EXAMPLE_RESULTS["E17"]
    )
    assert best.prediction.front_time_us == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E18"], rel=1e-12
    )
    assert best.prediction.tail_time_us == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E19"], rel=1e-12
    )
    assert any(c.code == "SEARCH_INVERTS_WORKBOOK_APPROXIMATION" for c in result.caveats)
    assert any(c.code == "NO_CONFIRMED_SAFE_CANDIDATE" for c in result.caveats)


def test_fast_search_finds_the_same_optimum_as_exhaustive_search():
    """The claim that the bracketing search is optimal, checked by brute force.

    Enumerates every stage count and every buildable resistor pair, scoring each
    with the same margin definition, and confirms the fast search's best margin is
    the global best. Also records how much work each approach needed.
    """
    request = example_request()
    result = optimise(request)
    assert result.best is not None

    front_values, _ = buildable_resistances(request.inventory.front)
    tail_values, _ = buildable_resistances(request.inventory.tail)
    minimum = required_stages(
        EXAMPLE_V, EXAMPLE_STAGE_V_MAX, WORKBOOK_EXAMPLE_CIRCUIT.efficiency
    )

    brute_best = -math.inf
    brute_argmax = None
    evaluated = 0
    for stages in range(minimum, WORKBOOK_EXAMPLE_RATINGS.max_stages + 1):
        stage_v = charging_voltage_per_stage(
            EXAMPLE_V, WORKBOOK_EXAMPLE_CIRCUIT.efficiency, stages
        )
        if stage_v > WORKBOOK_EXAMPLE_RATINGS.max_stage_voltage_v:
            continue
        for front_ohms in front_values:
            for tail_ohms in tail_values:
                evaluated += 1
                configuration = GeneratorConfiguration(
                    stages=stages,
                    charging_voltage_per_stage_v=stage_v,
                    front_resistance_per_stage_ohm=front_ohms,
                    tail_resistance_per_stage_ohm=tail_ohms,
                    impulse_type=ImpulseType.LIGHTNING,
                    circuit=WORKBOOK_EXAMPLE_CIRCUIT,
                    ratings=WORKBOOK_EXAMPLE_RATINGS,
                )
                prediction = predict(configuration, EXAMPLE_V)
                objectives = _quality_objectives(prediction, configuration.target)
                margin = min(o.margin for o in objectives if not o.vacuous)
                if margin > brute_best:
                    brute_best = margin
                    brute_argmax = (stages, front_ohms, tail_ohms)

    assert result.best.worst_margin == pytest.approx(brute_best, rel=1e-12)
    assert (
        result.best.stages,
        result.best.configuration.front_resistance_per_stage_ohm,
        result.best.configuration.tail_resistance_per_stage_ohm,
    ) == brute_argmax

    # And it got there with far less work.
    assert result.evaluations < evaluated / 50
    assert result.exhaustive_evaluations == evaluated


def test_optimiser_refuses_a_request_above_the_generator_ceiling():
    result = optimise(example_request(requested_voltage_v=kv_to_v(3000.0)))
    assert result.candidates == []
    assert any(c.code == "EXCEEDS_GENERATOR_CEILING" for c in result.caveats)


def test_optimiser_reports_switching_as_unbuildable_with_this_inventory():
    """A real limitation of the supplied inventory, not a model failure.

    Switching impulses need roughly 9,400-21,800 ohm/stage front resistance
    according to the dataset's own switching rows, but the stocked front resistors
    reach only 2,000 ohm/stage in series.
    """
    result = optimise(example_request(impulse_type=ImpulseType.SWITCHING))
    best = result.best
    assert best is not None
    # The closest buildable front resistance is the top of the stock, and it is
    # nowhere near enough, so the front time lands far outside its window.
    assert best.configuration.front_resistance_per_stage_ohm == max(
        achievable_per_stage_values(WORKBOOK_EXAMPLE_INVENTORY.front)
    )
    assert result.recommended is None


def test_keeping_the_installed_resistors_is_preferred_when_quality_is_equal():
    """The anti-climb behaviour: no physical change unless it buys something."""
    installed = InstalledSetup(
        stages=9,
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
    )
    result = optimise(
        example_request(
            installed=installed,
            weights=RankingWeights(),  # defaults, which include climb penalties
        )
    )
    best = result.best_unconfirmed
    assert result.recommended is None
    assert best is not None
    assert best.climb_operations == 0
    assert "No physical change needed" in best.climb_detail[0]


def test_a_climb_is_still_recommended_when_it_is_needed():
    installed = InstalledSetup(
        stages=9,
        front_resistance_per_stage_ohm=10.0,
        tail_resistance_per_stage_ohm=5.0,
    )
    result = optimise(example_request(installed=installed, weights=RankingWeights()))
    best = result.best_unconfirmed
    assert result.recommended is None
    assert best is not None
    assert best.climb_operations == 2
    assert any("front resistors" in d for d in best.climb_detail)
    assert any("tail resistors" in d for d in best.climb_detail)


def test_margin_is_never_sacrificed_to_avoid_a_climb_by_default():
    """Climb penalties break ties; they must not override waveform quality."""
    installed = InstalledSetup(
        stages=9,
        front_resistance_per_stage_ohm=10.0,
        tail_resistance_per_stage_ohm=5.0,
    )
    with_penalties = optimise(
        example_request(installed=installed, weights=RankingWeights())
    ).best_unconfirmed
    without_penalties = optimise(example_request()).best_unconfirmed
    assert with_penalties is not None and without_penalties is not None
    assert with_penalties.worst_margin == pytest.approx(
        without_penalties.worst_margin
    )


# --------------------------------------------------------------------------- #
# 10. The ML handoff
# --------------------------------------------------------------------------- #


def test_handoff_row_uses_the_datasets_own_column_names():
    frame = read_dataset()
    handoff = build_handoff(predict(example_configuration(), EXAMPLE_V))
    row = handoff.as_row()
    for column in DATASET_FEATURE_COLUMNS:
        assert column in row
    if frame is not None:
        for column in row:
            assert column in frame.columns, (
                f"{column} is not a column of the training dataset, so the ML "
                f"layer would have to adapt it"
            )


def test_handoff_values_match_the_workbook_example_in_laboratory_units():
    handoff = build_handoff(predict(example_configuration(), EXAMPLE_V))
    features = handoff.features
    assert features.impulse_type is ImpulseType.LIGHTNING
    assert features.test_kv == pytest.approx(1425.0)
    assert features.load_c_pf == pytest.approx(850.0)
    assert features.divider_c_pf == pytest.approx(500.0)
    assert features.stray_c_pf == pytest.approx(150.0)
    assert features.l_uh == pytest.approx(18.5)
    assert features.efficiency == pytest.approx(0.82)
    assert features.front_r_stage == 50.0
    assert features.tail_r_stage == 25.0
    assert handoff.physics_front_peak_us == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E18"], rel=1e-12
    )
    assert handoff.physics_crest_kv == pytest.approx(WORKED_EXAMPLE_RESULTS["E20"])
    assert handoff.predicted_crest_kv == pytest.approx(1425.0)


def test_knn_distance_is_zero_for_an_identical_row_and_grows_with_difference():
    base = MLFeatures(
        ImpulseType.LIGHTNING, 1425.0, 850.0, 500.0, 150.0, 18.5, 0.82, 50.0, 25.0
    )
    assert knn_distance(base, base) == 0.0
    further = MLFeatures(
        ImpulseType.LIGHTNING, 1500.0, 850.0, 500.0, 150.0, 18.5, 0.82, 50.0, 25.0
    )
    nearer = MLFeatures(
        ImpulseType.LIGHTNING, 1430.0, 850.0, 500.0, 150.0, 18.5, 0.82, 50.0, 25.0
    )
    assert knn_distance(base, nearer) < knn_distance(base, further)


def test_knn_distance_penalises_a_type_mismatch_by_exactly_one():
    lightning = MLFeatures(
        ImpulseType.LIGHTNING, 1425.0, 850.0, 500.0, 150.0, 18.5, 0.82, 50.0, 25.0
    )
    switching = MLFeatures(
        ImpulseType.SWITCHING, 1425.0, 850.0, 500.0, 150.0, 18.5, 0.82, 50.0, 25.0
    )
    assert knn_distance(lightning, switching) == pytest.approx(1.0)


def test_training_domain_uses_only_the_train_split():
    frame = read_dataset()
    if frame is None:
        pytest.skip("synthetic dataset not present")
    domain = training_domain()
    assert domain is not None
    train = frame[frame["Split"] == "Train"]
    low, high = domain["Test_kV"]
    assert low == pytest.approx(float(train["Test_kV"].min()))
    assert high == pytest.approx(float(train["Test_kV"].max()))
    # The full dataset is wider than the Train split, so using it would have
    # produced different bounds. This is the leak the restriction prevents.
    assert float(frame["Test_kV"].min()) <= low


def test_out_of_domain_configuration_is_flagged():
    configuration = example_configuration()
    handoff = build_handoff(predict(configuration, kv_to_v(2400.0)))
    if training_domain() is None:
        pytest.skip("synthetic dataset not present")
    assert not handoff.in_training_domain
    assert "Test_kV" in handoff.out_of_domain
    assert any(c.code == "OUTSIDE_ML_TRAINING_DOMAIN" for c in handoff.caveats)


def test_zero_residuals_leave_the_physics_baseline_untouched():
    """Correct behaviour before any model has been trained."""
    prediction = predict(example_configuration(), EXAMPLE_V)
    corrected = apply_residuals(prediction, ResidualCorrection())
    assert corrected["front_peak_us"] == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E18"], rel=1e-12
    )
    assert corrected["tail_us"] == pytest.approx(
        WORKED_EXAMPLE_RESULTS["E19"], rel=1e-12
    )
    assert corrected["crest_kv"] == pytest.approx(WORKED_EXAMPLE_RESULTS["E20"])


def test_residuals_are_added_the_way_the_workbook_adds_them():
    """Workbook H6-H8, checked against its own cached ML outputs."""
    prediction = predict(example_configuration(), EXAMPLE_V)
    cells = read_calculator_cells()
    if cells is None:
        pytest.skip("reference workbook not present")
    residuals = ResidualCorrection(
        front_peak_us=cells["H9"], tail_us=cells["H10"], crest_kv=cells["H11"]
    )
    corrected = apply_residuals(prediction, residuals)
    assert corrected["front_peak_us"] == pytest.approx(cells["H6"], rel=1e-9)
    assert corrected["tail_us"] == pytest.approx(cells["H7"], rel=1e-9)
    assert corrected["crest_kv"] == pytest.approx(cells["H8"], rel=1e-9)


def test_charging_voltage_trim_cancels_a_predicted_crest_error():
    """The no-climb correction: absorb voltage error at the control desk."""
    configuration = example_configuration()
    # Suppose the model predicts the generator will overshoot by 0.56%.
    trim = charging_voltage_trim(configuration, EXAMPLE_V, 1432.94658495185)
    assert trim.within_stage_rating
    assert trim.change_fraction < 0.0
    # Applying the trim must bring the ideal output back onto the request.
    corrected_output = ideal_output_voltage(
        configuration.stages, trim.trimmed_charging_voltage_v, EXAMPLE_ETA
    )
    overshoot = 1432.94658495185 / 1425.0
    assert corrected_output == pytest.approx(EXAMPLE_V / overshoot, rel=1e-12)
    assert "control-desk change" in trim.note


def test_charging_voltage_trim_refuses_to_exceed_the_stage_rating():
    configuration = example_configuration(stages=9)
    # A large predicted undershoot would need an impossible charging voltage.
    trim = charging_voltage_trim(configuration, EXAMPLE_V, 1000.0)
    assert not trim.within_stage_rating
    assert "cannot absorb this error" in trim.note


def test_handoff_warns_that_the_crest_residual_is_the_whole_model():
    handoff = build_handoff(predict(example_configuration(), EXAMPLE_V))
    assert any(
        c.code == "CREST_RESIDUAL_IS_THE_WHOLE_MODEL" for c in handoff.caveats
    )


# --------------------------------------------------------------------------- #
# 11. Honesty of the reported caveats
# --------------------------------------------------------------------------- #


def test_every_prediction_declares_the_crest_tautology_and_lack_of_validation():
    prediction = predict(example_configuration(), EXAMPLE_V)
    codes = {c.code for c in prediction.caveats}
    assert "CREST_IS_UNVALIDATED_LOADED_ESTIMATE" in codes
    assert "WORKBOOK_ECHO_IS_NOT_A_PREDICTION" in codes
    assert "NO_MEASURED_VALIDATION" in codes
    assert "FRONT_TIME_IS_EMPIRICAL" in codes


def test_validation_report_refuses_to_equate_prediction_with_compliance():
    configuration = example_configuration()
    report = validate_configuration(
        configuration, predict(configuration, EXAMPLE_V)
    )
    codes = {c.code for c in report.caveats}
    assert "PREDICTED_COMPLIANCE_IS_NOT_COMPLIANCE" in codes
    assert "TOLERANCE_EDITION_UNCONFIRMED" in codes
    assert "NO_CAPACITOR_RATINGS_SUPPLIED" in codes
    assert "SAFETY_UNCONFIRMED" in codes


# --------------------------------------------------------------------------- #
# 12. Intended prediction target, loaded crest, confirmed safety
# --------------------------------------------------------------------------- #


def ratings_with_evaluable_stress() -> GeneratorRatings:
    """Illustrative ratings that make every applicable hard check evaluable.

    Not datasheet values. Used only so the confirmed-safe path can be tested.
    """
    return GeneratorRatings.from_lab_units(
        intra_stage_topology=IntraStageTopology.PARALLEL,
        capacitor_ratings=CapacitorRatings.from_lab_units(
            rated_voltage_kv=250.0,
            rated_energy_j=1e5,
            rated_peak_current_a=2e4,
        ),
        resistor_energy_rating_j=1e6,
    )


def test_intended_timing_definitions_are_not_the_workbook_cells():
    lightning = timing_definition(ImpulseType.LIGHTNING)
    switching = timing_definition(ImpulseType.SWITCHING)
    assert "1.67" in lightning.front_or_peak
    assert "t90" in lightning.front_or_peak
    assert "time-to-peak" in switching.front_or_peak.lower()
    assert not lightning.workbook_matches_definition
    assert not switching.workbook_matches_definition
    assert "E20" in WORKBOOK_BASELINE["peak_voltage"]
    assert "not a prediction" in WORKBOOK_BASELINE["peak_voltage"]


def test_loaded_crest_is_independent_of_the_request():
    """Crest comes from N*V_stage*eta, not from echoing the requested voltage."""
    configuration = GeneratorConfiguration(
        stages=9,
        charging_voltage_per_stage_v=kv_to_v(150.0),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    prediction = predict(configuration, EXAMPLE_V)
    assert ideal_marx_voltage(9, kv_to_v(150.0)) == pytest.approx(kv_to_v(1350.0))
    assert loaded_crest_estimate(9, kv_to_v(150.0), EXAMPLE_ETA) == pytest.approx(
        kv_to_v(1350.0 * EXAMPLE_ETA)
    )
    assert prediction.crest_voltage_v == pytest.approx(kv_to_v(1350.0 * EXAMPLE_ETA))
    assert prediction.ideal_marx_voltage_v == pytest.approx(kv_to_v(1350.0))
    assert prediction.workbook_echo_voltage_v == pytest.approx(EXAMPLE_V)
    assert prediction.crest_voltage_v != prediction.workbook_echo_voltage_v


def test_confirmed_safe_requires_every_applicable_check():
    missing = example_configuration()
    missing_report = validate_configuration(
        missing, predict(missing, EXAMPLE_V), inventory=WORKBOOK_EXAMPLE_INVENTORY
    )
    assert missing_report.buildable
    assert not missing_report.confirmed_safe
    assert missing_report.safety_label == "buildable_unconfirmed"

    complete = example_configuration(ratings=ratings_with_evaluable_stress())
    complete_report = validate_configuration(
        complete, predict(complete, EXAMPLE_V), inventory=WORKBOOK_EXAMPLE_INVENTORY
    )
    assert complete_report.confirmed_safe
    assert complete_report.safety_verdict is Verdict.SAFE
    assert complete_report.safety_label == "safe"

    # Capacitor ratings without a resistor energy rating still cannot be safe.
    caps_only = example_configuration(
        ratings=GeneratorRatings.from_lab_units(
            intra_stage_topology=IntraStageTopology.PARALLEL,
            capacitor_ratings=CapacitorRatings.from_lab_units(
                rated_voltage_kv=250.0,
                rated_energy_j=1e5,
                rated_peak_current_a=2e4,
            ),
        )
    )
    caps_only_report = validate_configuration(
        caps_only, predict(caps_only, EXAMPLE_V), inventory=WORKBOOK_EXAMPLE_INVENTORY
    )
    assert caps_only_report.buildable
    assert not caps_only_report.confirmed_safe
    assert any(
        c.code == "STRESS_RESISTOR_ENERGY_DUTY" and c.indeterminate
        for c in caps_only_report.unevaluated()
    )


def test_recommended_requires_confirmed_safety():
    unconfirmed = optimise(example_request())
    assert unconfirmed.recommended is None
    assert unconfirmed.best_unconfirmed is not None
    assert unconfirmed.best_unconfirmed.presentation == "buildable_unconfirmed"

    confirmed = optimise(example_request(ratings=ratings_with_evaluable_stress()))
    assert confirmed.recommended is not None
    assert confirmed.recommended.confirmed_safe
    assert confirmed.recommended.presentation == "recommended"
    assert confirmed.best_unconfirmed is None
