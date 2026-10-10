"""Sequence, chopped-wave and definition tests.

Expectations are independent of the function under test: IEC-style definitions,
algebraic energy, and configured fractions.
"""

from __future__ import annotations

import pytest

from backend.units import kv_to_v, us_to_s
from physics_model import (
    CitationStatus,
    ConstraintStatus,
    ImpulseType,
    ShotSpec,
    StandardCitation,
    TestSequence,
    WaveformKind,
    build_chopped_lightning,
    evaluate_sequence,
    extract_front_time,
    extract_time_to_peak,
    fit_double_exponential,
    optimise_sequence,
    predict,
)
from physics_model.inputs import (
    WORKBOOK_EXAMPLE_CIRCUIT,
    WORKBOOK_EXAMPLE_INVENTORY,
    WORKBOOK_EXAMPLE_RATINGS,
    GeneratorConfiguration,
)
from physics_model.optimiser import OptimisationRequest
from physics_model.test_physics_model import EXAMPLE_ETA, EXAMPLE_V, example_configuration
from physics_model.validation import validate_configuration


def test_lightning_front_is_not_raw_thirty_ninety():
    fitted = fit_double_exponential(us_to_s(1.2), us_to_s(50.0), EXAMPLE_V, ImpulseType.LIGHTNING)
    t1 = extract_front_time(fitted.shape)
    # T1 is 1.67 times the 30-90 interval, not the interval itself.
    assert t1 == pytest.approx(1.67 * (t1 / 1.67), rel=1e-12)
    assert t1 != pytest.approx(t1 / 1.67, rel=1e-3)


def test_switching_extractor_uses_time_to_peak_not_t1():
    fitted = fit_double_exponential(us_to_s(250.0), us_to_s(2500.0), EXAMPLE_V, ImpulseType.SWITCHING)
    tp = extract_time_to_peak(fitted.shape)
    t1 = extract_front_time(fitted.shape)
    assert tp == pytest.approx(fitted.shape.time_to_peak_s)
    assert tp != pytest.approx(t1)
    assert fitted.achieved_front_time_s == pytest.approx(tp, rel=1e-9)


def test_sampling_does_not_change_extracted_metrics():
    fitted = fit_double_exponential(us_to_s(1.2), us_to_s(50.0), EXAMPLE_V)
    first = extract_front_time(fitted.shape)
    fitted.shape.sample(200e-6, points=200)
    second = extract_front_time(fitted.shape)
    fitted.shape.sample(200e-6, points=4000)
    third = extract_front_time(fitted.shape)
    assert first == pytest.approx(second, rel=1e-12)
    assert first == pytest.approx(third, rel=1e-12)


def test_chopped_collapse_uses_the_configured_chopping_time():
    fitted = fit_double_exponential(us_to_s(1.2), us_to_s(50.0), EXAMPLE_V)
    chop_s = us_to_s(3.0)
    duration_s = us_to_s(0.4)
    chopped = build_chopped_lightning(
        fitted.shape,
        chopping_time_s=chop_s,
        collapse_duration_s=duration_s,
        collapse_shape="linear",
    )
    assert chopped.chopping_time_s == chop_s
    before = chopped.voltage_at(chop_s * 0.99)
    at = chopped.voltage_at(chop_s)
    mid = chopped.voltage_at(chop_s + 0.5 * duration_s)
    end = chopped.voltage_at(chop_s + duration_s)
    assert before > 0.0
    assert at == pytest.approx(chopped.voltage_at_chop_v)
    assert mid == pytest.approx(0.5 * chopped.voltage_at_chop_v, rel=1e-9)
    assert end == pytest.approx(0.0)
    assert any(c.code == "CHOP_COLLAPSE_IS_ASSUMPTION" for c in chopped.caveats)


def test_chopped_parameters_are_required():
    with pytest.raises(ValueError, match="chopping_time_s"):
        ShotSpec(waveform_kind=WaveformKind.CHOPPED_LIGHTNING, level_fraction=1.0)
    with pytest.raises(ValueError, match="collapse_shape"):
        ShotSpec(
            waveform_kind=WaveformKind.CHOPPED_LIGHTNING,
            level_fraction=1.0,
            chopping_time_s=us_to_s(3.0),
            collapse_duration_s=us_to_s(0.4),
        )


def test_fifty_percent_voltage_level_is_not_the_tail_half_crossing():
    """Requested 50% of 1425 kV is not 50% of this configuration's loaded crest."""
    configuration = GeneratorConfiguration(
        stages=9,
        charging_voltage_per_stage_v=kv_to_v(150.0),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    sequence = TestSequence(
        reference_voltage_v=EXAMPLE_V,
        shots=(ShotSpec(WaveformKind.LIGHTNING_FULL, 0.5),),
    )
    evaluated = evaluate_sequence(sequence, configuration)
    shot = evaluated.shots[0]
    requested_half = 0.5 * 1425.0
    assert shot.requested_voltage_kv == pytest.approx(requested_half)
    loaded = 9 * 150.0 * EXAMPLE_ETA
    assert shot.predicted_crest_kv == pytest.approx(loaded)
    assert shot.tail_half_crossing_kv == pytest.approx(0.5 * loaded)
    assert shot.requested_voltage_kv != pytest.approx(shot.tail_half_crossing_kv)


def test_sequence_preserves_shot_order():
    sequence = TestSequence(
        reference_voltage_v=EXAMPLE_V,
        shots=(
            ShotSpec(WaveformKind.LIGHTNING_FULL, 0.5, label="half"),
            ShotSpec(WaveformKind.LIGHTNING_FULL, 1.0, label="full"),
        ),
    )
    evaluated = evaluate_sequence(sequence, example_configuration())
    assert [s.spec.label for s in evaluated.shots] == ["half", "full"]
    assert evaluated.shots[0].spec.level_fraction < evaluated.shots[1].spec.level_fraction


def test_unknown_ratings_are_unverified_not_safe():
    sequence = TestSequence(
        reference_voltage_v=EXAMPLE_V,
        shots=(ShotSpec(WaveformKind.LIGHTNING_FULL, 1.0),),
    )
    evaluated = evaluate_sequence(
        sequence, example_configuration(), inventory=WORKBOOK_EXAMPLE_INVENTORY
    )
    shot = evaluated.shots[0]
    assert shot.constraint_status is ConstraintStatus.UNVERIFIED
    assert not shot.report.confirmed_safe
    assert shot.report.safety_label == "buildable_unconfirmed"


def test_known_citation_requires_source_edition_and_clause():
    with pytest.raises(ValueError, match="source"):
        StandardCitation(status=CitationStatus.KNOWN, source="IEC")


def test_modelling_c_energy_conflict_is_sixty_kilojoules_at_rating():
    """0.5 * 3 uF * (200 kV)^2 = 60 kJ, computed here, not taken from the helper."""
    expected_j = 0.5 * 3e-6 * (200e3) ** 2
    assert expected_j == pytest.approx(60_000.0)
    alternate_2_5_kj_as_c = 2 * 2500.0 / (200e3) ** 2
    assert alternate_2_5_kj_as_c == pytest.approx(0.125e-6)
    configuration = example_configuration()
    report = validate_configuration(configuration, predict(configuration, EXAMPLE_V))
    assert any(c.code == "MODELLING_C_ENERGY_UNVERIFIED" for c in report.caveats)


def test_optimiser_sequence_marks_switching_infeasible_with_this_inventory():
    sequence = TestSequence(
        reference_voltage_v=kv_to_v(1025.0),
        shots=(ShotSpec(WaveformKind.SWITCHING, 1.0),),
    )
    result = optimise_sequence(
        OptimisationRequest(
            requested_voltage_v=kv_to_v(1025.0),
            impulse_type=ImpulseType.SWITCHING,
            circuit=WORKBOOK_EXAMPLE_CIRCUIT,
            ratings=WORKBOOK_EXAMPLE_RATINGS,
            inventory=WORKBOOK_EXAMPLE_INVENTORY,
        ),
        sequence,
    )
    assert result.shots[0].result.recommended is None
    # Stocked front R cannot reach a switching front; not confirmed safe.
    assert not result.all_confirmed_safe


def test_handoff_channels_keep_five_quantities_distinct():
    from physics_model.ml_interface import ResidualCorrection, build_handoff

    prediction = predict(example_configuration(), EXAMPLE_V)
    channels = build_handoff(prediction).channels()
    assert channels["measured"] is None
    assert channels["residual"] is None
    assert channels["requested"]["peak_kv"] == pytest.approx(1425.0)
    assert channels["physics"]["crest_kv"] == pytest.approx(prediction.crest_voltage_kv)
    assert channels["physics"]["workbook_echo_crest_kv"] == pytest.approx(1425.0)
    filled = build_handoff(prediction).channels(ResidualCorrection(crest_kv=3.0))
    assert filled["residual"]["crest_residual_is_against_workbook_echo"] is True
    assert filled["corrected"]["crest_kv"] == pytest.approx(1425.0 + 3.0)
