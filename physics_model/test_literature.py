"""Tests for the open-literature catalogue.

Expectations come from the cited public sources recorded in
``data/literature/open_references.json``, not from calling the lookup and
echoing it back. The workbook's own H13 windows are the control group: the
literature layer must not silently replace them.
"""

from __future__ import annotations

import json

import pytest

from physics_model.inputs import IMPULSE_TARGETS, ImpulseType
from physics_model.literature import (
    LITERATURE_PATH,
    impulse_window,
    insulation_level_for_um,
    literature_impulse_target,
    load_catalogue,
    lookup,
    suggest_test_voltage_kv,
    typical_ranges,
    workbook_conflicts,
)


def _raw() -> dict:
    return json.loads(LITERATURE_PATH.read_text(encoding="utf-8"))


def test_catalogue_file_is_cited_json_not_code():
    raw = _raw()
    assert raw["impulse_windows"]
    assert raw["insulation_levels"]
    for window in raw["impulse_windows"]:
        assert window["sources"]
        for source in window["sources"]:
            assert source["title"]
            assert source["url"].startswith("http")


def test_lightning_literature_window_matches_public_iec_2010_citations():
    """Nord-IS / CIGRE citations: 1.2 us +/-30%, 50 us +/-20%, peak +/-3%."""
    window = impulse_window("lightning")
    assert window.front_time_us == 1.2
    assert window.front_min_us == pytest.approx(0.84)
    assert window.front_max_us == pytest.approx(1.56)
    assert window.tail_min_us == pytest.approx(40.0)
    assert window.tail_max_us == pytest.approx(60.0)
    assert window.crest_relative == pytest.approx(0.03)


def test_switching_literature_tail_is_sixty_percent_not_the_workbook_twenty():
    """MDPI Energies 2022 and the 2020 SI measuring-system paper: T2 +/-60%."""
    window = impulse_window("switching")
    assert window.front_time_us == 250.0
    assert window.front_min_us == pytest.approx(200.0)
    assert window.front_max_us == pytest.approx(300.0)
    assert window.tail_relative == pytest.approx(0.60)
    assert window.tail_min_us == pytest.approx(1000.0)
    assert window.tail_max_us == pytest.approx(4000.0)
    assert window.conflict_with_workbook

    workbook = IMPULSE_TARGETS[ImpulseType.SWITCHING]
    assert workbook.tail_fraction == pytest.approx(0.20)
    literature = literature_impulse_target(ImpulseType.SWITCHING)
    assert literature.tail_fraction == pytest.approx(0.60)
    assert literature.tail_min_s != workbook.tail_min_s


def test_literature_does_not_mutate_workbook_impulse_targets():
    """The ML feature row must keep using H13, or the residuals stop matching."""
    assert IMPULSE_TARGETS[ImpulseType.SWITCHING].tail_fraction == pytest.approx(0.20)
    assert IMPULSE_TARGETS[ImpulseType.LIGHTNING].front_min_s == pytest.approx(0.84e-6)


def test_1425_kv_is_a_listed_420_kv_lightning_level():
    """IS/IEC 60071-1:2006 Table 3, Um = 420 kV, public republication."""
    level = insulation_level_for_um(420.0)
    assert level is not None
    assert 1425.0 in level.lightning_impulse_kv
    assert 1050.0 in level.switching_impulse_kv
    suggested = suggest_test_voltage_kv(400.0, ImpulseType.LIGHTNING)
    assert suggested is not None
    assert 1425.0 in suggested


def test_unknown_voltage_class_is_not_interpolated():
    assert insulation_level_for_um(1100.0, tolerance_kv=20.0) is None
    assert suggest_test_voltage_kv(1100.0, ImpulseType.LIGHTNING) is None


def test_bushing_c1_range_is_the_published_examples():
    """370 pF (400 kV example) to 489 pF (420 kV OIP nameplate example)."""
    bushings = typical_ranges(kind="bushing", quantity="capacitance_pf")
    assert len(bushings) == 1
    assert bushings[0].contains(370.0)
    assert bushings[0].contains(489.0)
    assert not bushings[0].contains(1100.0)


def test_disc_capacitances_are_not_multiplied_into_a_string_load():
    """A 23-disc string is series. 23 * 40 pF is not a literature load capacitance."""
    discs = typical_ranges(kind="cap_and_pin_string", quantity="self_capacitance_pf")
    assert discs
    assert discs[0].contains(40.0)
    assert discs[0].contains(149.0)
    assert "series" in discs[0].note.lower()
    assert not discs[0].contains(850.0)


def test_lookup_finds_a_new_bushing_description():
    hit = lookup("420 kV RIP transformer bushing")
    assert hit.matched
    kinds = {item.kind for item in hit.ranges}
    assert "bushing" in kinds
    assert any(level.um_kv == 420.0 for level in hit.insulation_levels)


def test_lookup_finds_switching_and_reports_the_workbook_conflict():
    hit = lookup("switching impulse 250/2500")
    assert any(window.impulse_family == "switching" for window in hit.windows)
    conflicts = workbook_conflicts()
    assert any("60%" in item["detail"] for item in conflicts)


def test_lookup_miss_does_not_invent_a_number():
    hit = lookup("cryogenic superconducting fault limiter")
    assert not hit.matched
    assert hit.ranges == []
    assert any(c.code == "NO_LITERATURE_MATCH" for c in hit.caveats)


def test_empty_lookup_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        lookup("   ")


def test_catalogue_disclaimer_forbids_treating_literature_as_cpri_fact():
    catalogue = load_catalogue()
    lowered = catalogue.disclaimer.lower()
    assert "cpri" in lowered or "not" in lowered
    assert "override" in lowered or "never" in lowered
