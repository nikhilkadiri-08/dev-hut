"""Frontend-to-backend integration tests against the real ASGI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import config, store
from backend.main import app

client = TestClient(app)


@pytest.fixture
def isolated_profile(tmp_path, monkeypatch):
    """Keep admin saves out of backend/data while tests run."""
    monkeypatch.setattr(store, "PROFILE_PATH", tmp_path / "generator_profile.json")
    yield

WORKBOOK_PAYLOAD = {
    "test_object_type_id": "workbook_example",
    "impulse_type": "lightning",
    "required_peak_kv": 1425.0,
    "circuit": {
        "load_capacitance_pf": 850.0,
        "divider_capacitance_pf": 500.0,
        "stray_capacitance_pf": 150.0,
        "stray_inductance_uh": 18.5,
        "expected_efficiency": 0.82,
    },
}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_drives_the_form():
    payload = client.get("/api/metadata").json()

    assert payload["generator"]["max_stages"] == 15
    assert payload["generator"]["max_test_voltage_kv"] == 2400.0
    assert {item["impulse_type"] for item in payload["impulse_types"]} == {
        "lightning",
        "switching",
    }
    assert any(item["id"] == "workbook_example" for item in payload["test_objects"])
    assert len(payload["inventory"]["front"]) == 10
    assert len(payload["inventory"]["tail"]) == 12
    assert payload["inventory"]["max_front_ohms_per_stage"] == 2000.0
    assert payload["inventory"]["max_tail_ohms_per_stage"] == 800.0


def test_metadata_reports_unimplemented_capabilities_honestly():
    capabilities = client.get("/api/metadata").json()["capabilities"]
    assert capabilities["validation"] == "implemented"
    assert capabilities["physics_engine"] == "not_implemented"
    assert capabilities["ml_residual_model"] == "not_implemented"


def test_metadata_exposes_the_assumption_register():
    payload = client.get("/api/metadata").json()
    keys = {entry["key"] for entry in payload["assumptions"]}
    assert "physics.crest_voltage" in keys
    assert "standard.edition" in keys
    assert payload["assumptions_requiring_attention"]
    assert all(
        entry["status"] != "supplied"
        for entry in payload["assumptions_requiring_attention"]
    )


def test_validate_accepts_the_workbook_example():
    response = client.post("/api/validate", json=WORKBOOK_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["normalized"]["derived"]["stages_required"] == 9


def test_validate_rejects_an_impossible_voltage():
    payload = {**WORKBOOK_PAYLOAD, "required_peak_kv": 3000.0}
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["normalized"] is None
    assert any(e["code"] == "PEAK_VOLTAGE_ABOVE_RANGE" for e in body["errors"])


def test_missing_required_field_returns_a_friendly_message():
    payload = {k: v for k, v in WORKBOOK_PAYLOAD.items() if k != "circuit"}
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert body["errors"][0]["message"] == "This value is required."
    assert body["errors"][0]["field"] == "circuit"


def test_non_numeric_input_returns_a_friendly_message():
    payload = {**WORKBOOK_PAYLOAD, "required_peak_kv": "quite high"}
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["message"] == "Enter a number."
    assert body["errors"][0]["field"] == "required_peak_kv"


def test_unsupported_impulse_type_is_rejected():
    payload = {**WORKBOOK_PAYLOAD, "impulse_type": "chopped"}
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["message"] == "Choose one of the listed options."


def test_unknown_field_is_rejected_rather_than_ignored():
    payload = {**WORKBOOK_PAYLOAD, "front_resistance_ohms": 50}
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "SCHEMA_EXTRA_FORBIDDEN"


def test_optimise_refuses_rather_than_fabricating_a_recommendation():
    response = client.post("/api/optimise", json=WORKBOOK_PAYLOAD)
    assert response.status_code == 501
    body = response.json()
    assert "not implemented" in body["detail"].lower()
    # It still returns the validation outcome so the UI has something useful to show.
    assert body["validation"]["ok"] is True


@pytest.mark.parametrize("impulse_type,voltage", [("lightning", 1425.0), ("switching", 1025.0)])
def test_both_impulse_types_validate_end_to_end(impulse_type, voltage):
    payload = {**WORKBOOK_PAYLOAD, "impulse_type": impulse_type, "required_peak_kv": voltage}
    body = client.post("/api/validate", json=payload).json()
    assert body["ok"] is True
    spec = config.IMPULSE_SPECS[impulse_type]
    assert body["normalized"]["target_front_time_us"] == spec.front_time.nominal


# ---------------------------------------------------------------------------
# Admin interface
# ---------------------------------------------------------------------------

OPERATOR_PAYLOAD = {
    "test_object_type_id": "workbook_example",
    "impulse_type": "lightning",
    "required_peak_kv": 1425.0,
    "load_capacitance_pf": 850.0,
    "divider_capacitance_pf": 500.0,
    "stray_capacitance_pf": 150.0,
}


def admin_defaults() -> dict:
    return client.get("/api/admin/profile").json()["stored"]["profile"]


def test_profile_starts_at_the_supplied_material_values(isolated_profile):
    body = client.get("/api/admin/profile").json()
    assert body["ok"] is True
    assert body["stored"]["is_factory_default"] is True
    assert body["stored"]["updated_at"] is None
    assert body["stored"]["profile"]["max_stages"] == 15
    assert body["reach"]["max_achievable_peak_kv"] == pytest.approx(2460.0)


def test_saving_a_profile_persists_it(isolated_profile):
    profile = {**admin_defaults(), "max_stages": 12}
    response = client.put(
        "/api/admin/profile", json={"profile": profile, "updated_by": "hv lab"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["stored"]["is_factory_default"] is False
    assert body["stored"]["updated_by"] == "hv lab"

    # A fresh read sees the saved value.
    assert client.get("/api/admin/profile").json()["stored"]["profile"]["max_stages"] == 12


def test_an_invalid_profile_is_not_persisted(isolated_profile):
    profile = {**admin_defaults(), "max_stage_voltage_kv": 250.0}
    body = client.put("/api/admin/profile", json={"profile": profile}).json()

    assert body["ok"] is False
    assert any(e["code"] == "STAGE_VOLTAGE_ABOVE_RATING" for e in body["errors"])
    # The stored profile is untouched.
    assert client.get("/api/admin/profile").json()["stored"]["is_factory_default"] is True


def test_reset_returns_to_the_supplied_material_values(isolated_profile):
    client.put("/api/admin/profile", json={"profile": {**admin_defaults(), "max_stages": 9}})
    assert client.get("/api/admin/profile").json()["stored"]["profile"]["max_stages"] == 9

    body = client.post("/api/admin/profile/reset").json()
    assert body["stored"]["is_factory_default"] is True
    assert body["stored"]["profile"]["max_stages"] == 15


# ---------------------------------------------------------------------------
# Operator interface
# ---------------------------------------------------------------------------


def test_operator_request_uses_the_stored_profile(isolated_profile):
    body = client.post("/api/test-request/validate", json=OPERATOR_PAYLOAD).json()
    assert body["ok"] is True
    assert body["normalized"]["derived"]["stages_required"] == 9


def test_operator_is_told_when_no_setup_has_been_saved(isolated_profile):
    body = client.post("/api/test-request/validate", json=OPERATOR_PAYLOAD).json()
    assert any(n["code"] == "PROFILE_NOT_CONFIGURED" for n in body["notices"])


def test_the_notice_disappears_once_a_setup_is_saved(isolated_profile):
    client.put("/api/admin/profile", json={"profile": admin_defaults(), "updated_by": "lab"})
    body = client.post("/api/test-request/validate", json=OPERATOR_PAYLOAD).json()
    assert not any(n["code"] == "PROFILE_NOT_CONFIGURED" for n in body["notices"])


def test_an_admin_change_constrains_the_operator(isolated_profile):
    """The headline behaviour of the split: the admin sets what is possible."""
    request = {**OPERATOR_PAYLOAD, "required_peak_kv": 2000.0}
    assert client.post("/api/test-request/validate", json=request).json()["ok"] is True

    client.put("/api/admin/profile", json={"profile": {**admin_defaults(), "max_stages": 10}})

    body = client.post("/api/test-request/validate", json=request).json()
    assert body["ok"] is False
    assert any(e["code"] == "EXCEEDS_GENERATOR_CEILING" for e in body["errors"])


def test_admin_efficiency_default_reaches_the_operator(isolated_profile):
    client.put(
        "/api/admin/profile", json={"profile": {**admin_defaults(), "default_efficiency": 0.7}}
    )
    body = client.post("/api/test-request/validate", json=OPERATOR_PAYLOAD).json()
    # 1425 / (200 * 0.7) = 10.2 -> 11 stages, against 9 at 82% efficiency.
    assert body["normalized"]["derived"]["stages_required"] == 11


def test_operator_cannot_submit_generator_fields(isolated_profile):
    payload = {**OPERATOR_PAYLOAD, "max_stages": 15}
    response = client.post("/api/test-request/validate", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "SCHEMA_EXTRA_FORBIDDEN"


def test_operator_load_capacitance_is_required(isolated_profile):
    payload = {k: v for k, v in OPERATOR_PAYLOAD.items() if k != "load_capacitance_pf"}
    response = client.post("/api/test-request/validate", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["message"] == "This value is required."
    assert body["errors"][0]["field"] == "load_capacitance_pf"
