"""Frontend-to-backend integration tests against the real ASGI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.main import app

client = TestClient(app)

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
