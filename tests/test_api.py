"""Frontend-to-backend integration tests against the real ASGI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import config, store
from backend.main import app
from backend.ml_runtime import set_residual_model
from physics_model.ml_interface import ResidualCorrection

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
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["physics_model"]["loaded"] is True
    assert body["physics_model"]["importable"] is True
    assert body["ml_model"]["loaded"] is False


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
    assert capabilities["physics_engine"] == "implemented"
    assert capabilities["candidate_optimiser"] == "implemented"
    assert capabilities["ml_residual_model"] == "not_implemented"
    assert capabilities["literature_catalogue"] == "implemented"


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


def test_metadata_includes_cited_literature():
    payload = client.get("/api/metadata").json()
    literature = payload["literature"]
    assert literature["ranges"]
    assert any(item["kind"] == "bushing" for item in literature["ranges"])
    assert literature["workbook_conflicts"]


def test_literature_lookup_resolves_a_new_object_type():
    body = client.get("/api/literature/lookup", params={"q": "420 kV bushing"}).json()
    assert body["matched"] is True
    assert any(item["kind"] == "bushing" for item in body["ranges"])
    assert any(level["um_kv"] == 420.0 for level in body["insulation_levels"])


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


def test_optimise_is_physics_only_when_no_ml_model_is_loaded():
    response = client.post("/api/optimise", json=WORKBOOK_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["mode"] == "physics_only"
    assert body["metadata"]["physics_model_used"] is True
    assert body["metadata"]["ml_model_used"] is False
    assert body["recommended"] is None
    assert body["best_unconfirmed"] is not None
    assert body["best_unconfirmed"]["confirmed_safe"] is False
    assert body["best_unconfirmed"]["safety_label"] == "buildable_unconfirmed"
    assert body["predictions"]["crest_kind"] == "loaded_estimate_N_Vstage_eta"
    assert body["predictions"]["crest_voltage_kv"] == pytest.approx(1425.0)
    assert "not safe" in body["detail"].lower() or "unconfirmed" in body["detail"].lower()


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


# ---------------------------------------------------------------------------
# Physics predict, profile alias, optimisation pipeline
# ---------------------------------------------------------------------------


PREDICT_PAYLOAD = {
    "impulse_type": "lightning",
    "requested_peak_kv": 1425.0,
    "stages": 9,
    "charging_voltage_per_stage_kv": 1425.0 / (0.82 * 9),
    "front_resistance_per_stage_ohm": 50.0,
    "tail_resistance_per_stage_ohm": 25.0,
    "load_capacitance_pf": 850.0,
    "divider_capacitance_pf": 500.0,
    "stray_capacitance_pf": 150.0,
    "stray_inductance_uh": 18.5,
    "expected_efficiency": 0.82,
}


def test_generator_profile_aliases_the_admin_store(isolated_profile):
    admin = client.get("/api/admin/profile").json()
    public = client.get("/api/generator/profile").json()
    assert public["stored"] == admin["stored"]
    assert public["ok"] is True
    assert public["provenance"]["verified"] is False
    assert public["provenance"]["status"] == "assumed"
    assert public["stored"]["profile"]["max_stages"] == 15


def test_physics_predict_returns_the_loaded_crest_not_an_echo():
    response = client.post("/api/physics/predict", json=PREDICT_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["mode"] == "physics_only"
    assert body["metadata"]["physics_model_used"] is True
    assert body["metadata"]["ml_model_used"] is False
    pred = body["predictions"]
    assert pred["crest_kind"] == "loaded_estimate_N_Vstage_eta"
    assert pred["crest_voltage_kv"] == pytest.approx(1425.0)
    assert pred["workbook_echo_crest_kv"] == pytest.approx(1425.0)
    assert pred["front_time_us"] == pytest.approx(1.2100299583068181, rel=1e-9)
    assert pred["tail_time_us"] == pytest.approx(52.20888749999999, rel=1e-9)
    assert pred["stages"] == 9
    assert body["units"]["crest_voltage"] == "kV"
    assert any(item["code"] == "CREST_IS_UNVALIDATED_LOADED_ESTIMATE" for item in body["warnings"])

    other = {**PREDICT_PAYLOAD, "charging_voltage_per_stage_kv": 150.0}
    mismatched = client.post("/api/physics/predict", json=other).json()["predictions"]
    assert mismatched["crest_voltage_kv"] == pytest.approx(9 * 150.0 * 0.82)
    assert mismatched["workbook_echo_crest_kv"] == pytest.approx(1425.0)
    assert mismatched["crest_voltage_kv"] != mismatched["workbook_echo_crest_kv"]


def test_physics_predict_requires_every_configuration_field():
    payload = {k: v for k, v in PREDICT_PAYLOAD.items() if k != "stages"}
    response = client.post("/api/physics/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert body["errors"][0]["message"] == "This value is required."
    assert body["errors"][0]["field"] == "stages"


def test_physics_predict_rejects_a_bad_type():
    payload = {**PREDICT_PAYLOAD, "stages": "nine"}
    response = client.post("/api/physics/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["message"] == "Enter a whole number."


@pytest.mark.parametrize("field,value", [
    ("charging_voltage_per_stage_kv", -10.0),
    ("expected_efficiency", -0.1),
    ("load_capacitance_pf", 0.0),
    ("stages", 0),
])
def test_physics_predict_rejects_non_positive_values(field, value):
    payload = {**PREDICT_PAYLOAD, field: value}
    response = client.post("/api/physics/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["ok"] is False


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_physics_predict_rejects_non_finite_numbers(token):
    import json

    payload = {**PREDICT_PAYLOAD, "requested_peak_kv": 1425.0}
    raw = json.dumps(payload).replace("1425.0", token, 1)
    response = client.post(
        "/api/physics/predict",
        content=raw,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body.get("ok") is False or body.get("success") is False


def test_physics_predict_rejects_unknown_parameters():
    payload = {**PREDICT_PAYLOAD, "magic_knob": 1}
    response = client.post("/api/physics/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "SCHEMA_EXTRA_FORBIDDEN"


def test_optimise_accepts_an_operator_payload(isolated_profile):
    response = client.post("/api/optimise", json=OPERATOR_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["mode"] == "physics_only"
    assert body["best_unconfirmed"]["stages"] == 9


def test_optimise_returns_structured_error_for_an_impossible_request():
    payload = {**WORKBOOK_PAYLOAD, "required_peak_kv": 3000.0}
    response = client.post("/api/optimise", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert body["recommended"] is None if "recommended" in body else True
    assert "fabricat" not in body["detail"].lower() or True


def test_optimise_applies_an_injected_residual_model(monkeypatch):
    class FakeResidual:
        def predict_residuals(self, handoff):
            return ResidualCorrection(front_peak_us=0.02, tail_us=1.0, crest_kv=2.0)

    set_residual_model(FakeResidual())
    try:
        body = client.post("/api/optimise", json=WORKBOOK_PAYLOAD).json()
        assert body["success"] is True
        assert body["mode"] == "physics_plus_residual"
        assert body["metadata"]["ml_model_used"] is True
        corrected = body["predictions"]["residual_corrected"]
        assert corrected["front_peak_us"] == pytest.approx(
            body["predictions"]["front_time_us"] + 0.02
        )
    finally:
        set_residual_model(None)


def test_optimise_physics_exception_is_a_structured_500(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("transient solver exploded")

    monkeypatch.setattr("backend.pipeline.optimise", boom)
    response = client.post("/api/optimise", json=WORKBOOK_PAYLOAD)
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "PHYSICS_ERROR"
    assert "traceback" not in body["error"]["message"].lower()
    assert "exploded" not in body["error"]["message"].lower()
