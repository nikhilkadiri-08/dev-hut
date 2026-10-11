"""ANN residual pipeline tests. Oracles are not the function under test alone."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.ml_runtime import set_residual_model
from backend.pipeline import apply_residual_waveforms, optimise_from_request
from physics_model.ann_model import (
    ARTIFACT_META,
    FEATURE_ORDER,
    MODEL_VERSION,
    NO_IMPROVEMENT_REASON,
    activation_decision,
    benchmark_weighted_knn,
    features_from_handoff,
    knn_distance_matrix,
    load_artifact,
    load_dataset_splits,
    reevaluate_artifact,
    train_ann,
)
from physics_model.calculations import predict
from physics_model.inputs import (
    WORKBOOK_EXAMPLE_CIRCUIT,
    WORKBOOK_EXAMPLE_INVENTORY,
    WORKBOOK_EXAMPLE_RATINGS,
    GeneratorConfiguration,
    ImpulseType,
)
from physics_model.ml_interface import ResidualCorrection, build_handoff, knn_distance
from tests.test_api import PREDICT_PAYLOAD, WORKBOOK_PAYLOAD

from backend.units import kv_to_v

client = TestClient(app)


def _workbook_prediction():
    configuration = GeneratorConfiguration(
        stages=9,
        charging_voltage_per_stage_v=kv_to_v(1425.0 / (0.82 * 9)),
        front_resistance_per_stage_ohm=50.0,
        tail_resistance_per_stage_ohm=25.0,
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
    )
    return predict(configuration, kv_to_v(1425.0))


def test_train_split_excludes_hidden_test_rows():
    splits = load_dataset_splits()
    assert set(splits["train"]["Split"].unique()) == {"Train"}
    assert (splits["train"]["Split"] == "Hidden Test").sum() == 0
    raw = pd.read_csv(splits["train"].attrs.get("path") if False else "data/synthetic_dataset.csv")
    hidden_ids = set(raw.loc[raw["Split"] == "Hidden Test", "ID"])
    train_ids = set(splits["train"]["ID"])
    assert hidden_ids.isdisjoint(train_ids)


def test_feature_order_matches_handoff_width():
    prediction = _workbook_prediction()
    vector = features_from_handoff(build_handoff(prediction))
    assert list(FEATURE_ORDER) == [
        "is_switching",
        "Test_kV",
        "Load_C_pF",
        "Divider_C_pF",
        "Stray_C_pF",
        "L_uH",
        "Efficiency",
        "Front_R_Stage",
        "Tail_R_Stage",
        "Physics_FrontPeak_us",
        "Physics_Tail_us",
        "Physics_Crest_kV",
    ]
    assert vector.shape == (len(FEATURE_ORDER),)


def test_tiny_training_run_never_sees_hidden_test():
    model, scores = train_ann(hidden=4, epochs=3, batch_size=128)
    assert model.train_split == "Train"
    assert model.feature_order == FEATURE_ORDER
    assert scores["n_hidden_test"] == 300
    assert "ann" in scores["hidden_test"]


def test_physics_only_when_artifact_missing(monkeypatch):
    set_residual_model(None)
    monkeypatch.setattr("backend.pipeline.get_residual_model", lambda: None)
    body = client.post("/api/physics/predict", json=PREDICT_PAYLOAD).json()
    assert body["mode"] == "physics_only"
    assert body["metadata"]["ml_model_used"] is False
    assert body["predictions"]["corrected_waveform"] is None


def test_corrected_crest_is_not_forced_to_the_request():
    prediction = _workbook_prediction()

    class ShiftCrest:
        def predict_residuals(self, handoff):
            return ResidualCorrection(front_peak_us=0.01, tail_us=0.4, crest_kv=7.5)

    bundle = apply_residual_waveforms(prediction, ShiftCrest())
    assert bundle["applied"] is True
    corrected = bundle["corrected_metrics"]["crest_kv"]
    requested = prediction.workbook_echo_voltage_kv
    assert corrected != pytest.approx(requested)
    assert corrected == pytest.approx(prediction.crest_voltage_kv + 7.5)


def test_ood_input_does_not_apply_ann():
    class AlwaysOn:
        def predict_residuals(self, handoff):
            return ResidualCorrection(front_peak_us=0.2, tail_us=1.0, crest_kv=5.0)

    payload = {
        **PREDICT_PAYLOAD,
        "requested_peak_kv": 80.0,
        "stages": 2,
        "charging_voltage_per_stage_kv": 50.0,
    }
    set_residual_model(AlwaysOn())
    try:
        body = client.post("/api/physics/predict", json=payload).json()
    finally:
        set_residual_model(None)
    assert body["mode"] == "physics_only"
    assert body["metadata"]["ml_model_used"] is False
    assert body["predictions"]["corrected_waveform"] is None
    assert any(item["code"] == "ANN_NOT_APPLIED_OOD" for item in body["warnings"])


def test_inventory_candidate_uses_stocked_ohms():
    body = client.post("/api/optimise", json=WORKBOOK_PAYLOAD).json()
    candidate = body["candidate"]
    front_stock = {item.ohms for item in WORKBOOK_EXAMPLE_INVENTORY.front}
    tail_stock = {item.ohms for item in WORKBOOK_EXAMPLE_INVENTORY.tail}
    assert candidate["front_resistance_per_stage_ohm"] in front_stock or any(
        abs(candidate["front_resistance_per_stage_ohm"] - n * ohm) < 1e-9
        for ohm in front_stock
        for n in range(1, 5)
    )
    assert candidate["tail_resistance_per_stage_ohm"] in tail_stock or any(
        abs(candidate["tail_resistance_per_stage_ohm"] - n * ohm) < 1e-9
        for ohm in tail_stock
        for n in range(1, 5)
    )


def test_optimise_returns_both_waveform_keys_and_mode():
    body = client.post("/api/optimise", json=WORKBOOK_PAYLOAD).json()
    assert body["mode"] == "physics_only"
    assert "physics_waveform" in body
    assert "corrected_waveform" in body
    assert body["physics_waveform"] is not None
    assert body["corrected_waveform"] is None


@pytest.mark.ann
def test_optimise_with_ann_returns_corrected_curve():
    model, _ = train_ann(hidden=6, epochs=8, batch_size=128)
    set_residual_model(model)
    try:
        body = client.post("/api/optimise", json=WORKBOOK_PAYLOAD).json()
    finally:
        set_residual_model(None)
    assert body["mode"] in {"physics_plus_ann", "physics_only"}
    assert "physics_waveform" in body
    assert "corrected_waveform" in body
    if body["mode"] == "physics_plus_ann":
        assert body["metadata"]["ml_model_used"] is True
        assert body["corrected_waveform"]["model_kind"] == "reduced_order_double_exponential"
        physics_crest = body["predictions"]["crest_voltage_kv"]
        corrected_crest = body["predictions"]["corrected_metrics"]["crest_kv"]
        assert corrected_crest == pytest.approx(physics_crest)
        assert body["corrected_waveform"]["metrics"]["crest_kv"] == pytest.approx(physics_crest)


def _constant_specialist(spec, value: float, *, active: bool, reason: str) -> None:
    spec.weights = tuple(np.zeros_like(weight) for weight in spec.weights)
    spec.biases = tuple(np.zeros_like(bias) for bias in spec.biases)
    spec.target_mean = value
    spec.target_scale = 1.0
    spec.active = active
    spec.reason = reason


def test_activation_gate_requires_a_strict_validation_win_and_blocks_crest():
    assert activation_decision(0.5, 0.5, "front_us") == (False, NO_IMPROVEMENT_REASON)
    assert activation_decision(0.4, 0.5, "tail_us")[0] is True
    assert activation_decision(0.1, 1.0, "crest_kv")[0] is False


def test_inactive_head_does_not_change_that_output():
    model, _ = train_ann(hidden=4, epochs=2, batch_size=128)
    for spec in model.specialists:
        if spec.output_name == "front_us":
            _constant_specialist(spec, 0.2, active=False, reason=NO_IMPROVEMENT_REASON)
        elif spec.output_name == "tail_us":
            _constant_specialist(
                spec, 0.8, active=True, reason="validation MAE strictly better than a zero residual"
            )
        else:
            _constant_specialist(spec, 9.0, active=False, reason="crest diagnostic only")
    prediction = _workbook_prediction()
    handoff = build_handoff(prediction)
    assert handoff.in_training_domain is True
    residuals = model.predict_residuals(handoff)
    assert residuals.front_peak_us == 0.0
    assert residuals.crest_kv == 0.0
    assert residuals.tail_us == pytest.approx(0.8)
    bundle = apply_residual_waveforms(prediction, model)
    assert bundle["applied"] is True
    assert bundle["mode"] == "physics_plus_ann"
    assert bundle["corrected_metrics"]["front_time_us"] == pytest.approx(prediction.front_time_us)
    assert bundle["corrected_metrics"]["tail_time_us"] == pytest.approx(prediction.tail_time_us + 0.8)
    assert bundle["corrected_metrics"]["crest_kv"] == pytest.approx(prediction.crest_voltage_kv)
    assert bundle["corrected_waveform"]["metrics"]["crest_kv"] == pytest.approx(
        prediction.crest_voltage_kv
    )


def test_failed_validation_gate_is_not_applied():
    model, scores = train_ann(hidden=4, epochs=2, batch_size=128)
    for spec in model.specialists:
        if spec.validation_mae >= spec.zero_residual_mae or spec.output_name == "crest_kv":
            assert spec.active is False
    for spec in model.specialists:
        spec.active = False
        spec.reason = NO_IMPROVEMENT_REASON
    prediction = _workbook_prediction()
    residuals = model.predict_residuals(build_handoff(prediction))
    assert residuals.front_peak_us == 0.0
    assert residuals.tail_us == 0.0
    assert residuals.crest_kv == 0.0
    bundle = apply_residual_waveforms(prediction, model)
    assert bundle["applied"] is False
    assert bundle["mode"] == "physics_only"
    assert bundle["reason"] == "NO_ACTIVE_ANN_HEAD"
    assert bundle["corrected_waveform"] is None
    assert "ann" in scores["hidden_test"]


def test_shipped_lightning_example_does_not_move_loaded_crest():
    model = load_artifact()
    assert model.version == MODEL_VERSION
    assert model.applies_crest(ImpulseType.LIGHTNING) is False
    set_residual_model(model)
    try:
        body = client.post("/api/optimise", json=WORKBOOK_PAYLOAD).json()
    finally:
        set_residual_model(None)
    assert body["success"] is True
    physics_crest = body["predictions"]["crest_voltage_kv"]
    physics_front = body["predictions"]["front_time_us"]
    physics_tail = body["predictions"]["tail_time_us"]
    heads = body["metadata"]["ann_heads"]
    assert heads["crest_kv"]["applied"] is False
    if body["mode"] == "physics_only":
        assert body["corrected_waveform"] is None
        assert body["metadata"]["residual_reason"] == "NO_ACTIVE_ANN_HEAD"
    else:
        assert body["mode"] == "physics_plus_ann"
        corrected = body["predictions"]["corrected_metrics"]
        assert corrected["crest_kv"] == pytest.approx(physics_crest)
        assert body["corrected_waveform"]["metrics"]["crest_kv"] == pytest.approx(physics_crest)
        if heads["front_us"]["applied"] is False:
            assert corrected["front_time_us"] == pytest.approx(physics_front)
        if heads["tail_us"]["applied"] is False:
            assert corrected["tail_time_us"] == pytest.approx(physics_tail)


def test_hidden_test_metadata_matches_fresh_evaluation():
    meta = json.loads(ARTIFACT_META.read_text(encoding="utf-8"))
    fresh = reevaluate_artifact()
    for split in ("validation", "hidden_test"):
        for kind in ("ann", "zero_residual", "weighted_knn_k7"):
            for output in ("front_us", "tail_us", "crest_kv"):
                for metric in ("mae", "rmse"):
                    assert fresh[split][kind][output][metric] == pytest.approx(
                        meta["scores"][split][kind][output][metric]
                    )
    assert meta["synthetic_only"] is True
    assert meta["laboratory_accuracy"] is False
    assert meta["crest_head_applied"] is False
    assert meta["version"] == MODEL_VERSION
    for record in meta["specialists"]:
        if record["output"] == "crest_kv":
            assert record["active"] is False
        if record["validation_mae"] >= record["zero_residual_validation_mae"]:
            assert record["active"] is False
        if (
            record["output"] != "crest_kv"
            and record["hidden_test_mae"] is not None
            and record["hidden_test_mae"] >= record["zero_residual_hidden_test_mae"]
        ):
            assert record["active"] is False


def test_stale_artifact_version_does_not_load(tmp_path: Path):
    source = ARTIFACT_META.parent
    meta = json.loads((source / "ann_residual.meta.json").read_text(encoding="utf-8"))
    meta["version"] = "ann_residual_v1"
    (tmp_path / "ann_residual.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "ann_residual.npz").write_bytes((source / "ann_residual.npz").read_bytes())
    with pytest.raises(ValueError, match="stale ANN artifact version"):
        load_artifact(tmp_path)


def test_knn_benchmark_matches_scalar_distance_on_a_sample():
    splits = load_dataset_splits()
    query = splits["validation"].head(1)
    train = splits["train"]
    matrix = knn_distance_matrix(query, train)
    row = query.to_dict(orient="records")[0]
    features = _row_features(row)
    scalar = np.array(
        [knn_distance(features, _row_features(ref)) for ref in train.to_dict(orient="records")],
        dtype=np.float64,
    )
    assert matrix.shape == (1, len(train))
    assert matrix[0] == pytest.approx(scalar)


def _row_features(row: dict):
    from physics_model.inputs import ImpulseType
    from physics_model.ml_interface import MLFeatures

    return MLFeatures(
        impulse_type=ImpulseType(str(row["Impulse_Type"])),
        test_kv=float(row["Test_kV"]),
        load_c_pf=float(row["Load_C_pF"]),
        divider_c_pf=float(row["Divider_C_pF"]),
        stray_c_pf=float(row["Stray_C_pF"]),
        l_uh=float(row["L_uH"]),
        efficiency=float(row["Efficiency"]),
        front_r_stage=float(row["Front_R_Stage"]),
        tail_r_stage=float(row["Tail_R_Stage"]),
    )


def test_knn_benchmark_exists_and_is_not_called_by_optimise():
    assert callable(benchmark_weighted_knn)
    source = inspect.getsource(optimise_from_request)
    assert "benchmark_weighted_knn" not in source
    assert "weighted_knn" not in source
    pipeline_src = inspect.getsource(apply_residual_waveforms)
    assert "benchmark_weighted_knn" not in pipeline_src


def test_ann_training_history_endpoint_is_indexed_by_epoch():
    response = client.get("/api/ann/training")
    assert response.status_code == 200
    body = response.json()
    assert body["synthetic_only"] is True
    assert body["laboratory_accuracy"] is False
    assert body["weights_recorded_per_epoch"] is True
    assert body["epoch_count"] > 0
    assert body["active_heads"]["crest_kv"]["Lightning"] is False
    assert body["active_heads"]["crest_kv"]["Switching"] is False
    assert len(body["heads"]) == 6
    for head in body["heads"]:
        history = head["history"]
        assert head["epochs_executed"] == len(history)
        assert len(history) > 0
        assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
        assert head["architecture"]["input_features"] == list(FEATURE_ORDER)
        assert head["architecture"]["hidden_sizes"]
        assert head["architecture"]["output_size"] == 1
        assert 1 <= head["best_epoch"] <= len(history)
        best = history[head["best_epoch"] - 1]
        assert best["val_mae"] == pytest.approx(min(row["val_mae"] for row in history))
        for key in ("train_loss", "val_loss", "train_mae", "val_mae"):
            assert isinstance(history[0][key], float)


def test_training_history_stores_a_weight_snapshot_per_epoch():
    body = client.get("/api/ann/training").json()
    moved = False
    for head in body["heads"]:
        history = head["history"]
        assert len(history) == head["epochs_executed"]
        assert len(history) > 0
        first = history[0]
        assert len(first["weights"]) == len(head["architecture"]["hidden_sizes"]) + 1
        assert len(first["biases"]) == len(first["weights"])
        assert len(first["weights"][0]) == head["architecture"]["input_size"]
        assert len(first["weights"][0][0]) == head["architecture"]["hidden_sizes"][0]
        if len(history) > 1 and first["weights"] != history[1]["weights"]:
            moved = True
        assert head["example"]["split"] == "Train"
        assert "not a live generator shot" in head["example"]["label"].lower()
    assert moved


def test_lightning_crest_has_hidden_to_hidden_links_that_change_each_epoch():
    body = client.get("/api/ann/training").json()
    crest = next(
        head
        for head in body["heads"]
        if head["output"] == "crest_kv" and head["impulse"] == "Lightning"
    )
    assert crest["architecture"]["input_size"] == 12
    assert crest["architecture"]["hidden_sizes"] == [16, 8]
    assert crest["architecture"]["hidden_activation"] == "tanh"
    history = crest["history"]
    assert len(history) >= 2
    first = history[0]["weights"]
    second = history[1]["weights"]
    assert len(first) == 3
    assert len(first[0]) == 12 and len(first[0][0]) == 16
    assert len(first[1]) == 16 and len(first[1][0]) == 8
    assert len(first[2]) == 8 and len(first[2][0]) == 1
    assert first != second
    single_hidden = [
        head
        for head in body["heads"]
        if len(head["architecture"]["hidden_sizes"]) == 1
    ]
    assert single_hidden
    for head in single_hidden:
        assert len(head["history"]) > 1
        assert len(head["history"][0]["weights"]) == 2
        assert head["history"][0]["weights"] != head["history"][1]["weights"]


def test_ml_status_reports_the_real_architecture_not_the_reference():
    from backend.ml_runtime import get_residual_model

    previous = get_residual_model()
    set_residual_model(load_artifact())
    try:
        body = client.get("/api/ml/status", params={"impulse": "lightning"}).json()
    finally:
        set_residual_model(previous)
    assert body["status"] == "ready"
    assert body["available"] is True
    assert body["synthetic_only"] is True
    assert body["laboratory_validated"] is False
    assert body["architecture_matches_reference"] is False
    assert body["reference"]["input_nodes"] == 16
    assert body["reference"]["hidden_layers"] == [32, 32]
    assert body["reference"]["hidden_activation"] == "relu"
    diagram = body["diagram"]
    assert diagram["input_count"] == 12
    assert diagram["shared_hidden_layers"] is False
    assert diagram["hidden_activation"] == "tanh"
    assert [head["output"] for head in diagram["heads"]] == ["front_us", "tail_us", "crest_kv"]
    for head in diagram["heads"]:
        assert head["hidden_activation"] == "tanh"
        assert head["layer_sizes"][0] == 12
        assert head["layer_sizes"][-1] == 1
        assert head["matrices"][0]["rows"] == 12
        assert len(head["matrices"][0]["normalized"]) == 12
        assert len(head["matrices"][0]["normalized"][0]) == head["matrices"][0]["cols"]
    assert diagram["heads"][0]["hidden_sizes"] != [32, 32] or diagram["input_count"] != 16


def test_ml_residuals_returns_backend_values_and_does_not_add_crest():
    from backend.ml_runtime import get_residual_model

    previous = get_residual_model()
    set_residual_model(load_artifact())
    try:
        body = client.post("/api/ml/residuals", json=WORKBOOK_PAYLOAD).json()
    finally:
        set_residual_model(previous)
    assert body["status"] == "finished"
    assert body["laboratory_validated"] is False
    assert body["architecture_matches_reference"] is False
    residuals = body["residuals"]
    assert set(residuals) == {"front_us", "tail_us", "crest_kv"}
    for item in residuals.values():
        assert math.isfinite(item["raw"])
        assert math.isfinite(item["added"])
    assert residuals["crest_kv"]["applied"] is False
    assert residuals["crest_kv"]["added"] == pytest.approx(0.0)
    assert body["corrected"]["crest_kv"] == pytest.approx(body["physics"]["crest_kv"])
    assert len(body["features"]) == 12
    assert body["features"][0]["name"] == "is_switching"


def test_ml_status_unavailable_without_a_model():
    from backend.ml_runtime import get_residual_model

    previous = get_residual_model()
    set_residual_model(None)
    try:
        body = client.get("/api/ml/status").json()
        residuals = client.post("/api/ml/residuals", json=WORKBOOK_PAYLOAD).json()
    finally:
        set_residual_model(previous)
    assert body["status"] == "unavailable"
    assert body["diagram"] is None
    assert body["architecture_matches_reference"] is False
    assert residuals["status"] == "unavailable"
    assert residuals["residuals"] is None
