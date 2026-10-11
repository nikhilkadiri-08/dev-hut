"""Bridge from API schemas to ``physics_model``. No formulas are re-derived here."""

from __future__ import annotations

import math
from typing import Any

from backend import store
from backend.ml_runtime import ResidualPredictor, get_residual_model
from backend.schemas import (
    GeneratorProfile,
    InventoryIn,
    PhysicsPredictIn,
    PredictRatingsIn,
    ShotSpecIn,
    TestRequestIn,
    TestSequenceIn,
)
from backend.units import f_to_pf, f_to_uf, kv_to_v, s_to_us, us_to_s, v_to_kv
from backend.validation import validate_request
from physics_model import (
    GeneratorConfiguration,
    GeneratorRatings,
    ImpulseType,
    OptimisationRequest,
    ResistorInventory,
    TestCircuit,
    apply_residuals,
    build_handoff,
    optimise,
    predict,
    series_equivalent_capacitance,
)
from physics_model.calculations import Caveat, PhysicsPrediction
from physics_model.ann_model import (
    ARCHITECTURE_DISCREPANCY,
    FEATURE_PRESENTATION,
    REFERENCE_ARCHITECTURE,
    AnnResidualModel,
    features_from_handoff,
    residuals_are_usable,
)
from physics_model.inputs import IMPULSE_TARGETS
from physics_model.waveform import (
    realize_from_targets,
    realize_impulse,
    serialize_realized_waveform,
)
from physics_model.ml_interface import ResidualCorrection
from physics_model.optimiser import Candidate, OptimisationResult, optimise_sequence
from physics_model.test_sequence import (
    CitationStatus,
    ShotSpec,
    StandardCitation,
    TestSequence,
    WaveformKind,
    evaluate_sequence,
    evaluate_shot,
)

_API_TO_IMPULSE = {
    "lightning": ImpulseType.LIGHTNING,
    "switching": ImpulseType.SWITCHING,
}

PREDICT_UNITS = {
    "front_time": "us",
    "tail_time": "us",
    "crest_voltage": "kV",
    "workbook_echo_crest": "kV",
    "ideal_marx_voltage": "kV",
    "requested_peak": "kV",
    "charging_voltage_per_stage": "kV",
    "generator_capacitance": "uF",
    "total_load_capacitance": "pF",
    "equivalent_capacitance": "pF",
    "stage_utilisation": "fraction",
    "stored_energy": "J",
}


class PipelineError(Exception):
    """Structured failure the API layer turns into HTTP 422 or 500."""

    def __init__(self, code: str, message: str, *, http_status: int = 422, details: list | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or []


def physics_import_status() -> dict[str, Any]:
    try:
        from physics_model import predict as _predict
        from physics_model import optimise as _optimise

        assert callable(_predict) and callable(_optimise)
        return {"loaded": True, "importable": True}
    except Exception as exc:  # pragma: no cover - import is present in this repo
        return {
            "loaded": False,
            "importable": False,
            "error": type(exc).__name__,
        }


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError("NON_FINITE", f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineError("NON_FINITE", f"{name} must be a finite number.")
    return number


def ratings_from_profile(profile: GeneratorProfile) -> GeneratorRatings:
    return GeneratorRatings.from_lab_units(
        max_stages=profile.max_stages,
        min_stages=profile.min_active_stages,
        max_stage_voltage_kv=profile.max_stage_voltage_kv,
        stage_capacitance_uf=profile.stage_capacitance_uf,
        capacitors_per_stage=profile.capacitors_per_stage,
    )


def ratings_from_predict(block: PredictRatingsIn) -> GeneratorRatings:
    return GeneratorRatings.from_lab_units(
        max_stages=block.max_stages,
        min_stages=block.min_active_stages,
        max_stage_voltage_kv=block.max_stage_voltage_kv,
        stage_capacitance_uf=block.stage_capacitance_uf,
        capacitors_per_stage=block.capacitors_per_stage,
    )


def inventory_from_api(inventory: InventoryIn) -> ResistorInventory:
    front = [(item.ohms, item.units_per_stage) for item in inventory.front]
    tail = [(item.ohms, item.units_per_stage) for item in inventory.tail]
    front.extend((ohms, 1) for ohms in inventory.external_front_ohms)
    tail.extend((ohms, 1) for ohms in inventory.external_tail_ohms)
    return ResistorInventory.from_pairs(
        front,
        tail,
        external_front_ohms=inventory.external_front_ohms,
        external_tail_ohms=inventory.external_tail_ohms,
    )


def circuit_from_request(request: TestRequestIn) -> TestCircuit:
    circuit = request.circuit
    return TestCircuit.from_lab_units(
        load_capacitance_pf=circuit.load_capacitance_pf,
        divider_capacitance_pf=circuit.divider_capacitance_pf,
        stray_capacitance_pf=circuit.stray_capacitance_pf,
        stray_inductance_uh=circuit.stray_inductance_uh,
        efficiency=circuit.expected_efficiency,
    )


def _caveats_to_warnings(caveats: list[Caveat]) -> list[dict[str, str]]:
    return [
        {"code": item.code, "message": item.message, "severity": item.severity.value}
        for item in caveats
    ]


def serialize_prediction(
    prediction: PhysicsPrediction,
    *,
    chopping_time_s: float | None = None,
    collapse_duration_s: float | None = None,
    collapse_shape: str | None = None,
) -> dict[str, Any]:
    c_eq = series_equivalent_capacitance(
        prediction.generator_capacitance_f, prediction.total_load_capacitance_f
    )
    try:
        realized = realize_impulse(
            prediction,
            chopping_time_s=chopping_time_s,
            collapse_duration_s=collapse_duration_s,
            collapse_shape=collapse_shape,
        )
    except ValueError as exc:
        raise PipelineError("INVALID_WAVEFORM_INPUT", str(exc)) from exc
    if not realized.consistency_ok:
        raise PipelineError(
            "WAVEFORM_INCONSISTENT",
            "Sampled waveform does not match the reported crest or timing "
            "within tolerance. No curve was fabricated.",
        )
    waveform = serialize_realized_waveform(realized)
    return {
        "front_time_us": prediction.front_time_us,
        "tail_time_us": prediction.tail_time_us,
        "crest_voltage_kv": prediction.crest_voltage_kv,
        "crest_kind": "loaded_estimate_N_Vstage_eta",
        "workbook_echo_crest_kv": prediction.workbook_echo_voltage_kv,
        "workbook_echo_label": (
            "Workbook E20 (= requested voltage). Not an independent crest prediction."
        ),
        "ideal_marx_voltage_kv": v_to_kv(prediction.ideal_marx_voltage_v),
        "requested_peak_kv": v_to_kv(prediction.requested_voltage_v),
        "stages": prediction.configuration.stages,
        "charging_voltage_per_stage_kv": v_to_kv(
            prediction.configuration.charging_voltage_per_stage_v
        ),
        "generator_capacitance_uf": f_to_uf(prediction.generator_capacitance_f),
        "total_load_capacitance_pf": f_to_pf(prediction.total_load_capacitance_f),
        "equivalent_capacitance_pf": f_to_pf(c_eq),
        "stage_utilisation": prediction.stage_utilisation,
        "stored_energy_j": prediction.stored_energy_j,
        "channels": build_handoff(prediction).channels(),
        "waveform": waveform,
        "extracted_front_time_us": waveform["metrics"]["front_time_us"],
        "extracted_tail_time_us": waveform["metrics"]["tail_time_us"],
        "peak_time_us": waveform["metrics"]["peak_time_us"],
        "waveform_warnings": waveform["warnings"],
    }


SYNTHETIC_TRAINING_WARNING = {
    "code": "SYNTHETIC_TRAINING_NOT_LABORATORY",
    "message": (
        "The ANN residual model was trained only on the synthetic Train split. "
        "Synthetic MAE/RMSE is not laboratory accuracy and is not a substitute "
        "for measured CPRI waveforms."
    ),
    "severity": "warning",
}

CREST_TAUTOLOGY_WARNING = {
    "code": "CREST_RESIDUAL_FIT_TO_E20_TAUTOLOGY",
    "message": (
        "The crest residual is not applied. Physics_Crest_kV equals Test_kV in "
        "the synthetic file (workbook E20), so the crest head only describes "
        "scatter around that tautology. Corrected crest stays on the loaded "
        "physics crest N*V_stage*eta. The ANN does not move it toward or away "
        "from the request."
    ),
    "severity": "warning",
}


def _interp_kv(source: dict | None, time_us: list[float]) -> list[float] | None:
    if source is None:
        return None
    import numpy as np

    return [
        float(v)
        for v in np.interp(time_us, source["time_us"], source["voltage_kv"])
    ]


def _requested_waveform(prediction: PhysicsPrediction) -> dict:
    target = IMPULSE_TARGETS[prediction.configuration.impulse_type]
    realized = realize_from_targets(
        target.front_time_s,
        target.tail_time_s,
        prediction.requested_voltage_v,
        prediction.configuration.impulse_type,
        model_id="requested_shape_not_a_measurement",
        peak_caveat_code="REQUESTED_SHAPE_IS_NOT_MEASURED",
        peak_caveat_message=(
            "This curve is a double exponential of the requested times and "
            "voltage. It is a reference overlay, not a measurement."
        ),
    )
    payload = serialize_realized_waveform(realized)
    payload["disclaimer"] = "requested shape, not a measurement"
    payload["assumptions"] = [
        "IEC-style requested front/tail and the requested peak voltage.",
        "Not a measured waveform and not a physics prediction.",
    ]
    return payload


def _quality_snapshot(front_us: float, tail_us: float, crest_kv: float, prediction: PhysicsPrediction) -> list[dict[str, Any]]:
    target = IMPULSE_TARGETS[prediction.configuration.impulse_type]
    crest_min, crest_max = target.crest_window_v(prediction.requested_voltage_v)
    checks = [
        (
            "FRONT_TIME_IN_WINDOW",
            front_us,
            s_to_us(target.front_min_s),
            s_to_us(target.front_max_s),
            "us",
        ),
        (
            "TAIL_TIME_IN_WINDOW",
            tail_us,
            s_to_us(target.tail_min_s),
            s_to_us(target.tail_max_s),
            "us",
        ),
        (
            "CREST_VOLTAGE_IN_WINDOW",
            crest_kv,
            v_to_kv(crest_min),
            v_to_kv(crest_max),
            "kV",
        ),
    ]
    return [
        {
            "code": code,
            "value": value,
            "minimum": lo,
            "maximum": hi,
            "unit": unit,
            "passed": lo <= value <= hi,
        }
        for code, value, lo, hi, unit in checks
    ]


def apply_residual_waveforms(
    prediction: PhysicsPrediction,
    model: ResidualPredictor | None,
    *,
    chopping_time_s: float | None = None,
    collapse_duration_s: float | None = None,
    collapse_shape: str | None = None,
) -> dict[str, Any]:
    """Physics waveform plus optional ANN-corrected waveform.

    Equipment safety is not recomputed here. A residual model cannot mark a
    candidate confirmed_safe.
    """
    physics = serialize_prediction(
        prediction,
        chopping_time_s=chopping_time_s,
        collapse_duration_s=collapse_duration_s,
        collapse_shape=collapse_shape,
    )
    warnings = list(physics.pop("waveform_warnings", []))
    requested = _requested_waveform(prediction)
    result = {
        "physics": physics,
        "corrected_waveform": None,
        "corrected_metrics": None,
        "residual_corrected": None,
        "requested_waveform": requested,
        "waveform_overlay": {
            "time_us": physics["waveform"]["time_us"],
            "physics_kv": physics["waveform"]["voltage_kv"],
            "corrected_kv": None,
            "requested_kv": _interp_kv(requested, physics["waveform"]["time_us"]),
            "units": {"time": "us", "voltage": "kV"},
            "labels": {
                "physics": "physics reduced-order V(t)",
                "corrected": "ANN-corrected reduced-order V(t)",
                "requested": "requested shape, not a measurement",
            },
        },
        "applied": False,
        "mode": "physics_only",
        "ml_model_used": False,
        "reason": None,
        "ann_heads": None,
        "warnings": warnings,
        "constraint_results": {
            "physics": _quality_snapshot(
                prediction.front_time_us,
                prediction.tail_time_us,
                prediction.crest_voltage_kv,
                prediction,
            ),
            "corrected": None,
        },
    }
    if model is None:
        result["reason"] = "NO_RESIDUAL_MODEL"
        return result

    handoff = build_handoff(prediction)
    result["warnings"].extend(_caveats_to_warnings(handoff.caveats))
    if not handoff.in_training_domain:
        result["reason"] = "OUTSIDE_ML_TRAINING_DOMAIN"
        result["warnings"].append(
            {
                "code": "ANN_NOT_APPLIED_OOD",
                "message": (
                    "The residual model was not applied because this configuration "
                    "is outside the synthetic Train domain. Response stays physics-only."
                ),
                "severity": "warning",
            }
        )
        return result

    try:
        residuals = model.predict_residuals(handoff)
    except Exception as exc:
        result["reason"] = "RESIDUAL_PREDICT_FAILED"
        result["warnings"].append(
            {
                "code": "ANN_PREDICT_FAILED",
                "message": f"Residual model failed ({exc}). Physics waveform retained.",
                "severity": "warning",
            }
        )
        return result
    if not isinstance(residuals, ResidualCorrection):
        result["reason"] = "INCOMPATIBLE_RESIDUAL"
        return result

    impulse = prediction.configuration.impulse_type
    crest_head_applied = True
    if isinstance(model, AnnResidualModel):
        result["ann_heads"] = model.public_head_status(impulse)
        if not model.applies_timing(impulse):
            result["reason"] = "NO_ACTIVE_ANN_HEAD"
            result["warnings"].append(
                {
                    "code": "ANN_HEADS_INACTIVE",
                    "message": model.physics_only_reason(impulse),
                    "severity": "warning",
                }
            )
            return result
        crest_head_applied = model.applies_crest(impulse)
        if not crest_head_applied:
            residuals = ResidualCorrection(
                front_peak_us=residuals.front_peak_us,
                tail_us=residuals.tail_us,
                crest_kv=0.0,
            )

    corrected_front = prediction.front_time_us + residuals.front_peak_us
    corrected_tail = prediction.tail_time_us + residuals.tail_us
    corrected_crest = prediction.crest_voltage_kv + residuals.crest_kv
    unusable = residuals_are_usable(
        corrected_front,
        corrected_tail,
        corrected_crest,
        residuals,
        prediction.configuration.impulse_type,
    )
    if unusable is not None:
        result["reason"] = "ABSURD_OR_UNUSABLE_RESIDUAL"
        result["warnings"].append(
            {
                "code": "ANN_RESIDUAL_REJECTED",
                "message": (
                    f"Residuals were not applied ({unusable}). "
                    "The physics waveform was retained."
                ),
                "severity": "warning",
            }
        )
        return result

    try:
        realized = realize_from_targets(
            us_to_s(corrected_front),
            us_to_s(corrected_tail),
            kv_to_v(corrected_crest),
            prediction.configuration.impulse_type,
            chopping_time_s=chopping_time_s,
            collapse_duration_s=collapse_duration_s,
            collapse_shape=collapse_shape,
            model_id=(
                "double_exp_ann_corrected_loaded_crest"
                if crest_head_applied
                else "double_exp_ann_corrected_timing_physics_crest"
            ),
            peak_caveat_code=(
                "WAVEFORM_PEAK_IS_LOADED_PLUS_SYNTHETIC_RESIDUAL"
                if crest_head_applied
                else "WAVEFORM_PEAK_IS_LOADED_CREST"
            ),
            peak_caveat_message=(
                "Peak is loaded crest N*V_stage*eta plus a synthetic crest "
                "residual trained on E20 tautology scatter. Not a learned transient."
                if crest_head_applied
                else (
                    "Peak is the loaded crest N*V_stage*eta. The crest residual "
                    "was not applied. Timing, if corrected, is a synthetic scalar "
                    "residual. Not a learned transient."
                )
            ),
        )
    except ValueError as exc:
        result["reason"] = "CORRECTED_FIT_FAILED"
        result["warnings"].append(
            {
                "code": "ANN_WAVEFORM_FIT_FAILED",
                "message": f"Corrected double-exponential fit failed ({exc}). Physics retained.",
                "severity": "warning",
            }
        )
        return result
    if not realized.consistency_ok:
        result["reason"] = "CORRECTED_WAVEFORM_INCONSISTENT"
        result["warnings"].append(
            {
                "code": "ANN_WAVEFORM_INCONSISTENT",
                "message": "Corrected samples disagreed with reported scalars. Physics retained.",
                "severity": "warning",
            }
        )
        return result

    corrected_wave = serialize_realized_waveform(realized)
    workbook_h8 = apply_residuals(prediction, residuals)
    result["applied"] = True
    result["ml_model_used"] = True
    result["mode"] = (
        "physics_plus_ann" if isinstance(model, AnnResidualModel) else "physics_plus_residual"
    )
    result["corrected_waveform"] = corrected_wave
    result["residual_corrected"] = workbook_h8
    result["corrected_metrics"] = {
        "front_time_us": corrected_front,
        "tail_time_us": corrected_tail,
        "crest_kv": corrected_crest,
        "crest_kind": (
            "loaded_estimate_plus_synthetic_crest_residual"
            if crest_head_applied and residuals.crest_kv != 0.0
            else "loaded_estimate_N_Vstage_eta"
        ),
        "crest_residual_applied": bool(crest_head_applied and residuals.crest_kv != 0.0),
        "extracted_front_time_us": corrected_wave["metrics"]["front_time_us"],
        "extracted_tail_time_us": corrected_wave["metrics"]["tail_time_us"],
        "peak_time_us": corrected_wave["metrics"]["peak_time_us"],
        "workbook_h8_crest_kv": workbook_h8["crest_kv"],
        "requested_peak_kv": v_to_kv(prediction.requested_voltage_v),
        "errors_vs_request": {
            "front_us": corrected_front - s_to_us(IMPULSE_TARGETS[prediction.configuration.impulse_type].front_time_s),
            "tail_us": corrected_tail - s_to_us(IMPULSE_TARGETS[prediction.configuration.impulse_type].tail_time_s),
            "crest_kv": corrected_crest - v_to_kv(prediction.requested_voltage_v),
        },
    }
    result["constraint_results"]["corrected"] = _quality_snapshot(
        corrected_front, corrected_tail, corrected_crest, prediction
    )
    result["waveform_overlay"]["corrected_kv"] = _interp_kv(
        corrected_wave, physics["waveform"]["time_us"]
    )
    result["warnings"].append(SYNTHETIC_TRAINING_WARNING)
    if crest_head_applied and residuals.crest_kv != 0.0:
        result["warnings"].append(
            {
                "code": "CREST_RESIDUAL_FIT_TO_E20_TAUTOLOGY",
                "message": (
                    "Physics_Crest_kV equals Test_kV in the synthetic file (workbook E20). "
                    "A crest residual was added to the loaded estimate. It is not forced "
                    "to the request, and it is not an independent crest measurement."
                ),
                "severity": "warning",
            }
        )
    else:
        result["warnings"].append(CREST_TAUTOLOGY_WARNING)
    if isinstance(model, AnnResidualModel):
        result["warnings"].append(
            {
                "code": "ANN_HEADS_APPLIED",
                "message": model.applied_message(impulse),
                "severity": "warning",
            }
        )
    result["warnings"].extend(corrected_wave.get("warnings", []))
    result["reason"] = None
    return result


_RESIDUAL_LABELS = {
    "front_us": ("Front", "us", "front_time_us"),
    "tail_us": ("Tail", "us", "tail_time_us"),
    "crest_kv": ("Crest", "kV", "crest_kv"),
}


def build_ann_inference(
    prediction: PhysicsPrediction,
    model: ResidualPredictor | None,
    bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Residuals for the diagram. Raw network outputs stay distinct from values that were added."""
    physics = {
        "front_time_us": prediction.front_time_us,
        "tail_time_us": prediction.tail_time_us,
        "crest_kv": prediction.crest_voltage_kv,
    }
    target = IMPULSE_TARGETS[prediction.configuration.impulse_type]
    requested = {
        "peak_kv": v_to_kv(prediction.requested_voltage_v),
        "front_time_us": s_to_us(target.front_time_s),
        "tail_time_us": s_to_us(target.tail_time_s),
    }
    base = {
        "synthetic_only": True,
        "laboratory_validated": False,
        "architecture_matches_reference": False,
        "reference": REFERENCE_ARCHITECTURE,
        "discrepancy": ARCHITECTURE_DISCREPANCY,
        "impulse": prediction.configuration.impulse_type.value,
        "physics": physics,
        "requested": requested,
        "residuals": None,
        "corrected": None,
        "features": None,
        "in_domain": None,
        "mode": None if bundle is None else bundle.get("mode"),
    }
    if model is None:
        return {
            **base,
            "status": "unavailable",
            "available": False,
            "message": "No residual model is loaded. The diagram has no prediction to show.",
        }
    if not isinstance(model, AnnResidualModel):
        return {
            **base,
            "status": "unavailable",
            "available": False,
            "message": "The loaded residual model is not the saved ANN, so its layers are not drawn.",
        }

    handoff = build_handoff(prediction)
    try:
        raw = model.raw_output_vector(handoff)
        vector = features_from_handoff(handoff)
    except (ValueError, KeyError) as exc:
        return {
            **base,
            "status": "error",
            "available": True,
            "message": f"ANN inference failed ({exc}).",
        }

    applied_bundle = bool(bundle and bundle.get("applied"))
    metrics = None if bundle is None else bundle.get("corrected_metrics")
    added = {"front_us": 0.0, "tail_us": 0.0, "crest_kv": 0.0}
    if applied_bundle and isinstance(metrics, dict):
        added = {
            "front_us": float(metrics["front_time_us"]) - prediction.front_time_us,
            "tail_us": float(metrics["tail_time_us"]) - prediction.tail_time_us,
            "crest_kv": float(metrics["crest_kv"]) - prediction.crest_voltage_kv,
        }
    heads = model.public_head_status(prediction.configuration.impulse_type)
    residuals = {}
    for key, (label, unit, _physics_key) in _RESIDUAL_LABELS.items():
        head = heads[key]
        head_active = bool(head["applied"])
        if key == "crest_kv":
            was_added = head_active and applied_bundle and added[key] != 0.0
        else:
            was_added = head_active and applied_bundle
        residuals[key] = {
            "label": label,
            "unit": unit,
            "raw": raw[key],
            "added": added[key],
            "applied": was_added,
            "head_active": head_active,
            "reason": head["reason"],
        }
    corrected = {
        "front_time_us": physics["front_time_us"] + added["front_us"],
        "tail_time_us": physics["tail_time_us"] + added["tail_us"],
        "crest_kv": physics["crest_kv"] + added["crest_kv"],
    }
    features = [
        {
            "name": name,
            "group": group,
            "label": label,
            "value": float(vector[index]),
        }
        for index, (name, group, label) in enumerate(FEATURE_PRESENTATION)
    ]
    if not handoff.in_training_domain:
        message = (
            "The point is outside the synthetic training range. "
            "Residuals were computed for display and were not added."
        )
    elif applied_bundle:
        message = model.applied_message(prediction.configuration.impulse_type)
    else:
        message = (
            bundle.get("reason")
            if bundle and bundle.get("reason")
            else "Residuals were not added. The physics prediction was kept."
        )
    return {
        **base,
        "status": "finished",
        "available": True,
        "in_domain": handoff.in_training_domain,
        "features": features,
        "residuals": residuals,
        "corrected": corrected,
        "message": message,
    }


def serialize_candidate(candidate: Candidate) -> dict[str, Any]:
    report = candidate.report
    return {
        "stages": candidate.stages,
        "charging_voltage_per_stage_kv": v_to_kv(
            candidate.configuration.charging_voltage_per_stage_v
        ),
        "front_resistance_per_stage_ohm": (
            candidate.configuration.front_resistance_per_stage_ohm
        ),
        "tail_resistance_per_stage_ohm": (
            candidate.configuration.tail_resistance_per_stage_ohm
        ),
        "front_build": str(candidate.front_build),
        "tail_build": str(candidate.tail_build),
        "front_time_us": candidate.prediction.front_time_us,
        "tail_time_us": candidate.prediction.tail_time_us,
        "crest_voltage_kv": candidate.prediction.crest_voltage_kv,
        "crest_kind": "loaded_estimate_N_Vstage_eta",
        "workbook_echo_crest_kv": candidate.prediction.workbook_echo_voltage_kv,
        "confirmed_safe": candidate.confirmed_safe,
        "buildable": candidate.buildable,
        "compliant_workbook": candidate.compliant,
        "safety_label": report.safety_label,
        "presentation": candidate.presentation,
        "worst_margin": candidate.worst_margin,
        "climb_operations": candidate.climb_operations,
        "climb_detail": list(candidate.climb_detail),
        "unevaluated_checks": [item.code for item in report.unevaluated()],
        "waveform": serialize_prediction(candidate.prediction)["waveform"],
    }


def _shot_spec_from_api(item: ShotSpecIn) -> ShotSpec:
    return ShotSpec(
        waveform_kind=WaveformKind(item.waveform_kind),
        level_fraction=item.level_fraction,
        chopping_time_s=None if item.chopping_time_us is None else us_to_s(item.chopping_time_us),
        collapse_duration_s=(
            None if item.collapse_duration_us is None else us_to_s(item.collapse_duration_us)
        ),
        collapse_shape=item.collapse_shape,
        label=item.label or "",
    )


def sequence_from_api(body: TestSequenceIn) -> TestSequence:
    if not body.shots:
        raise PipelineError("EMPTY_SEQUENCE", "A sequence must contain at least one shot.")
    known = bool(body.source and body.edition and body.clause)
    citation = StandardCitation(
        source=body.source,
        edition=body.edition,
        clause=body.clause,
        status=CitationStatus.KNOWN if known else CitationStatus.UNRESOLVED,
    )
    try:
        return TestSequence(
            reference_voltage_v=kv_to_v(body.reference_peak_kv),
            shots=tuple(_shot_spec_from_api(item) for item in body.shots),
            name=body.name or "user_configured",
            citation=citation,
        )
    except ValueError as exc:
        raise PipelineError("INVALID_SEQUENCE", str(exc)) from exc


def serialize_shot(evaluation) -> dict[str, Any]:
    chop_kwargs: dict[str, Any] = {}
    if evaluation.spec.waveform_kind.value == "chopped_lightning":
        chop_kwargs = {
            "chopping_time_s": evaluation.spec.chopping_time_s,
            "collapse_duration_s": evaluation.spec.collapse_duration_s,
            "collapse_shape": evaluation.spec.collapse_shape,
        }
    pred = serialize_prediction(evaluation.prediction, **chop_kwargs)
    chop = None
    if evaluation.chopped is not None:
        chop = {
            "chopping_time_us": evaluation.spec.chopping_time_s * 1e6,
            "collapse_duration_us": evaluation.spec.collapse_duration_s * 1e6,
            "collapse_shape": evaluation.spec.collapse_shape,
            "voltage_at_chop_kv": v_to_kv(evaluation.chopped.voltage_at_chop_v),
            "preliminary": True,
        }
    return {
        "label": evaluation.spec.label,
        "waveform_kind": evaluation.spec.waveform_kind.value,
        "level_fraction": evaluation.spec.level_fraction,
        "requested_peak_kv": evaluation.requested_voltage_kv,
        "predicted_crest_kv": evaluation.predicted_crest_kv,
        "crest_kind": "loaded_estimate_N_Vstage_eta",
        "tail_half_crossing_kv": evaluation.tail_half_crossing_kv,
        "constraint_status": evaluation.constraint_status.value,
        "confirmed_safe": evaluation.report.confirmed_safe,
        "safety_label": evaluation.report.safety_label,
        "predictions": pred,
        "chop": chop,
        "warnings": _caveats_to_warnings(evaluation.caveats),
    }


def predict_from_body(
    body: PhysicsPredictIn, *, profile: GeneratorProfile | None = None
) -> dict[str, Any]:
    _finite("requested_peak_kv", body.requested_peak_kv)
    _finite("charging_voltage_per_stage_kv", body.charging_voltage_per_stage_kv)
    _finite("front_resistance_per_stage_ohm", body.front_resistance_per_stage_ohm)
    _finite("tail_resistance_per_stage_ohm", body.tail_resistance_per_stage_ohm)
    _finite("load_capacitance_pf", body.load_capacitance_pf)
    _finite("divider_capacitance_pf", body.divider_capacitance_pf)
    _finite("stray_capacitance_pf", body.stray_capacitance_pf)
    _finite("stray_inductance_uh", body.stray_inductance_uh)
    _finite("expected_efficiency", body.expected_efficiency)

    stored = profile or store.load_profile().profile
    ratings = ratings_from_predict(body.ratings) if body.ratings is not None else ratings_from_profile(stored)

    try:
        circuit = TestCircuit.from_lab_units(
            load_capacitance_pf=body.load_capacitance_pf,
            divider_capacitance_pf=body.divider_capacitance_pf,
            stray_capacitance_pf=body.stray_capacitance_pf,
            stray_inductance_uh=body.stray_inductance_uh,
            efficiency=body.expected_efficiency,
        )
        configuration = GeneratorConfiguration(
            stages=body.stages,
            charging_voltage_per_stage_v=kv_to_v(body.charging_voltage_per_stage_kv),
            front_resistance_per_stage_ohm=body.front_resistance_per_stage_ohm,
            tail_resistance_per_stage_ohm=body.tail_resistance_per_stage_ohm,
            impulse_type=_API_TO_IMPULSE[body.impulse_type],
            circuit=circuit,
            ratings=ratings,
        )
        prediction = predict(configuration, kv_to_v(body.requested_peak_kv))
    except ValueError as exc:
        raise PipelineError("INVALID_CONFIGURATION", str(exc)) from exc

    warnings = _caveats_to_warnings(prediction.caveats)
    chop_kwargs: dict[str, Any] = {}
    if body.waveform_kind == "chopped_lightning":
        if body.chopping_time_us is None or body.collapse_duration_us is None or body.collapse_shape is None:
            raise PipelineError(
                "CHOP_PARAMETERS_REQUIRED",
                "Chopped lightning requires chopping_time_us, collapse_duration_us "
                "and collapse_shape. They are not defaulted.",
            )
        chop_kwargs = {
            "chopping_time_s": us_to_s(body.chopping_time_us),
            "collapse_duration_s": us_to_s(body.collapse_duration_us),
            "collapse_shape": body.collapse_shape,
        }
    bundle = apply_residual_waveforms(prediction, get_residual_model(), **chop_kwargs)
    predictions = bundle["physics"]
    predictions["corrected_waveform"] = bundle["corrected_waveform"]
    predictions["corrected_metrics"] = bundle["corrected_metrics"]
    predictions["requested_waveform"] = bundle["requested_waveform"]
    predictions["waveform_overlay"] = bundle["waveform_overlay"]
    if bundle["residual_corrected"] is not None:
        predictions["residual_corrected"] = bundle["residual_corrected"]
    warnings = warnings + bundle["warnings"]
    payload = {
        "success": True,
        "mode": bundle["mode"],
        "predictions": predictions,
        "units": PREDICT_UNITS,
        "warnings": warnings,
        "metadata": {
            "physics_model_used": True,
            "ml_model_used": bundle["ml_model_used"],
            "residual_reason": bundle["reason"],
            "ann_heads": bundle.get("ann_heads"),
            "ratings_source": "request" if body.ratings is not None else "stored_profile",
        },
    }

    if body.sequence is not None:
        sequence = sequence_from_api(body.sequence)
        evaluated = evaluate_sequence(sequence, configuration)
        payload["sequence"] = {
            "name": sequence.name,
            "citation": {
                "status": sequence.citation.status.value,
                "source": sequence.citation.source,
                "edition": sequence.citation.edition,
                "clause": sequence.citation.clause,
                "note": sequence.citation.note,
            },
            "shots": [serialize_shot(shot) for shot in evaluated.shots],
            "all_confirmed_safe": evaluated.all_confirmed_safe,
        }
        payload["warnings"] = payload["warnings"] + [
            {"code": "SEQUENCE_IS_USER_CONFIGURED",
             "message": sequence.citation.note,
             "severity": "notice"}
        ]

    if body.waveform_kind == "chopped_lightning":
        spec = ShotSpec(
            waveform_kind=WaveformKind.CHOPPED_LIGHTNING,
            level_fraction=1.0,
            chopping_time_s=us_to_s(body.chopping_time_us),
            collapse_duration_s=us_to_s(body.collapse_duration_us),
            collapse_shape=body.collapse_shape,
        )
        shot = evaluate_shot(spec, configuration, kv_to_v(body.requested_peak_kv))
        payload["chop"] = serialize_shot(shot)["chop"]
        payload["warnings"] = payload["warnings"] + serialize_shot(shot)["warnings"]

    return payload


def _apply_optional_residuals(
    candidate: Candidate, model: ResidualPredictor
) -> tuple[dict[str, float], list[dict[str, str]]]:
    handoff = build_handoff(candidate.prediction)
    residuals = model.predict_residuals(handoff)
    if not isinstance(residuals, ResidualCorrection):
        raise PipelineError(
            "INCOMPATIBLE_ML_ARTIFACT",
            "Loaded residual model did not return a ResidualCorrection.",
            http_status=500,
        )
    corrected = apply_residuals(candidate.prediction, residuals)
    extra = _caveats_to_warnings(handoff.caveats)
    extra.append(
        {
            "code": "ML_RESIDUALS_APPLIED",
            "message": (
                "A residual model was applied to the reported times and crest. "
                "It did not re-search the catalogue and is not laboratory validation."
            ),
            "severity": "warning",
        }
    )
    return corrected, extra


def _summarise_optimise(
    result: OptimisationResult,
    *,
    mode: str,
    chosen: Candidate | None,
    meets_window: bool,
) -> str:
    if chosen is None:
        return (
            f"{mode.replace('_', '-')}: no catalogue combination was buildable. "
            "No settings were fabricated."
        )
    if not meets_window:
        return (
            f"{mode.replace('_', '-')}: no stocked combination meets the acceptance "
            f"window. The curve is the physics prediction for the closest stocked "
            f"setup, not a recommendation: {chosen.stages} stages, "
            f"front {chosen.configuration.front_resistance_per_stage_ohm:g} ohm/stage, "
            f"tail {chosen.configuration.tail_resistance_per_stage_ohm:g} ohm/stage, "
            f"workbook front/peak {chosen.prediction.front_time_us:.4g} us, "
            f"tail {chosen.prediction.tail_time_us:.4g} us, "
            f"loaded crest {chosen.prediction.crest_voltage_kv:.4g} kV. "
            "Missing resistors were not invented. Decision support only; "
            "does not operate equipment."
        )
    label = (
        "recommended (confirmed safe)"
        if chosen.confirmed_safe
        else "buildable-unconfirmed (not safe)"
    )
    return (
        f"{mode.replace('_', '-')}: {label}. {chosen.stages} stages, "
        f"front {chosen.configuration.front_resistance_per_stage_ohm:g} ohm/stage, "
        f"tail {chosen.configuration.tail_resistance_per_stage_ohm:g} ohm/stage, "
        f"front {chosen.prediction.front_time_us:.4g} us, "
        f"tail {chosen.prediction.tail_time_us:.4g} us, "
        f"loaded crest {chosen.prediction.crest_voltage_kv:.4g} kV. "
        "Decision support only; does not operate equipment."
    )


def optimise_from_request(
    request: TestRequestIn,
    *,
    residual_model: ResidualPredictor | None = None,
) -> dict[str, Any]:
    report = validate_request(request)
    if not report.ok:
        raise PipelineError(
            "VALIDATION_FAILED",
            "The request has blocking validation errors, so optimisation was not run.",
            details=[item.model_dump() for item in report.errors],
        )

    model = residual_model if residual_model is not None else get_residual_model()
    try:
        physics_request = OptimisationRequest(
            requested_voltage_v=kv_to_v(request.required_peak_kv),
            impulse_type=_API_TO_IMPULSE[request.impulse_type],
            circuit=circuit_from_request(request),
            ratings=ratings_from_profile(
                GeneratorProfile(
                    max_stages=request.generator.max_stages,
                    min_active_stages=request.generator.min_active_stages,
                    max_stage_voltage_kv=request.generator.max_stage_voltage_kv,
                    stage_capacitance_uf=request.generator.stage_capacitance_uf,
                )
            ),
            inventory=inventory_from_api(request.inventory),
        )
        result = optimise(physics_request)
    except ValueError as exc:
        raise PipelineError("OPTIMISE_INPUT_INVALID", str(exc)) from exc
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "PHYSICS_ERROR",
            "The physics optimiser raised an unexpected error. No settings were fabricated.",
            http_status=500,
        ) from exc

    # Compliant choices are the only ones that may be called a result inside
    # the window. A buildable but non-compliant candidate is still predicted,
    # so an unbuildable switching optimum does not hide the waveform.
    compliant_choice = result.recommended or result.best_unconfirmed
    source = compliant_choice or result.best
    meets_window = compliant_choice is not None
    warnings = _caveats_to_warnings(result.caveats)
    bundle = None
    if source is not None:
        bundle = apply_residual_waveforms(source.prediction, model)
        warnings.extend(bundle["warnings"])
    mode = bundle["mode"] if bundle is not None else "physics_only"
    ml_used = bool(bundle and bundle["ml_model_used"])

    recommended = (
        serialize_candidate(result.recommended) if result.recommended is not None else None
    )
    best_unconfirmed = (
        serialize_candidate(result.best_unconfirmed)
        if result.best_unconfirmed is not None
        else None
    )
    if recommended is not None:
        presented = recommended
        presentation = "recommended"
    elif best_unconfirmed is not None:
        presented = best_unconfirmed
        presentation = "buildable_unconfirmed"
    elif result.best is not None:
        presented = serialize_candidate(result.best)
        presentation = "closest_stocked_noncompliant"
    else:
        presented = None
        presentation = "none"

    predictions = None
    if source is not None and bundle is not None:
        predictions = bundle["physics"]
        predictions["corrected_waveform"] = bundle["corrected_waveform"]
        predictions["corrected_metrics"] = bundle["corrected_metrics"]
        predictions["requested_waveform"] = bundle["requested_waveform"]
        predictions["waveform_overlay"] = bundle["waveform_overlay"]
        predictions["constraint_results"] = bundle["constraint_results"]
        if bundle["residual_corrected"] is not None:
            predictions["residual_corrected"] = bundle["residual_corrected"]

    payload = {
        "success": True,
        "mode": mode,
        "detail": _summarise_optimise(
            result, mode=mode, chosen=source, meets_window=meets_window
        ),
        "predictions": predictions,
        "units": PREDICT_UNITS,
        "warnings": warnings,
        "metadata": {
            "physics_model_used": True,
            "ml_model_used": ml_used,
            "residual_reason": None if bundle is None else bundle["reason"],
            "ann_heads": None if bundle is None else bundle.get("ann_heads"),
            "evaluations": result.evaluations,
            "presentation": presentation,
            "hardware_control": False,
        },
        "physics_waveform": None if predictions is None else predictions.get("waveform"),
        "corrected_waveform": None if bundle is None else bundle["corrected_waveform"],
        "ranked_alternatives": [serialize_candidate(item) for item in result.candidates[:5]],
        "inventory_note": (
            "Front and tail resistances are taken from the stocked inventory. "
            "The ANN does not invent resistor values or override equipment limits."
        ),
        "recommended": recommended,
        "best_unconfirmed": best_unconfirmed,
        "candidate": presented,
        "ann_inference": (
            build_ann_inference(source.prediction, model, bundle)
            if source is not None
            else {
                "status": "error",
                "available": isinstance(model, AnnResidualModel),
                "synthetic_only": True,
                "laboratory_validated": False,
                "architecture_matches_reference": False,
                "reference": REFERENCE_ARCHITECTURE,
                "discrepancy": ARCHITECTURE_DISCREPANCY,
                "residuals": None,
                "message": "No configuration was produced, so the network was not run.",
            }
        ),
        "validation": report.model_dump(),
    }

    if request.sequence is not None:
        sequence = sequence_from_api(request.sequence)
        sequenced = optimise_sequence(physics_request, sequence)
        payload["sequence"] = {
            "name": sequence.name,
            "citation": {
                "status": sequence.citation.status.value,
                "source": sequence.citation.source,
                "edition": sequence.citation.edition,
                "clause": sequence.citation.clause,
                "note": sequence.citation.note,
            },
            "infeasible": sequenced.infeasible,
            "all_confirmed_safe": sequenced.all_confirmed_safe,
            "shots": [
                {
                    "waveform_kind": item.spec.waveform_kind.value,
                    "level_fraction": item.spec.level_fraction,
                    "requested_peak_kv": v_to_kv(item.requested_voltage_v),
                    "constraint_status": (
                        item.evaluation.constraint_status.value
                        if item.evaluation is not None
                        else "failed"
                    ),
                    "confirmed_safe": (
                        item.result.recommended is not None
                    ),
                    "safety_label": (
                        item.evaluation.report.safety_label
                        if item.evaluation is not None
                        else "unbuildable"
                    ),
                    "predicted_crest_kv": (
                        item.evaluation.predicted_crest_kv
                        if item.evaluation is not None
                        else None
                    ),
                    "tail_half_crossing_kv": (
                        item.evaluation.tail_half_crossing_kv
                        if item.evaluation is not None
                        else None
                    ),
                    "recommended": (
                        serialize_candidate(item.result.recommended)
                        if item.result.recommended is not None
                        else None
                    ),
                    "best_unconfirmed": (
                        serialize_candidate(item.result.best_unconfirmed)
                        if item.result.best_unconfirmed is not None
                        else None
                    ),
                    "rejection_reasons": [reason for _, reason in item.result.rejected[:8]],
                }
                for item in sequenced.shots
            ],
        }
        if sequenced.infeasible:
            payload["detail"] = (
                payload["detail"]
                + " Sequence contains a shot with no feasible catalogue candidate."
            )
        elif not sequenced.all_confirmed_safe:
            payload["detail"] = (
                payload["detail"]
                + " Sequence shots are buildable-unconfirmed, not recommended/safe."
            )

    return payload


def profile_provenance(stored) -> dict[str, Any]:
    if stored.is_factory_default:
        return {
            "verified": False,
            "status": "assumed",
            "source": "supplied_material_default",
            "note": (
                "No generator setup has been saved. These values come from the "
                "supplied problem statement and workbook. They are assumed, not "
                "verified against this laboratory's equipment."
            ),
        }
    return {
        "verified": False,
        "status": "unknown",
        "source": "admin_saved",
        "updated_at": stored.updated_at,
        "updated_by": stored.updated_by,
        "note": (
            "Recorded in the generator-setup interface. Saving a profile is not "
            "laboratory verification of ratings or inventory."
        ),
    }
