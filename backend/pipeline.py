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
    TestRequestIn,
)
from backend.units import f_to_pf, f_to_uf, kv_to_v, v_to_kv
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
from physics_model.ml_interface import ResidualCorrection
from physics_model.optimiser import Candidate, OptimisationResult

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


def serialize_prediction(prediction: PhysicsPrediction) -> dict[str, Any]:
    c_eq = series_equivalent_capacitance(
        prediction.generator_capacitance_f, prediction.total_load_capacitance_f
    )
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
    return {
        "success": True,
        "mode": "physics_only",
        "predictions": serialize_prediction(prediction),
        "units": PREDICT_UNITS,
        "warnings": warnings,
        "metadata": {
            "physics_model_used": True,
            "ml_model_used": False,
            "ratings_source": "request" if body.ratings is not None else "stored_profile",
        },
    }


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
) -> str:
    if chosen is None:
        return (
            f"{mode.replace('_', '-')}: no catalogue combination was both buildable "
            "and workbook-compliant. No settings were fabricated."
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

    chosen = result.recommended or result.best_unconfirmed
    mode = "physics_plus_residual" if model is not None else "physics_only"
    ml_used = model is not None
    warnings = _caveats_to_warnings(result.caveats)
    residual_correction = None
    if chosen is not None and model is not None:
        residual_correction, extra = _apply_optional_residuals(chosen, model)
        warnings.extend(extra)

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
    else:
        presented = None
        presentation = "none"

    predictions = None
    if chosen is not None:
        predictions = serialize_prediction(chosen.prediction)
        if residual_correction is not None:
            predictions["residual_corrected"] = residual_correction

    return {
        "success": True,
        "mode": mode,
        "detail": _summarise_optimise(result, mode=mode, chosen=chosen),
        "predictions": predictions,
        "units": PREDICT_UNITS,
        "warnings": warnings,
        "metadata": {
            "physics_model_used": True,
            "ml_model_used": ml_used,
            "evaluations": result.evaluations,
            "presentation": presentation,
            "hardware_control": False,
        },
        "recommended": recommended,
        "best_unconfirmed": best_unconfirmed,
        "candidate": presented,
        "validation": report.model_dump(),
    }


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
