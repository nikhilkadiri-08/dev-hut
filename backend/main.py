"""FastAPI application.

Input validation, the stored generator profile, and a physics-only prediction /
optimisation pipeline that calls ``physics_model`` rather than re-deriving formulas.
No hardware control. A residual ML model is optional and is not loaded by default.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import Request

from backend import config, store
from backend.ml_runtime import get_residual_model, load_default_residual_model, load_status
from backend.pipeline import (
    PipelineError,
    optimise_from_request,
    physics_import_status,
    predict_from_body,
    profile_provenance,
)
from backend.schemas import (
    IssueOut,
    OperatorRequestIn,
    PhysicsPredictIn,
    ProfileResponse,
    ProfileUpdateIn,
    TestRequestIn,
    ValidationResponse,
    merge_operator_request,
)
from backend.validation import validate_profile, validate_request

app = FastAPI(
    title="Impulse Generator Setting Optimiser",
    version="0.1.0",
    description=(
        "Decision-support tool for high-voltage impulse testing. Recommends candidate "
        "impulse generator configurations for engineer review. It does not operate any "
        "equipment and does not replace approved test procedures."
    ),
)

@app.on_event("startup")
def _load_ann_residual_model() -> None:
    load_default_residual_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schema errors become the same shape as engineering validation errors
# ---------------------------------------------------------------------------

_FRIENDLY_SCHEMA_MESSAGES = {
    "missing": "This value is required.",
    "float_parsing": "Enter a number.",
    "int_parsing": "Enter a whole number.",
    "literal_error": "Choose one of the listed options.",
    "extra_forbidden": "This field is not recognised.",
}


@app.exception_handler(RequestValidationError)
async def _schema_errors_as_validation_response(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    issues = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"] if part != "body")
        friendly = _FRIENDLY_SCHEMA_MESSAGES.get(error["type"])
        message = friendly or error.get("msg", "Invalid value.")
        if not friendly:
            message = message[0].upper() + message[1:] if message else message
        issues.append(
            IssueOut(
                severity="error",
                code=f"SCHEMA_{error['type'].upper()}",
                message=message,
                field=location or None,
            )
        )
    body = ValidationResponse(ok=False, errors=issues)
    return JSONResponse(status_code=422, content=body.model_dump())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    """Signpost, because the dashboard is served by Vite on a different port."""
    return {
        "service": app.title,
        "dashboard": "http://localhost:5173",
        "api_docs": "/docs",
        "note": "This port serves the API only. The user interface is on port 5173.",
    }


@app.get("/api/health")
def health() -> dict:
    """Liveness plus whether the physics package imported. ML absence is not a failure."""
    physics = physics_import_status()
    return {
        "status": "ok",
        "version": app.version,
        "physics_model": physics,
        "ml_model": {
            **load_status(),
            "used_by_default": False,
        },
    }


@app.get("/api/metadata")
def metadata() -> dict:
    """Everything the form needs: catalogues, limits, tolerances and caveats.

    The frontend renders from this payload rather than hardcoding engineering values,
    so a correction to ``backend/config.py`` reaches the UI without a frontend change.
    """
    limits = config.GENERATOR
    return {
        "generator": {
            "max_stages": limits.max_stages,
            "min_active_stages": limits.min_active_stages,
            "max_stage_voltage_kv": limits.max_stage_voltage_kv,
            "stage_capacitance_uf": limits.stage_capacitance_uf,
            "capacitors_per_stage": limits.capacitors_per_stage,
            "min_test_voltage_kv": limits.min_test_voltage_kv,
            "max_test_voltage_kv": limits.max_test_voltage_kv,
            "max_charge_voltage_kv": limits.max_charge_voltage_kv,
        },
        "standard": {
            "id": config.STANDARD_ID,
            "label": config.STANDARD_LABEL,
            "edition_note": config.STANDARD_EDITION_NOTE,
        },
        "impulse_types": [spec.to_dict() for spec in config.IMPULSE_SPECS.values()],
        "test_objects": [item.to_dict() for item in config.TEST_OBJECT_CATALOGUE],
        "inventory": {
            "front": [item.to_dict() for item in config.FRONT_RESISTORS],
            "tail": [item.to_dict() for item in config.TAIL_RESISTORS],
            "max_front_ohms_per_stage": config.max_series_resistance_per_stage(
                config.FRONT_RESISTORS
            ),
            "max_tail_ohms_per_stage": config.max_series_resistance_per_stage(
                config.TAIL_RESISTORS
            ),
        },
        "defaults": {
            "required_peak_kv": config.DEFAULT_TEST_VOLTAGE_KV,
            "divider_capacitance_pf": config.DEFAULT_DIVIDER_CAPACITANCE_PF,
            "stray_capacitance_pf": config.DEFAULT_STRAY_CAPACITANCE_PF,
            "stray_inductance_uh": config.DEFAULT_STRAY_INDUCTANCE_UH,
            "expected_efficiency": config.DEFAULT_EFFICIENCY,
        },
        "ml_domain": config.ml_domain(),
        "literature": _literature_payload(),
        "assumptions": config.ASSUMPTIONS.to_list(),
        "assumptions_requiring_attention": [
            entry.to_dict() for entry in config.ASSUMPTIONS.requiring_attention()
        ],
        "capabilities": {
            "validation": "implemented",
            "admin_generator_setup": "implemented",
            "operator_test_request": "implemented",
            "physics_engine": "implemented",
            "ml_residual_model": "not_implemented",
            "candidate_optimiser": "implemented",
            "compliance_checker": "not_implemented",
            "robustness_analysis": "not_implemented",
            "measured_waveform_import": "not_implemented",
            "literature_catalogue": "implemented",
        },
    }


def _literature_payload() -> dict:
    from physics_model.literature import load_catalogue, workbook_conflicts

    catalogue = load_catalogue()
    payload = catalogue.to_dict()
    payload["workbook_conflicts"] = workbook_conflicts()
    return payload


@app.get("/api/literature")
def literature_catalogue() -> dict:
    """Cited open-reference values for objects and editions the workbook does not cover."""
    return _literature_payload()


@app.get("/api/literature/lookup")
def literature_lookup(q: str = Query(..., min_length=1, max_length=200)) -> dict:
    """Match a new test object, voltage class or impulse type against the catalogue."""
    from physics_model.literature import lookup

    return lookup(q).to_dict()


def _parse_optimise_body(raw: object) -> TestRequestIn:
    if not isinstance(raw, dict):
        raise RequestValidationError(
            [
                {
                    "type": "dict_type",
                    "loc": ("body",),
                    "msg": "Request body must be a JSON object.",
                    "input": raw,
                }
            ]
        )
    try:
        if "circuit" in raw:
            return TestRequestIn.model_validate(raw)
        operator = OperatorRequestIn.model_validate(raw)
        return merge_operator_request(operator, store.load_profile().profile)
    except ValidationError as exc:
        _reraise_schema_error(exc)
        raise AssertionError("schema error should have been reraised") from exc


@app.get("/api/ml/status")
def ml_status(impulse: str = Query(default="lightning")) -> dict:
    """Architecture of the saved ANN. Weights are the learned ones, normalized for drawing.

    This does not run inference and does not retrain.
    """
    from physics_model.ann_model import (
        ARCHITECTURE_DISCREPANCY,
        REFERENCE_ARCHITECTURE,
        AnnResidualModel,
    )

    names = {"lightning": "Lightning", "switching": "Switching"}
    if impulse not in names:
        raise HTTPException(
            status_code=422,
            detail="impulse must be lightning or switching.",
        )
    model = get_residual_model()
    status = load_status()
    payload = {
        "status": "unavailable",
        "available": False,
        "synthetic_only": True,
        "laboratory_validated": False,
        "architecture_matches_reference": False,
        "reference": REFERENCE_ARCHITECTURE,
        "discrepancy": ARCHITECTURE_DISCREPANCY,
        "detail": status.get("detail"),
        "diagram": None,
        "active_heads": status.get("active_heads"),
    }
    if not isinstance(model, AnnResidualModel):
        return payload
    payload["status"] = "ready"
    payload["available"] = True
    payload["diagram"] = model.diagram(names[impulse])
    payload["active_heads"] = model.active_summary()
    return payload


@app.post("/api/ml/residuals")
async def ml_residuals(request: Request) -> JSONResponse:
    """Run the saved ANN on the configuration the optimiser would select.

    Accepts the same body as ``POST /api/optimise``. Does not retrain.
    """
    try:
        raw = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content=ValidationResponse(
                ok=False,
                errors=[
                    IssueOut(
                        severity="error",
                        code="SCHEMA_JSON",
                        message="Request body must be JSON.",
                    )
                ],
            ).model_dump(),
        )
    try:
        body = _parse_optimise_body(raw)
    except RequestValidationError:
        raise
    try:
        payload = optimise_from_request(body)
    except PipelineError as exc:
        return _pipeline_error_response(exc)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "available": get_residual_model() is not None,
                "residuals": None,
                "message": "The residual request failed. No prediction was invented.",
                "synthetic_only": True,
                "laboratory_validated": False,
            },
        )
    inference = payload.get("ann_inference")
    if not isinstance(inference, dict):
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "residuals": None,
                "message": "The residual request did not return an inference record.",
                "synthetic_only": True,
                "laboratory_validated": False,
            },
        )
    return JSONResponse(status_code=200, content=inference)


@app.get("/api/ann/training")
def ann_training_history() -> dict:
    """Synthetic ANN training telemetry: architecture and per-epoch loss.

    This is not laboratory validation. Hidden Test is not a curve in this payload.
    """
    from physics_model.ann_model import load_training_history

    try:
        return load_training_history()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="ANN training history has not been written.",
        ) from exc


def _profile_response(stored) -> ProfileResponse:
    check = validate_profile(stored.profile)
    return ProfileResponse(
        stored=stored,
        ok=check.ok,
        errors=check.errors,
        warnings=check.warnings,
        notices=check.notices,
        reach=check.reach,
    )


@app.get("/api/admin/profile", response_model=ProfileResponse)
def read_profile() -> ProfileResponse:
    """The laboratory's saved generator setup, with its validation status."""
    return _profile_response(store.load_profile())


@app.get("/api/generator/profile")
def read_generator_profile() -> dict:
    """Alias of the stored admin profile, plus assumed/unknown provenance.

    Does not invent ratings or inventory. ``verified`` is always false: this
    application has no laboratory verification workflow.
    """
    stored = store.load_profile()
    body = _profile_response(stored).model_dump()
    body["provenance"] = profile_provenance(stored)
    return body


@app.put("/api/admin/profile", response_model=ProfileResponse)
def write_profile(update: ProfileUpdateIn) -> ProfileResponse:
    """Save a generator setup.

    Refuses to persist a profile with blocking errors, so an operator can never
    inherit a setup that exceeds the equipment's physical ratings.
    """
    check = validate_profile(update.profile)
    if not check.ok:
        return ProfileResponse(
            stored=store.load_profile(),
            ok=False,
            errors=check.errors,
            warnings=check.warnings,
            notices=check.notices,
            reach=check.reach,
        )
    return _profile_response(store.save_profile(update.profile, update.updated_by))


@app.post("/api/admin/profile/reset", response_model=ProfileResponse)
def reset_profile() -> ProfileResponse:
    """Discard the saved setup and return to the supplied-material values."""
    return _profile_response(store.reset_profile())


@app.post("/api/test-request/validate", response_model=ValidationResponse)
def validate_test_request(request: OperatorRequestIn) -> ValidationResponse:
    """Check a test request from the operator interface.

    Generator ratings and resistor inventory are taken from the stored admin profile
    rather than from the request, so the operator cannot accidentally describe hardware
    the laboratory does not have.
    """
    stored = store.load_profile()
    resolved = merge_operator_request(request, stored.profile)
    report = validate_request(resolved)
    if stored.is_factory_default:
        report.notices.append(
            IssueOut(
                severity="notice",
                code="PROFILE_NOT_CONFIGURED",
                message=(
                    "No generator setup has been saved yet, so the values taken from "
                    "the supplied problem statement and workbook are in use. Confirm "
                    "them in the generator setup interface against the actual equipment."
                ),
            )
        )
    return report


@app.post("/api/validate", response_model=ValidationResponse)
def validate(request: TestRequestIn) -> ValidationResponse:
    """Check and normalise a fully specified test request.

    Takes generator ratings and inventory inline instead of from the stored profile.
    Useful for tests and scripted runs; the operator interface uses
    ``/api/test-request/validate``.
    """
    return validate_request(request)


def _pipeline_error_response(exc: PipelineError) -> JSONResponse:
    payload = {
        "success": False,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
        "detail": exc.message,
        "metadata": {
            "physics_model_used": True,
            "ml_model_used": get_residual_model() is not None,
        },
    }
    return JSONResponse(status_code=exc.http_status, content=payload)


def _reraise_schema_error(exc: ValidationError) -> None:
    raise RequestValidationError(exc.errors()) from exc


@app.post("/api/physics/predict")
def physics_predict(body: PhysicsPredictIn) -> JSONResponse:
    """Run one configuration through ``physics_model.calculations.predict``.

    Crest is the preliminary loaded estimate ``N * V_stage * eta``, not the
    requested voltage. Workbook E20 is returned separately and labelled as an echo.
    """
    try:
        payload = predict_from_body(body)
    except PipelineError as exc:
        if exc.http_status >= 500:
            return _pipeline_error_response(exc)
        issues = [
            IssueOut(
                severity="error",
                code=exc.code,
                message=exc.message,
            )
        ]
        return JSONResponse(
            status_code=422,
            content=ValidationResponse(ok=False, errors=issues).model_dump(),
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "PHYSICS_ERROR",
                    "message": "The physics model raised an unexpected error.",
                    "details": [],
                },
                "detail": "The physics model raised an unexpected error.",
            },
        )
    return JSONResponse(status_code=200, content=payload)


@app.post("/api/optimise")
async def optimise_endpoint(request: Request) -> JSONResponse:
    """Physics-only catalogue search, unless a residual model has been injected.

    Accepts either a fully specified ``TestRequestIn`` (``circuit`` present) or an
    operator request that is merged with the stored generator profile. A candidate
    is ``recommended`` only when every applicable hard safety check is evaluable
    and passing. No hardware is controlled.
    """
    try:
        raw = await request.json()
    except Exception:
        return JSONResponse(
            status_code=422,
            content=ValidationResponse(
                ok=False,
                errors=[
                    IssueOut(
                        severity="error",
                        code="SCHEMA_JSON",
                        message="Request body must be JSON.",
                    )
                ],
            ).model_dump(),
        )

    if not isinstance(raw, dict):
        return JSONResponse(
            status_code=422,
            content=ValidationResponse(
                ok=False,
                errors=[
                    IssueOut(
                        severity="error",
                        code="SCHEMA_JSON",
                        message="Request body must be a JSON object.",
                    )
                ],
            ).model_dump(),
        )

    try:
        if "circuit" in raw:
            body = TestRequestIn.model_validate(raw)
        else:
            operator = OperatorRequestIn.model_validate(raw)
            body = merge_operator_request(operator, store.load_profile().profile)
    except ValidationError as exc:
        _reraise_schema_error(exc)

    try:
        payload = optimise_from_request(body)
    except PipelineError as exc:
        return _pipeline_error_response(exc)
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "PHYSICS_ERROR",
                    "message": (
                        "The physics optimiser raised an unexpected error. "
                        "No settings were fabricated."
                    ),
                    "details": [],
                },
                "detail": (
                    "The physics optimiser raised an unexpected error. "
                    "No settings were fabricated."
                ),
            },
        )
    return JSONResponse(status_code=200, content=payload)
