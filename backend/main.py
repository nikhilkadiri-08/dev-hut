"""FastAPI application.

Currently exposes the input layer only: metadata that drives the form, and validation
of a submitted test request. ``POST /api/optimise`` is deliberately wired up and
deliberately returns 501, so the frontend can integrate against the real endpoint
before the physics engine exists, and so nobody can mistake a placeholder for a result.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from backend import config, store
from backend.schemas import (
    IssueOut,
    OperatorRequestIn,
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
    return {"status": "ok", "version": app.version}


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
        "assumptions": config.ASSUMPTIONS.to_list(),
        "assumptions_requiring_attention": [
            entry.to_dict() for entry in config.ASSUMPTIONS.requiring_attention()
        ],
        "capabilities": {
            "validation": "implemented",
            "admin_generator_setup": "implemented",
            "operator_test_request": "implemented",
            "physics_engine": "not_implemented",
            "ml_residual_model": "not_implemented",
            "candidate_optimiser": "not_implemented",
            "compliance_checker": "not_implemented",
            "robustness_analysis": "not_implemented",
            "measured_waveform_import": "not_implemented",
        },
    }


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


@app.post("/api/optimise", status_code=501)
def optimise(request: TestRequestIn) -> JSONResponse:
    """Placeholder for the optimisation pipeline.

    Validates the request so the frontend sees real validation behaviour, then refuses
    rather than returning a fabricated recommendation.
    """
    report = validate_request(request)
    return JSONResponse(
        status_code=501,
        content={
            "detail": (
                "The optimisation pipeline is not implemented yet. The physics engine, "
                "residual model, candidate optimiser and compliance checker are still "
                "to be built, and this endpoint will not return a recommendation until "
                "they exist."
            ),
            "validation": report.model_dump(),
        },
    )
