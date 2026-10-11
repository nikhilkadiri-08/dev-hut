"""Optional residual-model hook.

The product model is the ANN residual corrector in ``physics_model.ann_model``.
The workbook kNN is a Hidden-Test comparison only and is never loaded here.

If the artifact is missing, fails to load, or is out of domain at inference,
callers must stay ``physics_only``. Tests may inject a stub via
:func:`set_residual_model`.
"""

from __future__ import annotations

from typing import Protocol

from physics_model.ml_interface import MLHandoff, ResidualCorrection

__all__ = [
    "ResidualPredictor",
    "get_residual_model",
    "set_residual_model",
    "load_status",
    "load_default_residual_model",
]


class ResidualPredictor(Protocol):
    def predict_residuals(self, handoff: MLHandoff) -> ResidualCorrection:
        """Return residuals in the dataset units (us, us, kV)."""


_loaded: ResidualPredictor | None = None
_STATUS_CLEARED: dict[str, object] = {
    "loaded": False,
    "kind": None,
    "detail": "No residual model has been loaded.",
    "synthetic_only": None,
    "version": None,
    "active_heads": None,
}
_status: dict[str, object] = dict(_STATUS_CLEARED)


def get_residual_model() -> ResidualPredictor | None:
    return _loaded


def load_status() -> dict[str, object]:
    return dict(_status)


def set_residual_model(model: ResidualPredictor | None) -> None:
    """Install or clear the process-wide residual model. Used by tests."""
    global _loaded
    _loaded = model
    if model is None:
        _status.clear()
        _status.update(_STATUS_CLEARED)
        _status["detail"] = "Residual model cleared."
        return
    kind = type(model).__name__
    synthetic = bool(getattr(model, "synthetic_only", False))
    detail = f"{kind} is installed."
    status_detail = getattr(model, "status_detail", None)
    if callable(status_detail):
        detail = status_detail()
    active = getattr(model, "active_summary", None)
    _status.update(
        {
            "loaded": True,
            "kind": kind,
            "detail": detail,
            "synthetic_only": synthetic,
            "version": getattr(model, "version", None),
            "active_heads": active() if callable(active) else None,
        }
    )


def load_default_residual_model() -> None:
    """Load the ANN artifact if present. Failures stay physics-only."""
    from physics_model.ann_model import load_artifact

    try:
        model = load_artifact()
    except FileNotFoundError:
        _status.clear()
        _status.update(_STATUS_CLEARED)
        _status["detail"] = "ANN artifact is not present; running physics-only."
        return
    except Exception as exc:
        _status.clear()
        _status.update(_STATUS_CLEARED)
        _status["detail"] = f"ANN artifact failed to load ({exc}). Physics-only."
        return
    set_residual_model(model)
    _status["kind"] = "ann_residual_mlp"
    _status["synthetic_only"] = True
    _status["version"] = model.version
    _status["active_heads"] = model.active_summary()
    _status["detail"] = model.status_detail()
