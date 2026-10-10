"""Optional residual-model hook.

No model is trained, downloaded or invented here. The workbook kNN in
``physics_model.ml_interface`` documents the spreadsheet only.

A compatible artifact is a Python object exposing
``predict_residuals(handoff) -> ResidualCorrection``. Tests may inject one via
:func:`set_residual_model`. Production leaves this unset, so ``/api/optimise``
runs as ``physics_only``.

To plug a real model later, implement that method, then call
``set_residual_model(instance)`` at process start (or pass the instance into
the pipeline). Do not reuse the workbook kNN as this project's ML model.
"""

from __future__ import annotations

from typing import Protocol

from physics_model.ml_interface import MLHandoff, ResidualCorrection

__all__ = [
    "ResidualPredictor",
    "get_residual_model",
    "set_residual_model",
]


class ResidualPredictor(Protocol):
    def predict_residuals(self, handoff: MLHandoff) -> ResidualCorrection:
        """Return residuals in the dataset units (us, us, kV)."""


_loaded: ResidualPredictor | None = None


def get_residual_model() -> ResidualPredictor | None:
    return _loaded


def set_residual_model(model: ResidualPredictor | None) -> None:
    """Install or clear the process-wide residual model. Used by tests."""
    global _loaded
    _loaded = model
