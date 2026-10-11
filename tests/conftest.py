"""Test isolation for the optional ANN residual model.

Startup may load an artifact. Tests that assert physics-only clear it unless
they opt in with ``@pytest.mark.ann``.
"""

from __future__ import annotations

import pytest

from backend.ml_runtime import set_residual_model


@pytest.fixture(autouse=True)
def _clear_residual_model_unless_marked(request):
    if request.node.get_closest_marker("ann"):
        yield
        return
    set_residual_model(None)
    yield
    set_residual_model(None)
