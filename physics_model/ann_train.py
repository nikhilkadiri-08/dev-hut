"""Train the ANN residual model on the synthetic Train split only.

Usage:
    .venv/bin/python -m physics_model.ann_train

Writes physics_model/artifacts/ann_residual.npz and ann_residual.meta.json.
Does not train kNN as the product model and does not download measured waveforms.
sklearn is not available here; the trainer is a NumPy MLP.
Synthetic scores are not laboratory accuracy.
"""

from __future__ import annotations

import json

from physics_model.ann_model import save_artifact, train_ann


def main() -> None:
    model, scores = train_ann()
    path = save_artifact(model, scores)
    summary = {
        "artifact": str(path),
        "version": model.version,
        "active_heads": model.active_summary(),
        "specialists": [spec.audit_record() for spec in model.specialists],
        "validation": {
            name: {
                "ann": scores["validation"]["ann"][name],
                "zero_residual": scores["validation"]["zero_residual"][name],
                "weighted_knn_k7": scores["validation"]["weighted_knn_k7"][name],
                "applied": scores["validation"]["applied"][name],
            }
            for name in ("front_us", "tail_us", "crest_kv")
        },
        "hidden_test": {
            name: {
                "ann": scores["hidden_test"]["ann"][name],
                "zero_residual": scores["hidden_test"]["zero_residual"][name],
                "weighted_knn_k7": scores["hidden_test"]["weighted_knn_k7"][name],
                "applied": scores["hidden_test"]["applied"][name],
            }
            for name in ("front_us", "tail_us", "crest_kv")
        },
    }
    print(json.dumps(summary, indent=2))
    print(
        "Targets are synthetic measured-minus-physics residuals. "
        "The crest head is diagnostic only and is not applied. "
        "No compatible open measured rows were used. "
        "These scores are not laboratory accuracy."
    )


if __name__ == "__main__":
    main()
