"""ANN residual corrector for front, tail, and crest scalars.

Product model: one NumPy MLP per output and per impulse type. sklearn is not
installed in this environment (Python 3.14), so the network is NumPy only.
Weighted kNN (k=7) is a workbook comparison computed on the same splits. The
optimiser and ``/api/optimise`` must not call it.

Targets are synthetic residuals, observed minus the physics baseline. The
network is never trained to predict the requested front, tail, or crest, and it
is never trained to cancel the optimiser.

``Physics_Crest_kV`` equals ``Test_kV`` on every synthetic row (workbook E20).
The crest head is trained only as a diagnostic of that scatter. It is not
applied: beating a zero residual would still move the loaded crest off the
request by memorising the tautology. Polarity is not a column and is not used.
No waveform samples and no observed/residual columns are inputs.

Validation chooses depth, width, weight decay, and the early-stopping epoch.
Hidden Test is scored once after that choice. A timing specialist whose
one-shot hidden-test MAE is not strictly better than a zero residual is not
applied; the weights are not retuned. Synthetic scores are not laboratory
accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from physics_model.inputs import ImpulseType
from physics_model.ml_interface import (
    DATASET_PATH,
    KNN_DISTANCE_SCALES,
    KNN_NEIGHBOURS,
    MLFeatures,
    MLHandoff,
    ResidualCorrection,
    knn_distance,
)

__all__ = [
    "FEATURE_ORDER",
    "TARGET_ORDER",
    "OUTPUT_NAMES",
    "ARTIFACT_DIR",
    "ARTIFACT_NPZ",
    "ARTIFACT_META",
    "MODEL_VERSION",
    "CREST_INACTIVE_REASON",
    "NO_IMPROVEMENT_REASON",
    "AnnResidualModel",
    "Specialist",
    "features_from_handoff",
    "features_from_row",
    "load_dataset_splits",
    "train_ann",
    "benchmark_weighted_knn",
    "zero_residual_baseline",
    "regression_scores",
    "activation_decision",
    "save_artifact",
    "load_artifact",
    "load_training_history",
    "reevaluate_artifact",
    "residuals_are_usable",
]

FEATURE_ORDER: tuple[str, ...] = (
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
)

# Group is a display category only. It is not an extra model input.
FEATURE_PRESENTATION: tuple[tuple[str, str, str], ...] = (
    ("is_switching", "test_request", "Switching"),
    ("Test_kV", "test_request", "Test kV"),
    ("Load_C_pF", "circuit", "Load C"),
    ("Divider_C_pF", "circuit", "Divider C"),
    ("Stray_C_pF", "circuit", "Stray C"),
    ("L_uH", "circuit", "Inductance"),
    ("Efficiency", "circuit", "Efficiency"),
    ("Front_R_Stage", "circuit", "Front R"),
    ("Tail_R_Stage", "circuit", "Tail R"),
    ("Physics_FrontPeak_us", "physics", "Physics front"),
    ("Physics_Tail_us", "physics", "Physics tail"),
    ("Physics_Crest_kV", "physics", "Physics crest"),
)

REFERENCE_ARCHITECTURE = {
    "input_nodes": 16,
    "hidden_layers": [32, 32],
    "hidden_activation": "relu",
    "outputs": ["front_us", "tail_us", "crest_kv"],
    "shared_network": True,
}

ARCHITECTURE_DISCREPANCY = (
    "The reference layout is 16 inputs, two hidden layers of 32 ReLU neurons, "
    "and 3 residual outputs in one network. This saved model has 12 inputs and "
    "a separate tanh network for each residual and impulse type. Polarity is not "
    "in the training table. Requested front time and tail time are not stored as "
    "features. Stage count and charging voltage are in the table but are not "
    "inputs of this artifact. The diagram draws the networks that actually run."
)

TARGET_ORDER: tuple[str, ...] = (
    "Residual_FrontPeak",
    "Residual_Tail",
    "Residual_Crest",
)

OUTPUT_NAMES: tuple[str, ...] = ("front_us", "tail_us", "crest_kv")

_OUTPUT_TO_COLUMN: dict[str, str] = {
    "front_us": "Residual_FrontPeak",
    "tail_us": "Residual_Tail",
    "crest_kv": "Residual_Crest",
}

_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "Observed_FrontPeak_us",
        "Observed_Tail_us",
        "Observed_Crest_kV",
        "Residual_FrontPeak",
        "Residual_Tail",
        "Residual_Crest",
        "Polarity",
    }
)

MODEL_VERSION = "ann_residual_v2"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_NPZ = ARTIFACT_DIR / "ann_residual.npz"
ARTIFACT_META = ARTIFACT_DIR / "ann_residual.meta.json"
TRAINING_HISTORY_PATH = ARTIFACT_DIR / "ann_training_history.json"
TRAINING_SNAPSHOTS_PATH = ARTIFACT_DIR / "ann_training_snapshots.npz"

SKLEARN_REASON = (
    "sklearn does not import in .venv (Python 3.14; ModuleNotFoundError). "
    "The product model is a NumPy MLP."
)

CREST_INACTIVE_REASON = (
    "Physics_Crest_kV equals Test_kV on every synthetic row (workbook E20 tautology). "
    "This head only fits scatter around that request echo. It is not applied, so the "
    "loaded crest N*V_stage*eta stays put. Beating a zero residual is not enough: "
    "there is no independent crest measurement in this file."
)

NO_IMPROVEMENT_REASON = "no held-out improvement; physics-only for this output"

HIDDEN_VETO_REASON = (
    "validation MAE beat zero residual, but the one-shot hidden-test MAE did not. "
    "Not applied. Hyperparameters were not retuned on hidden test."
)

_SANITY = {
    ImpulseType.LIGHTNING: (2.0, 20.0, 80.0),
    ImpulseType.SWITCHING: (80.0, 400.0, 80.0),
}

_RESISTANCE_SCALES = {
    "Lightning": 100.0,
    "Switching": 30000.0,
}

# Validation-only grid. Hidden Test is not an input to this list.
_SELECTION_GRID: tuple[tuple[tuple[int, ...], float, float], ...] = tuple(
    (sizes, weight_decay, learning_rate)
    for width in (16, 32)
    for sizes in ((width,), (width, max(width // 2, 8)))
    for weight_decay in (1e-4, 1e-3)
    for learning_rate in (0.01, 0.003)
)


def activation_decision(
    validation_mae: float, zero_mae: float, output_name: str
) -> tuple[bool, str]:
    """Ship a head only when validation MAE is strictly better than predicting 0.

    The crest head is never shipped. Its target is scatter around the E20 echo.
    """
    if output_name == "crest_kv":
        return False, CREST_INACTIVE_REASON
    if validation_mae < zero_mae:
        return True, "validation MAE strictly better than a zero residual"
    return False, NO_IMPROVEMENT_REASON


def features_from_row(row: pd.Series | dict) -> np.ndarray:
    if _LEAKAGE_COLUMNS.intersection(FEATURE_ORDER):
        raise AssertionError("feature order includes a leakage column")
    impulse = str(row["Impulse_Type"])
    return np.array(
        [
            1.0 if impulse == ImpulseType.SWITCHING.value else 0.0,
            float(row["Test_kV"]),
            float(row["Load_C_pF"]),
            float(row["Divider_C_pF"]),
            float(row["Stray_C_pF"]),
            float(row["L_uH"]),
            float(row["Efficiency"]),
            float(row["Front_R_Stage"]),
            float(row["Tail_R_Stage"]),
            float(row["Physics_FrontPeak_us"]),
            float(row["Physics_Tail_us"]),
            float(row["Physics_Crest_kV"]),
        ],
        dtype=np.float64,
    )


def features_from_handoff(handoff: MLHandoff) -> np.ndarray:
    features = handoff.features
    return np.array(
        [
            1.0 if features.impulse_type is ImpulseType.SWITCHING else 0.0,
            features.test_kv,
            features.load_c_pf,
            features.divider_c_pf,
            features.stray_c_pf,
            features.l_uh,
            features.efficiency,
            features.front_r_stage,
            features.tail_r_stage,
            handoff.physics_front_peak_us,
            handoff.physics_tail_us,
            handoff.physics_crest_kv,
        ],
        dtype=np.float64,
    )


def features_from_ml_features(
    features: MLFeatures, physics_front: float, physics_tail: float, physics_crest: float
) -> np.ndarray:
    return features_from_handoff(
        MLHandoff(
            features=features,
            physics_front_peak_us=physics_front,
            physics_tail_us=physics_tail,
            physics_crest_kv=physics_crest,
            predicted_crest_kv=physics_crest,
            requested_kv=features.test_kv,
            out_of_domain={},
            caveats=[],
        )
    )


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, len(FEATURE_ORDER)), dtype=np.float64)
    return np.stack([features_from_row(row) for row in frame.to_dict(orient="records")])


def load_dataset_splits(path: Path | None = None) -> dict[str, pd.DataFrame]:
    csv_path = path or DATASET_PATH
    frame = pd.read_csv(csv_path)
    if "Split" not in frame.columns:
        raise ValueError("synthetic dataset is missing the Split column")
    if "Polarity" in frame.columns:
        raise AssertionError("unexpected polarity column; do not silently use it")
    missing = [name for name in TARGET_ORDER if name not in frame.columns]
    if missing:
        raise ValueError(f"synthetic dataset is missing residual columns {missing}")
    train = frame[frame["Split"] == "Train"].copy()
    validation = frame[frame["Split"] == "Validation"].copy()
    hidden = frame[frame["Split"] == "Hidden Test"].copy()
    if train.empty:
        raise ValueError("Train split is empty")
    if set(train["Split"].unique()) != {"Train"}:
        raise ValueError(f"Train slice contains other splits: {train['Split'].unique()}")
    if set(train["ID"]).intersection(hidden["ID"]):
        raise ValueError("Train split leaked Hidden Test rows")
    if (train["Physics_Crest_kV"] - train["Test_kV"]).abs().max() != 0.0:
        raise ValueError("Physics_Crest_kV is not the Test_kV echo on the Train split")
    return {
        "train": train,
        "validation": validation,
        "hidden_test": hidden,
        "all": frame,
    }


def _targets(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, list(TARGET_ORDER)].to_numpy(dtype=np.float64)


def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def _mlp_forward(
    x: np.ndarray, weights: tuple[np.ndarray, ...], biases: tuple[np.ndarray, ...]
) -> np.ndarray:
    h = x
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w + b
        if i != last:
            h = np.tanh(h)
    return h


@dataclass
class Specialist:
    """One MLP for one output and one impulse type."""

    output_name: str
    impulse: str
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    active: bool
    reason: str
    hyperparams: dict
    validation_mae: float
    zero_residual_mae: float
    hidden_test_mae: float | None = None
    zero_residual_hidden_mae: float | None = None
    history: tuple[dict, ...] = ()
    epochs_executed: int = 0
    best_epoch: int = 0
    example: dict | None = None

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"feature width {None if x.ndim != 2 else x.shape[1]} does not match {len(FEATURE_ORDER)}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite ANN features")
        z = _standardize(x, self.feature_mean, self.feature_scale)
        raw = _mlp_forward(z, self.weights, self.biases).reshape(-1)
        return raw * self.target_scale + self.target_mean

    def audit_record(self) -> dict:
        return {
            "output": self.output_name,
            "impulse": self.impulse,
            "active": self.active,
            "reason": self.reason,
            "hyperparams": self.hyperparams,
            "validation_mae": self.validation_mae,
            "zero_residual_validation_mae": self.zero_residual_mae,
            "hidden_test_mae": self.hidden_test_mae,
            "zero_residual_hidden_test_mae": self.zero_residual_hidden_mae,
        }


@dataclass
class AnnResidualModel:
    """Per-output, per-impulse residual MLPs. Inactive heads predict 0."""

    specialists: tuple[Specialist, ...]
    feature_order: tuple[str, ...]
    version: str
    synthetic_only: bool
    train_split: str
    notes: dict = field(default_factory=dict)

    def specialist(self, output_name: str, impulse: str) -> Specialist:
        for item in self.specialists:
            if item.output_name == output_name and item.impulse == impulse:
                return item
        raise KeyError(f"no specialist for {output_name} / {impulse}")

    def applies_timing(self, impulse: ImpulseType) -> bool:
        name = impulse.value
        return any(
            self.specialist(output, name).active for output in ("front_us", "tail_us")
        )

    def applies_crest(self, impulse: ImpulseType) -> bool:
        return self.specialist("crest_kv", impulse.value).active

    def public_head_status(self, impulse: ImpulseType) -> dict[str, dict[str, object]]:
        name = impulse.value
        status: dict[str, dict[str, object]] = {}
        for output in OUTPUT_NAMES:
            spec = self.specialist(output, name)
            status[output] = {
                "applied": spec.active,
                "impulse": name,
                "reason": spec.reason,
            }
        return status

    def physics_only_reason(self, impulse: ImpulseType) -> str:
        name = impulse.value
        parts = [
            f"{output}: {self.specialist(output, name).reason}"
            for output in OUTPUT_NAMES
            if not self.specialist(output, name).active
        ]
        return (
            "physics-only: no timing head is applied for this impulse. "
            + " ".join(parts)
            + " Synthetic scores are not laboratory accuracy."
        )

    def applied_message(self, impulse: ImpulseType) -> str:
        name = impulse.value
        on = [output for output in ("front_us", "tail_us") if self.specialist(output, name).active]
        off = [
            f"{output} ({self.specialist(output, name).reason})"
            for output in ("front_us", "tail_us")
            if not self.specialist(output, name).active
        ]
        crest = self.specialist("crest_kv", name)
        text = f"ANN timing residual applied for {', '.join(on)}."
        if off:
            text += " Not applied: " + "; ".join(off) + "."
        text += f" Crest residual is not applied ({crest.reason})"
        text += " Synthetic training is not laboratory accuracy."
        return text

    def active_summary(self) -> dict[str, dict[str, bool]]:
        summary: dict[str, dict[str, bool]] = {}
        for output in OUTPUT_NAMES:
            summary[output] = {
                impulse: self.specialist(output, impulse).active
                for impulse in ("Lightning", "Switching")
            }
        return summary

    def any_timing_head_active(self) -> bool:
        return any(
            self.specialist(output, impulse).active
            for output in ("front_us", "tail_us")
            for impulse in ("Lightning", "Switching")
        )

    def status_detail(self) -> str:
        summary = self.active_summary()
        timing = [
            f"{output}/{impulse}"
            for output, flags in summary.items()
            if output != "crest_kv"
            for impulse, active in flags.items()
            if active
        ]
        crest = "Crest head is not applied (E20 tautology; physics loaded crest is unchanged)."
        if timing:
            applied = ", ".join(timing)
            return (
                f"ANN residual artifact {self.version} loaded. Active timing heads: {applied}. "
                f"{crest} Synthetic training only; not laboratory accuracy."
            )
        return (
            f"ANN residual artifact {self.version} loaded, but no timing head beat a zero "
            f"residual on the ship rule. Physics-only. {crest} Synthetic scores are not "
            "laboratory accuracy."
        )

    def predict_residuals(self, handoff: MLHandoff) -> ResidualCorrection:
        if tuple(self.feature_order) != FEATURE_ORDER:
            raise ValueError("artifact feature order does not match inference FEATURE_ORDER")
        if self.version != MODEL_VERSION:
            raise ValueError(
                f"stale ANN artifact version {self.version}; expected {MODEL_VERSION}"
            )
        x = features_from_handoff(handoff)
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite ANN features")
        impulse = handoff.features.impulse_type.value
        values: list[float] = []
        for output in OUTPUT_NAMES:
            spec = self.specialist(output, impulse)
            if not spec.active:
                values.append(0.0)
                continue
            raw = float(spec.predict_array(x.reshape(1, -1))[0])
            if not math.isfinite(raw):
                raise ValueError("ANN residual output was not finite")
            values.append(raw)
        return ResidualCorrection(
            front_peak_us=values[0],
            tail_us=values[1],
            crest_kv=values[2],
        )

    def _check_artifact_identity(self) -> None:
        if tuple(self.feature_order) != FEATURE_ORDER:
            raise ValueError("artifact feature order does not match inference FEATURE_ORDER")
        if self.version != MODEL_VERSION:
            raise ValueError(
                f"stale ANN artifact version {self.version}; expected {MODEL_VERSION}"
            )

    def raw_output_vector(self, handoff: MLHandoff) -> dict[str, float]:
        """Network outputs before the apply/reject gate. Inactive heads are still scored."""
        self._check_artifact_identity()
        x = features_from_handoff(handoff)
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite ANN features")
        impulse = handoff.features.impulse_type.value
        values: dict[str, float] = {}
        for output in OUTPUT_NAMES:
            spec = self.specialist(output, impulse)
            raw = float(spec.predict_array(x.reshape(1, -1))[0])
            if not math.isfinite(raw):
                raise ValueError("ANN residual output was not finite")
            values[output] = raw
        return values

    def diagram(self, impulse: str) -> dict:
        """Learned architecture and normalized weights for one impulse type."""
        self._check_artifact_identity()
        if impulse not in ("Lightning", "Switching"):
            raise ValueError(f"unknown impulse type {impulse}")
        heads = []
        for output in OUTPUT_NAMES:
            spec = self.specialist(output, impulse)
            hidden = [int(weight.shape[1]) for weight in spec.weights[:-1]]
            heads.append(
                {
                    "output": output,
                    "label": {"front_us": "Front", "tail_us": "Tail", "crest_kv": "Crest"}[output],
                    "unit": "kV" if output == "crest_kv" else "us",
                    "active": spec.active,
                    "reason": spec.reason,
                    "hidden_activation": "tanh",
                    "output_activation": "linear",
                    "layer_sizes": [int(spec.weights[0].shape[0]), *hidden, 1],
                    "hidden_sizes": hidden,
                    "matrices": [_matrix_payload(weight) for weight in spec.weights],
                }
            )
        return {
            "impulse": impulse,
            "input_features": [
                {"name": name, "group": group, "label": label}
                for name, group, label in FEATURE_PRESENTATION
            ],
            "input_count": len(FEATURE_ORDER),
            "heads": heads,
            "shared_hidden_layers": False,
            "hidden_activation": "tanh",
        }


def _matrix_payload(matrix: np.ndarray) -> dict:
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0 or not math.isfinite(scale):
        scale = 1.0
    normalized = np.round(matrix / scale, 4)
    return {
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "max_abs_weight": scale,
        "normalized": normalized.astype(float).tolist(),
    }


def residuals_are_usable(
    front_us: float,
    tail_us: float,
    crest_kv: float,
    residuals: ResidualCorrection,
    impulse_type: ImpulseType,
) -> str | None:
    """Return a reason the residual must be ignored, or None if it can be applied."""
    values = (
        residuals.front_peak_us,
        residuals.tail_us,
        residuals.crest_kv,
        front_us,
        tail_us,
        crest_kv,
    )
    if any(not math.isfinite(v) for v in values):
        return "non-finite residual or corrected scalar"
    if front_us <= 0.0 or tail_us <= 0.0 or crest_kv <= 0.0:
        return "corrected front, tail or crest is not positive"
    if front_us >= tail_us:
        return "corrected front is not shorter than corrected tail"
    front_lim, tail_lim, crest_lim = _SANITY[impulse_type]
    if abs(residuals.front_peak_us) > front_lim:
        return "front residual outside sanity bound"
    if abs(residuals.tail_us) > tail_lim:
        return "tail residual outside sanity bound"
    if abs(residuals.crest_kv) > crest_lim:
        return "crest residual outside sanity bound"
    return None


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_sizes: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    patience: int,
) -> dict:
    """Train on ``x_train`` only. Early stopping watches ``y_val`` MAE. No test rows."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale < 1e-12:
        y_scale = 1.0
    xt = _standardize(x_train, mean, scale)
    xv = _standardize(x_val, mean, scale)
    yt = (y_train - y_mean) / y_scale
    rng = np.random.default_rng(seed)
    dims = [xt.shape[1], *hidden_sizes, 1]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        weights.append(rng.normal(0.0, 1.0 / math.sqrt(fan_in), size=(fan_in, fan_out)))
        biases.append(np.zeros(fan_out, dtype=np.float64))
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    best_mae = float("inf")
    best_weights: tuple[np.ndarray, ...] | None = None
    best_biases: tuple[np.ndarray, ...] | None = None
    best_epoch = 0
    stall = 0
    history: list[dict[str, float | int]] = []
    n = xt.shape[0]
    weight_tuple = tuple(weights)
    bias_tuple = tuple(biases)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = xt[idx]
            yb = yt[idx]
            acts = [xb]
            h = xb
            last = len(weights) - 1
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = h @ w + b
                h = z if i == last else np.tanh(z)
                acts.append(h)
            pred = h.reshape(-1)
            delta = (2.0 * (pred - yb) / max(len(idx), 1)).reshape(-1, 1)
            grads_w: list[np.ndarray] = [np.zeros_like(w) for w in weights]
            grads_b: list[np.ndarray] = [np.zeros_like(b) for b in biases]
            for i in reversed(range(len(weights))):
                grads_w[i] = acts[i].T @ delta + weight_decay * weights[i]
                grads_b[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ weights[i].T) * (1.0 - acts[i] ** 2)
            step += 1
            b1c = 1.0 - beta1**step
            b2c = 1.0 - beta2**step
            for i in range(len(weights)):
                m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * grads_w[i]
                v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (grads_w[i] ** 2)
                weights[i] = weights[i] - learning_rate * (m_w[i] / b1c) / (
                    np.sqrt(v_w[i] / b2c) + eps
                )
                m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grads_b[i]
                v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (grads_b[i] ** 2)
                biases[i] = biases[i] - learning_rate * (m_b[i] / b1c) / (
                    np.sqrt(v_b[i] / b2c) + eps
                )
        current_weights = tuple(weights)
        current_biases = tuple(biases)
        pred_v = (
            _mlp_forward(xv, current_weights, current_biases).reshape(-1) * y_scale + y_mean
        )
        pred_train = (
            _mlp_forward(xt, current_weights, current_biases).reshape(-1) * y_scale + y_mean
        )
        val_err = pred_v - y_val
        train_err = pred_train - y_train
        mae = float(np.mean(np.abs(val_err)))
        train_mae = float(np.mean(np.abs(train_err)))
        train_loss = float(np.mean(train_err**2))
        val_loss = float(np.mean(val_err**2))
        if not all(math.isfinite(v) for v in (mae, train_mae, train_loss, val_loss)):
            break
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mae": train_mae,
                "val_mae": mae,
                "weights": tuple(np.array(w, copy=True, dtype=np.float64) for w in weights),
                "biases": tuple(np.array(b, copy=True, dtype=np.float64) for b in biases),
            }
        )
        if mae < best_mae:
            best_mae = mae
            best_weights = tuple(w.copy() for w in weights)
            best_biases = tuple(b.copy() for b in biases)
            best_epoch = epoch + 1
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_weights is None or best_biases is None:
        best_weights = tuple(w.copy() for w in weight_tuple)
        best_biases = tuple(b.copy() for b in bias_tuple)
        best_epoch = 0
    return {
        "weights": best_weights,
        "biases": best_biases,
        "feature_mean": mean,
        "feature_scale": scale,
        "target_mean": y_mean,
        "target_scale": y_scale,
        "epochs_ran": best_epoch,
        "epochs_executed": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "hidden_sizes": hidden_sizes,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }


def _make_specialist(
    output_name: str,
    impulse: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    fitted: dict,
    *,
    selection: str,
    example: dict | None = None,
) -> Specialist:
    spec = Specialist(
        output_name=output_name,
        impulse=impulse,
        weights=fitted["weights"],
        biases=fitted["biases"],
        feature_mean=fitted["feature_mean"],
        feature_scale=fitted["feature_scale"],
        target_mean=float(fitted["target_mean"]),
        target_scale=float(fitted["target_scale"]),
        active=False,
        reason="",
        hyperparams={
            "selection": selection,
            "hidden_sizes": list(fitted["hidden_sizes"]),
            "learning_rate": fitted["learning_rate"],
            "weight_decay": fitted["weight_decay"],
            "epochs_ran": fitted["epochs_ran"],
            "epochs_executed": int(fitted.get("epochs_executed") or 0),
            "best_epoch": int(fitted.get("best_epoch") or fitted["epochs_ran"]),
            "early_stopping": "validation_mae",
        },
        validation_mae=float("inf"),
        zero_residual_mae=float(np.mean(np.abs(y_val))),
        history=tuple(fitted.get("history") or ()),
        epochs_executed=int(fitted.get("epochs_executed") or 0),
        best_epoch=int(fitted.get("best_epoch") or fitted["epochs_ran"]),
        example=example,
    )
    pred = spec.predict_array(x_val)
    spec.validation_mae = float(np.mean(np.abs(pred - y_val)))
    active, reason = activation_decision(
        spec.validation_mae, spec.zero_residual_mae, output_name
    )
    spec.active = active
    spec.reason = reason
    return spec


def _train_example(frame: pd.DataFrame, impulse: str) -> dict:
    """One fixed Train row so the replay can draw a forward pass. Not a live shot."""
    row = frame.iloc[0]
    return {
        "id": int(row["ID"]),
        "split": "Train",
        "impulse": impulse,
        "features": [float(value) for value in features_from_row(row)],
        "label": (
            f"Train split row ID {int(row['ID'])} ({impulse}). "
            "Fixed replay example, not a live generator shot."
        ),
    }


def _select_specialists(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int,
    batch_size: int,
) -> tuple[Specialist, ...]:
    """Pick hyperparameters on Validation. This function does not receive Hidden Test."""
    chosen: list[Specialist] = []
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        column = _OUTPUT_TO_COLUMN[output_name]
        for impulse_index, impulse in enumerate(("Lightning", "Switching")):
            train_i = train[train["Impulse_Type"] == impulse]
            val_i = validation[validation["Impulse_Type"] == impulse]
            if train_i.empty or val_i.empty:
                raise ValueError(f"missing {impulse} rows for {output_name}")
            x_train = feature_matrix(train_i)
            y_train = train_i[column].to_numpy(dtype=np.float64)
            x_val = feature_matrix(val_i)
            y_val = val_i[column].to_numpy(dtype=np.float64)
            best_fitted: dict | None = None
            best_mae = float("inf")
            for config_index, (sizes, weight_decay, learning_rate) in enumerate(_SELECTION_GRID):
                fitted = _fit_mlp(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    hidden_sizes=sizes,
                    epochs=250,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    seed=seed + 1000 * output_index + 100 * impulse_index + config_index,
                    patience=40,
                )
                probe = Specialist(
                    output_name=output_name,
                    impulse=impulse,
                    weights=fitted["weights"],
                    biases=fitted["biases"],
                    feature_mean=fitted["feature_mean"],
                    feature_scale=fitted["feature_scale"],
                    target_mean=float(fitted["target_mean"]),
                    target_scale=float(fitted["target_scale"]),
                    active=False,
                    reason="",
                    hyperparams={},
                    validation_mae=float("inf"),
                    zero_residual_mae=float(np.mean(np.abs(y_val))),
                )
                mae = float(np.mean(np.abs(probe.predict_array(x_val) - y_val)))
                if mae < best_mae:
                    best_mae = mae
                    best_fitted = fitted
            if best_fitted is None:
                raise RuntimeError(f"no configuration trained for {output_name}/{impulse}")
            chosen.append(
                _make_specialist(
                    output_name,
                    impulse,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    best_fitted,
                    selection="validation_grid",
                    example=_train_example(train_i, impulse),
                )
            )
    return tuple(chosen)


def _fit_fixed_specialists(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[Specialist, ...]:
    """Small fixed run for tests. Still gates on validation. Does not see Hidden Test."""
    chosen: list[Specialist] = []
    for output_index, output_name in enumerate(OUTPUT_NAMES):
        column = _OUTPUT_TO_COLUMN[output_name]
        for impulse_index, impulse in enumerate(("Lightning", "Switching")):
            train_i = train[train["Impulse_Type"] == impulse]
            val_i = validation[validation["Impulse_Type"] == impulse]
            x_train = feature_matrix(train_i)
            y_train = train_i[column].to_numpy(dtype=np.float64)
            x_val = feature_matrix(val_i)
            y_val = val_i[column].to_numpy(dtype=np.float64)
            fitted = _fit_mlp(
                x_train,
                y_train,
                x_val,
                y_val,
                hidden_sizes=(hidden,),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=1e-4,
                seed=seed + 10 * output_index + impulse_index,
                patience=max(epochs, 1),
            )
            chosen.append(
                _make_specialist(
                    output_name,
                    impulse,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    fitted,
                    selection="fixed",
                    example=_train_example(train_i, impulse),
                )
            )
    return tuple(chosen)


def train_ann(
    *,
    dataset_path: Path | None = None,
    hidden: int | None = None,
    epochs: int | None = None,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    seed: int = 0,
) -> tuple[AnnResidualModel, dict]:
    """Train on Split==Train. Select on Validation. Score Hidden Test once, after that.

    Passing ``hidden`` or ``epochs`` skips the validation grid and fits that fixed
    shape. The crest head is still not applied. Used by tests.
    """
    splits = load_dataset_splits(dataset_path)
    train = splits["train"]
    validation = splits["validation"]
    if hidden is None and epochs is None:
        specialists = _select_specialists(
            train, validation, seed=seed, batch_size=batch_size
        )
        selection = "validation_grid"
    else:
        specialists = _fit_fixed_specialists(
            train,
            validation,
            hidden=16 if hidden is None else hidden,
            epochs=40 if epochs is None else epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        selection = "fixed"
    model = AnnResidualModel(
        specialists=specialists,
        feature_order=FEATURE_ORDER,
        version=MODEL_VERSION,
        synthetic_only=True,
        train_split="Train",
        notes={"selection": selection},
    )
    scores = evaluate_model(model, splits)
    _apply_hidden_test_ship_veto(model, scores)
    _refresh_applied_scores(model, splits, scores)
    scores["active_heads"] = model.active_summary()
    scores["selection"] = {
        "split": "Validation",
        "criterion": "mae_strictly_less_than_zero_residual",
        "mode": selection,
        "hidden_test_used_for_hyperparameters": False,
        "hidden_test_ship_veto": True,
        "ship_veto_rule": HIDDEN_VETO_REASON,
        "separate_impulse_heads": True,
        "separate_impulse_reason": (
            "Lightning and switching residuals differ by about two orders of magnitude. "
            "A pooled network can lower average MAE by fitting switching and still make "
            "the lightning residual worse than predicting zero."
        ),
    }
    scores["framework"] = "numpy_mlp"
    scores["sklearn_reason"] = SKLEARN_REASON
    return model, scores


def _apply_hidden_test_ship_veto(model: AnnResidualModel, scores: dict) -> None:
    """One look at Hidden Test. Does not change weights or rerun the grid."""
    by_impulse = scores["hidden_test"]["by_impulse"]
    val_by_impulse = scores["validation"]["by_impulse"]
    for spec in model.specialists:
        hidden_block = by_impulse[spec.impulse]
        val_block = val_by_impulse[spec.impulse]
        spec.hidden_test_mae = float(hidden_block["ann"][spec.output_name]["mae"])
        spec.zero_residual_hidden_mae = float(
            hidden_block["zero_residual"][spec.output_name]["mae"]
        )
        spec.validation_mae = float(val_block["ann"][spec.output_name]["mae"])
        spec.zero_residual_mae = float(val_block["zero_residual"][spec.output_name]["mae"])
        if spec.output_name == "crest_kv":
            spec.active = False
            spec.reason = CREST_INACTIVE_REASON
            continue
        if not spec.active:
            continue
        if not (spec.hidden_test_mae < spec.zero_residual_hidden_mae):
            spec.active = False
            spec.reason = HIDDEN_VETO_REASON


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    err = y_pred - y_true
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(OUTPUT_NAMES):
        out[name] = {
            "mae": float(np.mean(np.abs(err[:, i]))),
            "rmse": float(np.sqrt(np.mean(err[:, i] ** 2))),
        }
    return out


def zero_residual_baseline(y_true: np.ndarray) -> dict[str, dict[str, float]]:
    return regression_scores(y_true, np.zeros_like(y_true))


def _row_to_ml_features(row: pd.Series | dict) -> MLFeatures:
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


def knn_distance_matrix(query_frame: pd.DataFrame, train_frame: pd.DataFrame) -> np.ndarray:
    """Pairwise workbook distances. Rows are queries, columns are Train references."""
    columns = list(KNN_DISTANCE_SCALES)
    scales = np.array([KNN_DISTANCE_SCALES[name] for name in columns], dtype=np.float64)
    q_num = query_frame.loc[:, columns].to_numpy(dtype=np.float64)
    t_num = train_frame.loc[:, columns].to_numpy(dtype=np.float64)
    q_imp = query_frame["Impulse_Type"].to_numpy()
    t_imp = train_frame["Impulse_Type"].to_numpy()
    diff = (q_num[:, None, :] - t_num[None, :, :]) / scales
    distances = np.sum(diff * diff, axis=2)
    distances += (q_imp[:, None] != t_imp[None, :]).astype(np.float64)
    ref_scale = np.where(
        t_imp == ImpulseType.SWITCHING.value,
        _RESISTANCE_SCALES["Switching"],
        _RESISTANCE_SCALES["Lightning"],
    ).astype(np.float64)
    q_front = query_frame["Front_R_Stage"].to_numpy(dtype=np.float64)
    t_front = train_frame["Front_R_Stage"].to_numpy(dtype=np.float64)
    q_tail = query_frame["Tail_R_Stage"].to_numpy(dtype=np.float64)
    t_tail = train_frame["Tail_R_Stage"].to_numpy(dtype=np.float64)
    distances += ((q_front[:, None] - t_front[None, :]) / ref_scale) ** 2
    distances += ((q_tail[:, None] - t_tail[None, :]) / ref_scale) ** 2
    return distances


def benchmark_weighted_knn(
    query_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    k: int = KNN_NEIGHBOURS,
) -> np.ndarray:
    """Workbook-distance weighted kNN. Comparison baseline only. Not the product model.

    The distance matches :func:`physics_model.ml_interface.knn_distance`, including
    the type-mismatch penalty and the resistance scale taken from the reference row.
    """
    distances = knn_distance_matrix(query_frame, train_frame)
    train_y = train_frame.loc[:, list(TARGET_ORDER)].to_numpy(dtype=np.float64)
    k_use = min(k, distances.shape[1])
    if k_use < 1:
        raise ValueError("kNN training frame is empty")
    nearest = np.argpartition(distances, k_use - 1, axis=1)[:, :k_use]
    row = np.arange(distances.shape[0])[:, None]
    chosen_d = distances[row, nearest]
    chosen_y = train_y[nearest]
    weights = 1.0 / (chosen_d + 1e-6)
    return (weights[:, :, None] * chosen_y).sum(axis=1) / weights.sum(axis=1)[:, None]


def predict_network(model: AnnResidualModel, frame: pd.DataFrame) -> np.ndarray:
    """Ungated network residuals. Active flags are ignored so scores stay auditable."""
    x = feature_matrix(frame)
    impulses = frame["Impulse_Type"].to_numpy()
    out = np.zeros((len(frame), len(OUTPUT_NAMES)), dtype=np.float64)
    for index, output in enumerate(OUTPUT_NAMES):
        for impulse in ("Lightning", "Switching"):
            mask = impulses == impulse
            if not np.any(mask):
                continue
            out[mask, index] = model.specialist(output, impulse).predict_array(x[mask])
    return out


def predict_applied(model: AnnResidualModel, frame: pd.DataFrame) -> np.ndarray:
    """Residuals the API would add. Inactive specialists contribute 0."""
    raw = predict_network(model, frame)
    impulses = frame["Impulse_Type"].to_numpy()
    for index, output in enumerate(OUTPUT_NAMES):
        for impulse in ("Lightning", "Switching"):
            if model.specialist(output, impulse).active:
                continue
            raw[impulses == impulse, index] = 0.0
    return raw


def _by_impulse_scores(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    network: np.ndarray,
    applied: np.ndarray,
    knn: np.ndarray,
) -> dict[str, dict]:
    impulses = frame["Impulse_Type"].to_numpy()
    blocks: dict[str, dict] = {}
    for impulse in ("Lightning", "Switching"):
        mask = impulses == impulse
        blocks[impulse] = {
            "n": int(mask.sum()),
            "ann": regression_scores(y_true[mask], network[mask]),
            "applied": regression_scores(y_true[mask], applied[mask]),
            "zero_residual": zero_residual_baseline(y_true[mask]),
            "weighted_knn_k7": regression_scores(y_true[mask], knn[mask]),
        }
    return blocks


def _score_frame(
    model: AnnResidualModel, frame: pd.DataFrame, train: pd.DataFrame
) -> dict:
    y_true = _targets(frame)
    network = predict_network(model, frame)
    applied = predict_applied(model, frame)
    knn = benchmark_weighted_knn(frame, train)
    return {
        "ann": regression_scores(y_true, network),
        "applied": regression_scores(y_true, applied),
        "zero_residual": zero_residual_baseline(y_true),
        "weighted_knn_k7": regression_scores(y_true, knn),
        "by_impulse": _by_impulse_scores(frame, y_true, network, applied, knn),
    }


def _refresh_applied_scores(
    model: AnnResidualModel, splits: dict[str, pd.DataFrame], scores: dict
) -> None:
    for key in ("validation", "hidden_test"):
        frame = splits[key]
        y_true = _targets(frame)
        applied = predict_applied(model, frame)
        scores[key]["applied"] = regression_scores(y_true, applied)
        impulses = frame["Impulse_Type"].to_numpy()
        for impulse in ("Lightning", "Switching"):
            mask = impulses == impulse
            scores[key]["by_impulse"][impulse]["applied"] = regression_scores(
                y_true[mask], applied[mask]
            )


def evaluate_model(model: AnnResidualModel, splits: dict[str, pd.DataFrame]) -> dict:
    """Score Validation and Hidden Test. Call this only after hyperparameters are frozen."""
    train = splits["train"]
    return {
        "validation": _score_frame(model, splits["validation"], train),
        "hidden_test": _score_frame(model, splits["hidden_test"], train),
        "n_train": int(len(train)),
        "n_validation": int(len(splits["validation"])),
        "n_hidden_test": int(len(splits["hidden_test"])),
        "crest_note": CREST_INACTIVE_REASON,
        "open_measured_rows_used": False,
        "open_measured_rows_note": (
            "data/literature/open_references.json is cited typical values, not a table "
            "of measured front/tail/crest rows in the synthetic schema. No compatible "
            "open measured rows are in the repo. Training uses the synthetic Train split only."
        ),
        "scores_are_synthetic": True,
        "laboratory_accuracy": False,
        "active_heads": model.active_summary(),
    }


def _artifact_sha256(model: AnnResidualModel) -> str:
    digest = hashlib.sha256()
    digest.update(model.version.encode())
    digest.update("\0".join(model.feature_order).encode())
    for spec in sorted(model.specialists, key=lambda item: (item.output_name, item.impulse)):
        digest.update(spec.output_name.encode())
        digest.update(spec.impulse.encode())
        digest.update(b"\1" if spec.active else b"\0")
        digest.update(spec.reason.encode())
        arrays = (
            *spec.weights,
            *spec.biases,
            spec.feature_mean,
            spec.feature_scale,
            np.array([spec.target_mean, spec.target_scale], dtype=np.float64),
        )
        for array in arrays:
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _specialist_prefix(output_name: str, impulse: str) -> str:
    return f"{output_name}__{impulse}"


def _public_history(history: tuple[dict, ...] | list[dict]) -> list[dict]:
    """Scalar rows only. Weight snapshots live in the npz file."""
    rows = []
    for row in history:
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_mae": float(row["train_mae"]),
                "val_mae": float(row["val_mae"]),
            }
        )
    return rows


def training_history_document(model: AnnResidualModel) -> dict:
    """Scalar training telemetry. Exact per-epoch weights are stored beside this file.

    This is synthetic-training telemetry. It is not a laboratory measurement.
    """
    heads = []
    for spec in model.specialists:
        hidden_sizes = list((spec.hyperparams or {}).get("hidden_sizes") or [])
        n_layers = len(spec.history[0]["weights"]) if spec.history else 0
        heads.append(
            {
                "output": spec.output_name,
                "impulse": spec.impulse,
                "active": spec.active,
                "reason": spec.reason,
                "units": "kV" if spec.output_name == "crest_kv" else "us",
                "architecture": {
                    "input_size": len(FEATURE_ORDER),
                    "input_features": list(FEATURE_ORDER),
                    "hidden_sizes": hidden_sizes,
                    "hidden_activation": "tanh",
                    "output_size": 1,
                    "output_name": _OUTPUT_TO_COLUMN[spec.output_name],
                },
                "epochs_executed": spec.epochs_executed,
                "best_epoch": spec.best_epoch,
                "zero_residual_validation_mae": spec.zero_residual_mae,
                "n_weight_layers": n_layers,
                "feature_mean": [float(value) for value in spec.feature_mean],
                "feature_scale": [float(value) for value in spec.feature_scale],
                "target_mean": float(spec.target_mean),
                "target_scale": float(spec.target_scale),
                "example": spec.example,
                "history": _public_history(spec.history),
            }
        )
    epoch_count = max((int(head["epochs_executed"]) for head in heads), default=0)
    return {
        "available": True,
        "synthetic_only": True,
        "laboratory_accuracy": False,
        "scores_are_synthetic": True,
        "disclaimer": (
            "Synthetic training replay from the Train split, with Validation used "
            "for the kept checkpoint. Not laboratory validation and not a live controller."
        ),
        "loss_definition": (
            "train_loss and val_loss are the mean squared residual in original "
            "units (microseconds or kV) after each epoch. train_mae and val_mae "
            "are the mean absolute residual on the same splits."
        ),
        "weight_scale": (
            "Connection width and opacity use one fixed scale per network: the "
            "maximum absolute weight across every stored epoch of that network. "
            "Positive weights are blue and negative weights are red."
        ),
        "weights_recorded_per_epoch": True,
        "version": model.version,
        "feature_order": list(model.feature_order),
        "epoch_count": epoch_count,
        "active_heads": model.active_summary(),
        "heads": heads,
    }


def _snapshot_arrays(model: AnnResidualModel) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for spec in model.specialists:
        prefix = _specialist_prefix(spec.output_name, spec.impulse)
        for row in spec.history:
            epoch = int(row["epoch"])
            weights = row.get("weights")
            biases = row.get("biases")
            if not weights or not biases:
                raise ValueError(f"missing weight snapshot for {prefix} epoch {epoch}")
            for index, weight in enumerate(weights):
                arrays[f"{prefix}__e{epoch:04d}__w{index}"] = np.ascontiguousarray(
                    weight, dtype=np.float64
                )
            for index, bias in enumerate(biases):
                arrays[f"{prefix}__e{epoch:04d}__b{index}"] = np.ascontiguousarray(
                    bias, dtype=np.float64
                )
    return arrays


def save_training_history(model: AnnResidualModel, *, directory: Path | None = None) -> Path:
    directory = directory or ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAINING_HISTORY_PATH.name
    path.write_text(
        json.dumps(training_history_document(model), indent=2),
        encoding="utf-8",
    )
    snapshots = _snapshot_arrays(model)
    if snapshots:
        np.savez_compressed(directory / TRAINING_SNAPSHOTS_PATH.name, **snapshots)
    return path


def load_training_history(directory: Path | None = None) -> dict:
    directory = directory or ARTIFACT_DIR
    path = directory / TRAINING_HISTORY_PATH.name
    if not path.exists():
        raise FileNotFoundError(f"ANN training history missing at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ANN training history is not a JSON object")
    if not payload.get("weights_recorded_per_epoch"):
        return payload
    snap_path = directory / TRAINING_SNAPSHOTS_PATH.name
    if not snap_path.exists():
        raise FileNotFoundError(f"ANN weight snapshots missing at {snap_path}")
    blob = np.load(snap_path)
    try:
        for head in payload.get("heads", []):
            prefix = _specialist_prefix(str(head["output"]), str(head["impulse"]))
            n_layers = int(head["n_weight_layers"])
            for row in head["history"]:
                epoch = int(row["epoch"])
                row["weights"] = [
                    blob[f"{prefix}__e{epoch:04d}__w{index}"].tolist()
                    for index in range(n_layers)
                ]
                row["biases"] = [
                    blob[f"{prefix}__e{epoch:04d}__b{index}"].tolist()
                    for index in range(n_layers)
                ]
    finally:
        blob.close()
    return payload


def save_artifact(model: AnnResidualModel, scores: dict, *, directory: Path | None = None) -> Path:
    directory = directory or ARTIFACT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / ARTIFACT_NPZ.name
    meta_path = directory / ARTIFACT_META.name
    if model.version != MODEL_VERSION:
        raise ValueError(f"refusing to save version {model.version}")
    arrays: dict[str, np.ndarray] = {
        "feature_order": np.array(model.feature_order),
    }
    for spec in model.specialists:
        prefix = _specialist_prefix(spec.output_name, spec.impulse)
        arrays[f"{prefix}__n_layers"] = np.array([len(spec.weights)], dtype=np.int32)
        for index, (weight, bias) in enumerate(zip(spec.weights, spec.biases)):
            arrays[f"{prefix}__w{index}"] = np.ascontiguousarray(weight, dtype=np.float64)
            arrays[f"{prefix}__b{index}"] = np.ascontiguousarray(bias, dtype=np.float64)
        arrays[f"{prefix}__feature_mean"] = np.ascontiguousarray(spec.feature_mean, dtype=np.float64)
        arrays[f"{prefix}__feature_scale"] = np.ascontiguousarray(spec.feature_scale, dtype=np.float64)
        arrays[f"{prefix}__target_mean"] = np.array([spec.target_mean], dtype=np.float64)
        arrays[f"{prefix}__target_scale"] = np.array([spec.target_scale], dtype=np.float64)
        arrays[f"{prefix}__active"] = np.array([1 if spec.active else 0], dtype=np.int32)
    np.savez(npz_path, **arrays)
    scores = dict(scores)
    scores["active_heads"] = model.active_summary()
    meta = {
        "version": model.version,
        "feature_order": list(model.feature_order),
        "target_order": list(TARGET_ORDER),
        "output_names": list(OUTPUT_NAMES),
        "train_split": model.train_split,
        "synthetic_only": True,
        "model_kind": "ann_residual_mlp",
        "framework": "numpy_mlp",
        "sklearn_reason": SKLEARN_REASON,
        "polarity_used": False,
        "waveform_samples_used": False,
        "observed_columns_used_as_inputs": False,
        "trained_on_requested_waveform": False,
        "crest_head_applied": False,
        "crest_residual_limitation": CREST_INACTIVE_REASON,
        "not_a_waveform_network": True,
        "knn_is_benchmark_only": True,
        "scores_are_synthetic": True,
        "laboratory_accuracy": False,
        "open_measured_rows_used": False,
        "artifact_sha256": _artifact_sha256(model),
        "specialists": [spec.audit_record() for spec in model.specialists],
        "active_heads": model.active_summary(),
        "scores": scores,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if any(spec.history for spec in model.specialists):
        save_training_history(model, directory=directory)
    return npz_path


def _load_specialist(blob: np.lib.npyio.NpzFile, output_name: str, impulse: str, record: dict) -> Specialist:
    prefix = _specialist_prefix(output_name, impulse)
    n_layers = int(blob[f"{prefix}__n_layers"][0])
    weights = tuple(np.array(blob[f"{prefix}__w{index}"], dtype=np.float64) for index in range(n_layers))
    biases = tuple(np.array(blob[f"{prefix}__b{index}"], dtype=np.float64) for index in range(n_layers))
    active_npz = bool(int(blob[f"{prefix}__active"][0]))
    if active_npz != bool(record["active"]):
        raise ValueError(
            f"stale ANN artifact: active flag for {output_name}/{impulse} "
            "disagrees between weights and metadata"
        )
    hidden = record.get("hidden_test_mae")
    hidden_zero = record.get("zero_residual_hidden_test_mae")
    return Specialist(
        output_name=output_name,
        impulse=impulse,
        weights=weights,
        biases=biases,
        feature_mean=np.array(blob[f"{prefix}__feature_mean"], dtype=np.float64),
        feature_scale=np.array(blob[f"{prefix}__feature_scale"], dtype=np.float64),
        target_mean=float(blob[f"{prefix}__target_mean"][0]),
        target_scale=float(blob[f"{prefix}__target_scale"][0]),
        active=bool(record["active"]),
        reason=str(record["reason"]),
        hyperparams=dict(record.get("hyperparams") or {}),
        validation_mae=float(record["validation_mae"]),
        zero_residual_mae=float(record["zero_residual_validation_mae"]),
        hidden_test_mae=None if hidden is None else float(hidden),
        zero_residual_hidden_mae=None if hidden_zero is None else float(hidden_zero),
    )


def load_artifact(directory: Path | None = None) -> AnnResidualModel:
    directory = directory or ARTIFACT_DIR
    npz_path = directory / ARTIFACT_NPZ.name
    meta_path = directory / ARTIFACT_META.name
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"ANN artifact missing under {directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = str(meta.get("version", ""))
    if version != MODEL_VERSION:
        raise ValueError(f"stale ANN artifact version {version}; expected {MODEL_VERSION}")
    order = tuple(meta.get("feature_order", ()))
    if order != FEATURE_ORDER:
        raise ValueError(f"artifact feature order {order} != {FEATURE_ORDER}")
    if not bool(meta.get("synthetic_only", False)):
        raise ValueError("refusing ANN artifact that is not marked synthetic_only")
    records = {(item["output"], item["impulse"]): item for item in meta.get("specialists", [])}
    expected = {(output, impulse) for output in OUTPUT_NAMES for impulse in ("Lightning", "Switching")}
    if set(records) != expected:
        raise ValueError("ANN artifact specialist list does not match front/tail/crest × lightning/switching")
    blob = np.load(npz_path)
    specialists = tuple(
        _load_specialist(blob, output, impulse, records[(output, impulse)])
        for output, impulse in (
            (output, impulse) for output in OUTPUT_NAMES for impulse in ("Lightning", "Switching")
        )
    )
    model = AnnResidualModel(
        specialists=specialists,
        feature_order=order,
        version=version,
        synthetic_only=True,
        train_split=str(meta.get("train_split", "Train")),
    )
    expected_hash = str(meta.get("artifact_sha256", ""))
    actual_hash = _artifact_sha256(model)
    if expected_hash != actual_hash:
        raise ValueError(
            "stale or mismatched ANN artifact: metadata hash does not match the weight file"
        )
    return model


def reevaluate_artifact(directory: Path | None = None) -> dict:
    """Recompute validation and hidden-test scores from the saved weights.

    Does not train and does not change active flags. Hidden-test numbers in the
    metadata must match this function.
    """
    model = load_artifact(directory)
    splits = load_dataset_splits()
    scores = evaluate_model(model, splits)
    _refresh_applied_scores(model, splits, scores)
    return scores
