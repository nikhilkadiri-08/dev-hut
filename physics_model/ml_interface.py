"""The contract between the physics layer and the residual ML model.

No model is trained here. This module defines the handoff: exactly which features
the physics layer emits, in exactly the units and column order the existing
synthetic dataset uses, so the ML workstream can train on
``data/synthetic_dataset.csv`` without writing an adapter and without either side
guessing what the other means.

The division of labour, taken from the workbook
-----------------------------------------------
Physics produces a baseline (``E18``, ``E19``, ``E20``). The ML layer predicts the
*residual* between that baseline and what the generator actually does, and cells
``H6``-``H8`` add the two. So the ML model's job is error correction on three
scalars, not waveform prediction. :func:`apply_residuals` is that addition, kept
here so both sides use the same one.

Where the residual targets differ in kind
-----------------------------------------
Two of the three residuals are corrections to a real model. The third is not.
Because ``E20`` is ``=B7``, the physics crest "prediction" is the requested voltage
echoed back, so the crest residual is not a correction to a model - it *is* the
entire voltage-efficiency model, learned from data. That makes it the one residual
that cannot be sanity-checked against physics, and the one most in need of measured
data before anybody trusts it.

The reason this matters operationally
-------------------------------------
Crest error is correctable without touching the generator.
:func:`charging_voltage_trim` turns a predicted crest error into a revised charging
voltage, which is a control-desk setting. Front and tail errors are not: fixing
those means changing resistors, which means somebody climbs. So the right use of
the ML layer is to absorb voltage error into the charging setting and to choose
resistors with enough margin that timing error never needs a second attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend.units import f_to_pf, h_to_uh, kv_to_v, s_to_us, v_to_kv
from physics_model.calculations import Caveat, PhysicsPrediction, Severity
from physics_model.inputs import GeneratorConfiguration, ImpulseType

__all__ = [
    "DATASET_PATH",
    "DATASET_FEATURE_COLUMNS",
    "DATASET_PHYSICS_COLUMNS",
    "DATASET_RESIDUAL_COLUMNS",
    "KNN_DISTANCE_SCALES",
    "KNN_NEIGHBOURS",
    "MLFeatures",
    "ResidualCorrection",
    "VoltageTrim",
    "MLHandoff",
    "knn_distance",
    "training_domain",
    "apply_residuals",
    "charging_voltage_trim",
    "build_handoff",
]

DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "synthetic_dataset.csv"
)

#: Feature columns, in the order the dataset and 'ML Helper' columns A-I use.
DATASET_FEATURE_COLUMNS: tuple[str, ...] = (
    "Impulse_Type",
    "Test_kV",
    "Load_C_pF",
    "Divider_C_pF",
    "Stray_C_pF",
    "L_uH",
    "Efficiency",
    "Front_R_Stage",
    "Tail_R_Stage",
)

#: The physics baselines the ML layer corrects.
DATASET_PHYSICS_COLUMNS: tuple[str, ...] = (
    "Physics_FrontPeak_us",
    "Physics_Tail_us",
    "Physics_Crest_kV",
)

#: The regression targets: observed minus physics, per the dataset's own columns.
DATASET_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "Residual_FrontPeak",
    "Residual_Tail",
    "Residual_Crest",
)

#: Normalisation divisors from the distance formula in 'ML Helper' column J.
#: Efficiency is divided by 1, and the resistance scale depends on impulse type
#: because switching resistances are two orders of magnitude larger.
KNN_DISTANCE_SCALES: dict[str, float] = {
    "Test_kV": 1500.0,
    "Load_C_pF": 1300.0,
    "Divider_C_pF": 800.0,
    "Stray_C_pF": 350.0,
    "L_uH": 30.0,
    "Efficiency": 1.0,
}

#: Resistance divisors, by impulse type, from the same formula.
_RESISTANCE_SCALES: dict[ImpulseType, float] = {
    ImpulseType.LIGHTNING: 100.0,
    ImpulseType.SWITCHING: 30000.0,
}

#: ``k`` from 'ML Helper' column L, ``IF(K2<=7,1,0)``.
KNN_NEIGHBOURS = 7


@dataclass(frozen=True)
class MLFeatures:
    """The nine features the residual model consumes, in laboratory units.

    Units deliberately match the dataset rather than this package's SI convention,
    because the trained model will key on the dataset's scales. Conversion happens
    once, in :func:`build_handoff`, and nowhere else.
    """

    impulse_type: ImpulseType
    test_kv: float
    load_c_pf: float
    divider_c_pf: float
    stray_c_pf: float
    l_uh: float
    efficiency: float
    front_r_stage: float
    tail_r_stage: float

    def as_row(self) -> dict[str, float | str]:
        """A dict keyed by the dataset's own column names."""
        return {
            "Impulse_Type": self.impulse_type.value,
            "Test_kV": self.test_kv,
            "Load_C_pF": self.load_c_pf,
            "Divider_C_pF": self.divider_c_pf,
            "Stray_C_pF": self.stray_c_pf,
            "L_uH": self.l_uh,
            "Efficiency": self.efficiency,
            "Front_R_Stage": self.front_r_stage,
            "Tail_R_Stage": self.tail_r_stage,
        }

    def as_vector(self) -> list[float]:
        """Numeric features only, in dataset order, for an array-based model.

        ``Impulse_Type`` is excluded because it is categorical; the workbook's
        distance treats a type mismatch as a fixed penalty of 1 rather than as a
        number, and :func:`knn_distance` reproduces that.
        """
        return [
            self.test_kv,
            self.load_c_pf,
            self.divider_c_pf,
            self.stray_c_pf,
            self.l_uh,
            self.efficiency,
            self.front_r_stage,
            self.tail_r_stage,
        ]


def knn_distance(query: MLFeatures, reference: MLFeatures) -> float:
    """The workbook's squared distance from 'ML Helper' column J.

    Reproduced exactly, including that it is a squared distance rather than a
    Euclidean one and that the resistance scale is chosen from the *reference*
    row's impulse type. Returned as the workbook returns it, un-rooted, because the
    kNN weighting in column M uses ``1/(distance + 1e-6)`` on this value.
    """
    total = 0.0 if query.impulse_type is reference.impulse_type else 1.0
    for column, scale in KNN_DISTANCE_SCALES.items():
        q = float(query.as_row()[column])
        r = float(reference.as_row()[column])
        total += ((q - r) / scale) ** 2
    resistance_scale = _RESISTANCE_SCALES[reference.impulse_type]
    total += ((query.front_r_stage - reference.front_r_stage) / resistance_scale) ** 2
    total += ((query.tail_r_stage - reference.tail_r_stage) / resistance_scale) ** 2
    return total


@lru_cache(maxsize=1)
def training_domain() -> dict[str, tuple[float, float]] | None:
    """Per-feature ranges of the dataset's **Train** split, or ``None`` if absent.

    Only the Train rows are used. Including the validation or hidden-test rows
    would let evaluation data leak into the in-domain decision, which would make
    the resulting confidence claim circular.

    A configuration outside these ranges is one the residual model has no nearby
    neighbours for, so its correction is an extrapolation. That is worth knowing
    before trusting a prediction enough to set up a shot.
    """
    if not DATASET_PATH.exists():
        return None
    import pandas as pd

    frame = pd.read_csv(DATASET_PATH)
    train = frame[frame["Split"] == "Train"]
    numeric = [c for c in DATASET_FEATURE_COLUMNS if c != "Impulse_Type"]
    return {
        column: (float(train[column].min()), float(train[column].max()))
        for column in numeric
    }


@dataclass(frozen=True)
class ResidualCorrection:
    """ML-predicted residuals, in the dataset's units. Workbook ``H9``-``H11``."""

    front_peak_us: float = 0.0
    tail_us: float = 0.0
    crest_kv: float = 0.0

    @property
    def is_zero(self) -> bool:
        return (
            self.front_peak_us == 0.0
            and self.tail_us == 0.0
            and self.crest_kv == 0.0
        )


@dataclass(frozen=True)
class VoltageTrim:
    """A revised charging voltage that cancels a predicted crest error."""

    original_charging_voltage_v: float
    trimmed_charging_voltage_v: float
    predicted_crest_kv: float
    requested_crest_kv: float
    within_stage_rating: bool
    note: str

    @property
    def change_fraction(self) -> float:
        return (
            self.trimmed_charging_voltage_v / self.original_charging_voltage_v - 1.0
        )


def apply_residuals(
    prediction: PhysicsPrediction, residuals: ResidualCorrection
) -> dict[str, float]:
    """Workbook ``H6``-``H8``: physics baseline plus ML residual.

    Returns the corrected front/peak time and tail time in microseconds and the
    corrected crest in kV. With zero residuals this returns the physics baseline
    unchanged, which is the correct behaviour before any model is trained.
    """
    return {
        "front_peak_us": prediction.front_time_us + residuals.front_peak_us,
        "tail_us": prediction.tail_time_us + residuals.tail_us,
        "crest_kv": prediction.crest_voltage_kv + residuals.crest_kv,
    }


def charging_voltage_trim(
    configuration: GeneratorConfiguration,
    requested_voltage_v: float,
    predicted_crest_kv: float,
) -> VoltageTrim:
    """Revise the charging voltage so the predicted crest lands on the request.

    The output voltage is proportional to the charging voltage in the ideal model
    (``n * Vc * efficiency``), so scaling the charging voltage by
    ``requested / predicted`` cancels a predicted crest error to first order.

    This is the one correction that costs nothing to apply: the charging voltage is
    a control-desk setting, so trimming it does not send anybody up the generator.
    Timing errors have no equivalent, which is why the optimiser spends its effort
    on timing margin instead.

    The proportionality is exact only if the efficiency is itself independent of
    charging voltage. The supplied material takes efficiency as a constant input
    (``B12``) and says nothing about voltage dependence, so for large trims this is
    an assumption that needs measured confirmation.
    """
    requested_kv = v_to_kv(requested_voltage_v)
    original = configuration.charging_voltage_per_stage_v
    if predicted_crest_kv <= 0.0:
        return VoltageTrim(
            original_charging_voltage_v=original,
            trimmed_charging_voltage_v=original,
            predicted_crest_kv=predicted_crest_kv,
            requested_crest_kv=requested_kv,
            within_stage_rating=True,
            note=(
                "Predicted crest is not positive, so no trim can be computed. "
                "Left unchanged."
            ),
        )

    trimmed = original * (requested_kv / predicted_crest_kv)
    rating = configuration.ratings.max_stage_voltage_v
    within = trimmed <= rating
    if abs(trimmed / original - 1.0) < 1e-12:
        note = "Predicted crest already matches the request; no trim needed."
    elif within:
        note = (
            f"Set the charging voltage to {v_to_kv(trimmed):.4g} kV/stage instead of "
            f"{v_to_kv(original):.4g} kV/stage "
            f"({(trimmed / original - 1.0):+.2%}) to bring the predicted crest onto "
            f"{requested_kv:.4g} kV. This is a control-desk change: no physical "
            f"reconfiguration."
        )
    else:
        note = (
            f"Cancelling the predicted crest error would need "
            f"{v_to_kv(trimmed):.4g} kV/stage, above the "
            f"{v_to_kv(rating):.4g} kV stage rating. Use more stages instead; the "
            f"charging voltage cannot absorb this error."
        )
    return VoltageTrim(
        original_charging_voltage_v=original,
        trimmed_charging_voltage_v=trimmed,
        predicted_crest_kv=predicted_crest_kv,
        requested_crest_kv=requested_kv,
        within_stage_rating=within,
        note=note,
    )


@dataclass
class MLHandoff:
    """What the physics layer hands the ML layer for one configuration."""

    features: MLFeatures
    physics_front_peak_us: float
    physics_tail_us: float
    physics_crest_kv: float
    requested_kv: float
    out_of_domain: dict[str, tuple[float, tuple[float, float]]]
    caveats: list[Caveat]

    @property
    def in_training_domain(self) -> bool:
        return not self.out_of_domain

    def as_row(self) -> dict[str, float | str]:
        """One row in the dataset's own schema, ready to append or to predict on."""
        row = self.features.as_row()
        row["Physics_FrontPeak_us"] = self.physics_front_peak_us
        row["Physics_Tail_us"] = self.physics_tail_us
        row["Physics_Crest_kV"] = self.physics_crest_kv
        return row

    def corrected(self, residuals: ResidualCorrection) -> dict[str, float]:
        return {
            "front_peak_us": self.physics_front_peak_us + residuals.front_peak_us,
            "tail_us": self.physics_tail_us + residuals.tail_us,
            "crest_kv": self.physics_crest_kv + residuals.crest_kv,
        }


def build_handoff(prediction: PhysicsPrediction) -> MLHandoff:
    """Package one physics prediction into the ML layer's input contract.

    This is the single conversion point from the package's SI units to the
    dataset's laboratory units.
    """
    configuration = prediction.configuration
    circuit = configuration.circuit
    features = MLFeatures(
        impulse_type=configuration.impulse_type,
        test_kv=v_to_kv(prediction.requested_voltage_v),
        load_c_pf=f_to_pf(circuit.load_capacitance_f),
        divider_c_pf=f_to_pf(circuit.divider_capacitance_f),
        stray_c_pf=f_to_pf(circuit.stray_capacitance_f),
        l_uh=h_to_uh(circuit.stray_inductance_h),
        efficiency=circuit.efficiency,
        front_r_stage=configuration.front_resistance_per_stage_ohm,
        tail_r_stage=configuration.tail_resistance_per_stage_ohm,
    )

    out_of_domain: dict[str, tuple[float, tuple[float, float]]] = {}
    domain = training_domain()
    caveats: list[Caveat] = [
        Caveat(
            "CREST_RESIDUAL_IS_THE_WHOLE_MODEL",
            "Because the physics crest is the requested voltage echoed back, the "
            "crest residual is not a correction to a model; it is the entire "
            "voltage-efficiency model, learned from data. It cannot be "
            "cross-checked against physics.",
            Severity.WARNING,
        )
    ]
    if domain is None:
        caveats.append(
            Caveat(
                "TRAINING_DOMAIN_UNKNOWN",
                f"The training dataset was not found at {DATASET_PATH}, so whether "
                f"this configuration is inside the model's training domain could "
                f"not be checked.",
                Severity.WARNING,
            )
        )
    else:
        row = features.as_row()
        for column, (low, high) in domain.items():
            value = float(row[column])
            if value < low or value > high:
                out_of_domain[column] = (value, (low, high))
        if out_of_domain:
            listed = ", ".join(
                f"{name} {value:.4g} outside {low:.4g}-{high:.4g}"
                for name, (value, (low, high)) in out_of_domain.items()
            )
            caveats.append(
                Caveat(
                    "OUTSIDE_ML_TRAINING_DOMAIN",
                    f"The residual model was trained on a narrower range than this "
                    f"configuration: {listed}. Its correction would be an "
                    f"extrapolation from unrelated neighbours, so treat the "
                    f"corrected prediction as no better than the physics baseline.",
                    Severity.WARNING,
                )
            )

    return MLHandoff(
        features=features,
        physics_front_peak_us=prediction.front_time_us,
        physics_tail_us=prediction.tail_time_us,
        physics_crest_kv=prediction.crest_voltage_kv,
        requested_kv=v_to_kv(prediction.requested_voltage_v),
        out_of_domain=out_of_domain,
        caveats=caveats,
    )
