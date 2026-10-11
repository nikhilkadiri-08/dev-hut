"""Preliminary voltage-versus-time reconstruction. NOT a workbook output.

Does the workbook compute a waveform?
-------------------------------------
No. Every sheet was inspected. 'Hybrid Calculator' produces three scalars per
configuration - front/peak time (``E18``), tail time (``E19``) and a crest voltage
that is just the requested voltage (``E20``). 'Synthetic Dataset' stores the same
three scalars per row. There is no time axis and no sampled voltage anywhere in the
file.

So what is this module?
-----------------------
A double-exponential shape fitted so that its *extracted* front and tail times
equal the scalars the workbook's equations produce. That makes it useful for
plotting and for checking the timing definitions against each other, and it is
honest about being a reconstruction: it adds no new physics and cannot be more
accurate than the scalars it was fitted to.

Three things it is explicitly not:

* It is not a solution of the generator's circuit equations. No node analysis of
  the Marx network was performed, because the supplied material does not define
  the network in enough detail to do so.
* It is not a measured or validated CPRI waveform. No measured data was supplied.
* Its peak is not a circuit prediction. ``realize_impulse`` sets the peak to the
  loaded estimate ``N*V_stage*eta``. Workbook E20 is the request echoed back
  and is not a crest model. A reconstruction cannot invent a circuit solution
  the source lacked.

Timing definitions used here are from IEC 60060-1 and are supported by the
workbook: the 1.67 factor in ``E18`` is the 30%/90% front-time convention, which
fixes the definition this module must apply to stay consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from backend.units import s_to_us
from physics_model.calculations import (
    FRONT_TIME_DEFINITION_FACTOR,
    Caveat,
    PhysicsPrediction,
    Severity,
)
from physics_model.inputs import ImpulseType

__all__ = [
    "IS_WORKBOOK_OUTPUT",
    "DoubleExponential",
    "ChoppedLightning",
    "WaveformParameters",
    "fit_double_exponential",
    "extract_front_time",
    "extract_time_to_peak",
    "extract_tail_time",
    "extract_peak",
    "build_chopped_lightning",
    "RealizedWaveform",
    "MODEL_KIND",
    "realize_impulse",
    "realize_from_targets",
    "serialize_realized_waveform",
]

MODEL_KIND = "reduced_order_double_exponential"
_CONSISTENCY_REL = 2e-3

#: Guard against this module ever being mistaken for ported workbook content.
IS_WORKBOOK_OUTPUT = False

_FRONT_LOW_FRACTION = 0.30
_FRONT_HIGH_FRACTION = 0.90
_HALF_VALUE_FRACTION = 0.50


@dataclass(frozen=True)
class DoubleExponential:
    """``v(t) = amplitude * (exp(-t/tau_tail) - exp(-t/tau_front))``.

    The classical impulse shape. ``tau_front`` must be the smaller constant; it
    governs the rise, ``tau_tail`` the decay.
    """

    amplitude_v: float
    tau_front_s: float
    tau_tail_s: float

    def __post_init__(self) -> None:
        if self.tau_front_s <= 0.0 or self.tau_tail_s <= 0.0:
            raise ValueError(
                f"time constants must be positive, got tau_front={self.tau_front_s}, "
                f"tau_tail={self.tau_tail_s}"
            )
        if self.tau_front_s >= self.tau_tail_s:
            raise ValueError(
                f"tau_front ({self.tau_front_s} s) must be smaller than tau_tail "
                f"({self.tau_tail_s} s); otherwise the shape is not an impulse"
            )

    def voltage_at(self, t_s: float) -> float:
        if t_s <= 0.0:
            return 0.0
        return self.amplitude_v * (
            math.exp(-t_s / self.tau_tail_s) - math.exp(-t_s / self.tau_front_s)
        )

    @property
    def time_to_peak_s(self) -> float:
        """Analytic peak time: ``tau1*tau2/(tau2-tau1) * ln(tau2/tau1)``.

        From ``dv/dt = 0``. Exact for this shape.
        """
        tf, tt = self.tau_front_s, self.tau_tail_s
        return (tf * tt / (tt - tf)) * math.log(tt / tf)

    @property
    def peak_voltage_v(self) -> float:
        return self.voltage_at(self.time_to_peak_s)

    def sample(self, duration_s: float, points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Sample the shape for plotting.

        Uses a time base that is dense over the front and coarser over the tail, so
        a 1.2/50 us impulse does not need a million uniform points to render its
        rising edge. Returns ``(seconds, volts)``.
        """
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if points < 2:
            raise ValueError(f"points must be at least 2, got {points}")
        front_span = min(duration_s, 8.0 * self.time_to_peak_s)
        front_points = max(2, points // 2)
        t = np.concatenate(
            [
                np.linspace(0.0, front_span, front_points, endpoint=False),
                np.linspace(front_span, duration_s, points - front_points),
            ]
        )
        v = self.amplitude_v * (
            np.exp(-t / self.tau_tail_s) - np.exp(-t / self.tau_front_s)
        )
        return t, v


def _crossing_on_rise(shape: DoubleExponential, level_v: float) -> float:
    """Time at which the rising edge first reaches ``level_v``.

    The rise is strictly monotone on ``[0, t_peak]``, so bisection converges to
    machine precision. Exact root-finding beats sampling here: a sampled front
    would limit timing accuracy to the sample spacing.
    """
    lo, hi = 0.0, shape.time_to_peak_s
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if shape.voltage_at(mid) < level_v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _crossing_on_tail(shape: DoubleExponential, level_v: float) -> float:
    """Time at which the falling edge drops to ``level_v``.

    Monotone decreasing after the peak. The upper bracket is grown by doubling
    until the voltage is genuinely below the level.
    """
    lo = shape.time_to_peak_s
    hi = lo * 2.0
    for _ in range(200):
        if shape.voltage_at(hi) < level_v:
            break
        hi *= 2.0
    else:
        raise RuntimeError("failed to bracket the tail crossing")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if shape.voltage_at(mid) > level_v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def extract_peak(shape: DoubleExponential) -> float:
    """Peak voltage of the shape, in volts.

    Reports the peak of the reconstruction. This is *not* a prediction of the
    generator's output voltage: the peak was supplied as a fitting input because
    the workbook models no voltage efficiency.
    """
    return shape.peak_voltage_v


def extract_front_time(shape: DoubleExponential) -> float:
    """IEC 60060-1 lightning front time ``T1 = 1.67 * (t90 - t30)``, in seconds.

    The 30%/90% convention is the one the workbook's 1.67 factor implies, so this
    is the definition the scalar equations and the waveform must share.
    """
    peak = shape.peak_voltage_v
    t30 = _crossing_on_rise(shape, _FRONT_LOW_FRACTION * peak)
    t90 = _crossing_on_rise(shape, _FRONT_HIGH_FRACTION * peak)
    return FRONT_TIME_DEFINITION_FACTOR * (t90 - t30)


def extract_time_to_peak(shape: DoubleExponential) -> float:
    """Switching-impulse time to peak ``Tp``, in seconds.

    For switching impulses IEC 60060-1 measures the time to the actual peak rather
    than applying the 30%/90% construction, so switching uses this instead of
    :func:`extract_front_time`.
    """
    return shape.time_to_peak_s


def _virtual_origin(shape: DoubleExponential) -> float:
    """IEC virtual origin ``O1``: where the 30%-90% line crosses zero, in seconds.

    ``O1 = t30 - 0.5*(t90 - t30)``, from the slope of that line.
    """
    peak = shape.peak_voltage_v
    t30 = _crossing_on_rise(shape, _FRONT_LOW_FRACTION * peak)
    t90 = _crossing_on_rise(shape, _FRONT_HIGH_FRACTION * peak)
    return t30 - 0.5 * (t90 - t30)


def extract_tail_time(shape: DoubleExponential) -> float:
    """Time to half value ``T2``, measured from the virtual origin, in seconds."""
    half = _crossing_on_tail(shape, _HALF_VALUE_FRACTION * shape.peak_voltage_v)
    return half - _virtual_origin(shape)


def _extract_front_for(shape: DoubleExponential, impulse_type: ImpulseType) -> float:
    if impulse_type is ImpulseType.SWITCHING:
        return extract_time_to_peak(shape)
    return extract_front_time(shape)


@dataclass
class WaveformParameters:
    """A fitted shape plus an honest account of how well it fitted."""

    shape: DoubleExponential
    impulse_type: ImpulseType
    target_front_time_s: float
    target_tail_time_s: float
    achieved_front_time_s: float
    achieved_tail_time_s: float
    iterations: int
    converged: bool
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def front_time_error_s(self) -> float:
        return self.achieved_front_time_s - self.target_front_time_s

    @property
    def tail_time_error_s(self) -> float:
        return self.achieved_tail_time_s - self.target_tail_time_s

    def summary(self) -> str:
        return (
            f"{self.impulse_type.value} reconstruction: "
            f"front {s_to_us(self.achieved_front_time_s):.4g} us "
            f"(target {s_to_us(self.target_front_time_s):.4g}), "
            f"tail {s_to_us(self.achieved_tail_time_s):.4g} us "
            f"(target {s_to_us(self.target_tail_time_s):.4g}), "
            f"{'converged' if self.converged else 'NOT converged'} "
            f"in {self.iterations} iterations"
        )


def fit_double_exponential(
    front_time_s: float,
    tail_time_s: float,
    peak_voltage_v: float,
    impulse_type: ImpulseType = ImpulseType.LIGHTNING,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> WaveformParameters:
    """Fit a double exponential whose extracted timings match the given scalars.

    ``front_time_s`` and ``tail_time_s`` should come from
    :func:`physics_model.calculations.front_time` and
    :func:`~physics_model.calculations.tail_time`. ``peak_voltage_v`` is an input,
    not a fitted quantity, for the reason given in the module docstring.

    The two time constants are found by a scaled fixed-point iteration: the front
    time depends almost entirely on ``tau_front`` and the tail time almost entirely
    on ``tau_tail``, so correcting each by the ratio of target to achieved converges
    quickly. The near-diagonal structure is what makes this cheap; it is not an
    assumption about the generator.

    Raises:
        ValueError: if the requested times are non-positive or the front time is
            not shorter than the tail time, which no impulse shape can satisfy.
    """
    if front_time_s <= 0.0 or tail_time_s <= 0.0:
        raise ValueError(
            f"front_time_s and tail_time_s must be positive, got "
            f"{front_time_s} and {tail_time_s}"
        )
    if front_time_s >= tail_time_s:
        raise ValueError(
            f"front time ({s_to_us(front_time_s):.4g} us) must be shorter than tail "
            f"time ({s_to_us(tail_time_s):.4g} us); no double-exponential impulse "
            f"can have a front slower than its tail"
        )
    if peak_voltage_v <= 0.0:
        raise ValueError(f"peak_voltage_v must be positive, got {peak_voltage_v}")

    tau_tail = tail_time_s / math.log(2.0)
    tau_front = front_time_s / 3.0
    converged = False
    used = 0

    for iteration in range(1, max_iterations + 1):
        used = iteration
        tau_front = min(tau_front, 0.5 * tau_tail)
        probe = DoubleExponential(1.0, tau_front, tau_tail)
        achieved_front = _extract_front_for(probe, impulse_type)
        achieved_tail = extract_tail_time(probe)

        front_ratio = front_time_s / achieved_front
        tail_ratio = tail_time_s / achieved_tail
        if (
            abs(front_ratio - 1.0) < tolerance
            and abs(tail_ratio - 1.0) < tolerance
        ):
            converged = True
            break
        tau_front *= front_ratio
        tau_tail *= tail_ratio

    tau_front = min(tau_front, 0.5 * tau_tail)
    unit = DoubleExponential(1.0, tau_front, tau_tail)
    shape = DoubleExponential(
        peak_voltage_v / unit.peak_voltage_v, tau_front, tau_tail
    )

    caveats = [
        Caveat(
            "WAVEFORM_IS_RECONSTRUCTION",
            "This voltage-versus-time curve is not produced by the source "
            "workbook, which computes only scalar times. It is a double "
            "exponential fitted to those scalars, and adds no physics beyond them.",
            Severity.WARNING,
        ),
        Caveat(
            "WAVEFORM_PEAK_IS_AN_INPUT",
            "The peak of this curve was supplied, not predicted. Callers of "
            "realize_impulse pass the loaded estimate N*V_stage*eta. The "
            "workbook has no crest-voltage model (E20 is the request echoed).",
            Severity.WARNING,
        ),
    ]
    if not converged:
        caveats.append(
            Caveat(
                "WAVEFORM_FIT_NOT_CONVERGED",
                f"The shape fit did not reach the requested tolerance in "
                f"{max_iterations} iterations. Compare achieved against target "
                f"times before using this curve.",
                Severity.BLOCKING,
            )
        )

    return WaveformParameters(
        shape=shape,
        impulse_type=impulse_type,
        target_front_time_s=front_time_s,
        target_tail_time_s=tail_time_s,
        achieved_front_time_s=_extract_front_for(shape, impulse_type),
        achieved_tail_time_s=extract_tail_time(shape),
        iterations=used,
        converged=converged,
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# Chopped lightning. Preliminary. Not a circuit chop and not a CPRI measurement.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChoppedLightning:
    """A lightning-shaped front followed by an assumed rapid collapse.

    This is **not** a validated chopped-impulse model. The pre-chop segment is the
    same double-exponential reconstruction used for full waves. The collapse is an
    explicit configurable assumption (linear or exponential to zero over a stated
    duration). A smooth double exponential with an arbitrary cutoff is not treated
    as a laboratory chop.

    ``chopping_time_s`` and ``collapse_duration_s`` are required inputs. There is
    no default; the IEC clause for a standard chop is not in this repository.
    """

    pre_chop: DoubleExponential
    chopping_time_s: float
    collapse_duration_s: float
    collapse_shape: str
    caveats: tuple[Caveat, ...] = ()

    def __post_init__(self) -> None:
        if self.chopping_time_s <= 0.0:
            raise ValueError(
                f"chopping_time_s must be positive, got {self.chopping_time_s}"
            )
        if self.collapse_duration_s <= 0.0:
            raise ValueError(
                f"collapse_duration_s must be positive, got {self.collapse_duration_s}"
            )
        if self.collapse_shape not in ("linear", "exponential"):
            raise ValueError(
                f"collapse_shape must be 'linear' or 'exponential', got {self.collapse_shape!r}"
            )

    @property
    def voltage_at_chop_v(self) -> float:
        return self.pre_chop.voltage_at(self.chopping_time_s)

    @property
    def collapse_end_s(self) -> float:
        return self.chopping_time_s + self.collapse_duration_s

    def voltage_at(self, t_s: float) -> float:
        if t_s <= 0.0:
            return 0.0
        if t_s < self.chopping_time_s:
            return self.pre_chop.voltage_at(t_s)
        if t_s >= self.collapse_end_s:
            return 0.0
        start = self.voltage_at_chop_v
        frac = (t_s - self.chopping_time_s) / self.collapse_duration_s
        if self.collapse_shape == "linear":
            return start * (1.0 - frac)
        return start * math.exp(-8.0 * frac)

    def sample(self, duration_s: float, points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        if points < 2:
            raise ValueError(f"points must be at least 2, got {points}")
        t = np.linspace(0.0, duration_s, points)
        v = np.array([self.voltage_at(float(x)) for x in t])
        return t, v


def build_chopped_lightning(
    pre_chop: DoubleExponential,
    *,
    chopping_time_s: float,
    collapse_duration_s: float,
    collapse_shape: str,
) -> ChoppedLightning:
    """Attach an assumed collapse. All chop parameters must be supplied."""
    caveats = (
        Caveat(
            "CHOP_COLLAPSE_IS_ASSUMPTION",
            f"The collapse after {s_to_us(chopping_time_s):.4g} us is a "
            f"{collapse_shape} drop to zero over {s_to_us(collapse_duration_s):.4g} us. "
            "That duration and shape are configurable assumptions, not a CPRI "
            "measurement and not a solved chopping-gap circuit.",
            Severity.WARNING,
        ),
        Caveat(
            "CHOP_NOT_A_VALIDATED_IEC_WAVE",
            "No IEC edition or clause for chopped lightning is stored in this "
            "repository. The pre-chop shape is a reconstruction of workbook "
            "scalars; the collapse is not a validated standard chop.",
            Severity.WARNING,
        ),
    )
    return ChoppedLightning(
        pre_chop=pre_chop,
        chopping_time_s=chopping_time_s,
        collapse_duration_s=collapse_duration_s,
        collapse_shape=collapse_shape,
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# Sampled V(t) for the API and chart
# --------------------------------------------------------------------------- #


def _interpolated_crossing(
    time_s: np.ndarray, voltage_v: np.ndarray, level_v: float, *, rising: bool
) -> float | None:
    """Linear interpolation between the first pair of samples that brackets ``level_v``."""
    if rising:
        for i in range(1, len(voltage_v)):
            lo, hi = voltage_v[i - 1], voltage_v[i]
            if lo < level_v <= hi:
                span = hi - lo
                if span == 0.0:
                    return float(time_s[i])
                frac = (level_v - lo) / span
                return float(time_s[i - 1] + frac * (time_s[i] - time_s[i - 1]))
        return None
    peak_i = int(np.argmax(voltage_v))
    for i in range(peak_i + 1, len(voltage_v)):
        lo, hi = voltage_v[i], voltage_v[i - 1]
        if lo <= level_v < hi:
            span = hi - lo
            if span == 0.0:
                return float(time_s[i])
            frac = (hi - level_v) / span
            return float(time_s[i - 1] + frac * (time_s[i] - time_s[i - 1]))
    return None


def _downsample(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    max_points: int,
    keep_times_s: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    keep = [0, int(np.argmax(voltage_v)), len(time_s) - 1]
    for t_keep in keep_times_s:
        keep.append(int(np.argmin(np.abs(time_s - t_keep))))
    if len(time_s) <= max_points:
        return time_s, voltage_v
    index = np.unique(
        np.concatenate(
            [np.linspace(0, len(time_s) - 1, max_points).astype(int), np.array(keep, dtype=int)]
        )
    )
    return time_s[index], voltage_v[index]


def _inject_times(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    extra_times_s: list[float],
    voltage_at,
) -> tuple[np.ndarray, np.ndarray]:
    extras_t = [t for t in extra_times_s if t is not None and t >= 0.0]
    if not extras_t:
        return time_s, voltage_v
    extras_v = [float(voltage_at(float(t))) for t in extras_t]
    merged_t = np.concatenate([time_s, np.asarray(extras_t, dtype=float)])
    merged_v = np.concatenate([voltage_v, np.asarray(extras_v, dtype=float)])
    order = np.argsort(merged_t, kind="stable")
    return merged_t[order], merged_v[order]


def _rel_err(actual: float, expected: float) -> float:
    if expected == 0.0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)


@dataclass
class RealizedWaveform:
    """Sampled reduced-order V(t). Not a full-circuit Marx solution."""

    time_s: np.ndarray
    voltage_v: np.ndarray
    model_kind: str
    model_id: str
    crest_v: float
    peak_time_s: float
    front_time_s: float | None
    tail_time_s: float | None
    workbook_front_time_s: float
    workbook_tail_time_s: float
    t30_s: float | None
    t90_s: float | None
    t50_s: float | None
    virtual_origin_s: float | None
    unavailable: tuple[str, ...]
    consistency_ok: bool
    consistency: dict[str, float | bool | str]
    caveats: list[Caveat]
    chopped: bool
    impulse_type: ImpulseType


def realize_from_targets(
    front_time_s: float,
    tail_time_s: float,
    crest_v: float,
    impulse_type: ImpulseType,
    *,
    chopping_time_s: float | None = None,
    collapse_duration_s: float | None = None,
    collapse_shape: str | None = None,
    sample_points: int = 4000,
    api_points: int = 640,
    model_id: str = "double_exp_e18_e19_loaded_crest",
    peak_caveat_code: str = "WAVEFORM_PEAK_IS_LOADED_CREST",
    peak_caveat_message: str | None = None,
) -> RealizedWaveform:
    """Fit a double exponential to supplied scalars and sample V(t).

    ``crest_v`` is the intended peak. Callers pass the loaded estimate or a
    residual-corrected crest. This is not a full-circuit transient.
    """
    from types import SimpleNamespace

    prediction = SimpleNamespace(
        front_time_s=front_time_s,
        tail_time_s=tail_time_s,
        crest_voltage_v=crest_v,
        configuration=SimpleNamespace(impulse_type=impulse_type),
    )
    return realize_impulse(
        prediction,  # type: ignore[arg-type]
        chopping_time_s=chopping_time_s,
        collapse_duration_s=collapse_duration_s,
        collapse_shape=collapse_shape,
        sample_points=sample_points,
        api_points=api_points,
        model_id=model_id,
        peak_caveat_code=peak_caveat_code,
        peak_caveat_message=peak_caveat_message,
    )


def realize_impulse(
    prediction: PhysicsPrediction,
    *,
    chopping_time_s: float | None = None,
    collapse_duration_s: float | None = None,
    collapse_shape: str | None = None,
    sample_points: int = 4000,
    api_points: int = 640,
    model_id: str = "double_exp_e18_e19_loaded_crest",
    peak_caveat_code: str = "WAVEFORM_PEAK_IS_LOADED_CREST",
    peak_caveat_message: str | None = None,
) -> RealizedWaveform:
    """Fit E18/E19 scalars onto a double exponential whose peak is the loaded crest.

    Samples are drawn from that shape. Metrics are extracted from the same
    analytical crossings the samples come from, then checked against ``max(v)``.
    """
    if not math.isfinite(prediction.crest_voltage_v) or prediction.crest_voltage_v <= 0.0:
        raise ValueError("crest must be a positive finite voltage")
    if not math.isfinite(prediction.front_time_s) or not math.isfinite(prediction.tail_time_s):
        raise ValueError("front and tail times must be finite")

    impulse_type = prediction.configuration.impulse_type
    fitted = fit_double_exponential(
        prediction.front_time_s,
        prediction.tail_time_s,
        prediction.crest_voltage_v,
        impulse_type,
    )
    shape = fitted.shape
    caveats = list(fitted.caveats)
    caveats.append(
        Caveat(
            peak_caveat_code,
            peak_caveat_message
            or (
                "The peak of this sampled curve is the independently calculated "
                "loaded estimate N*V_stage*eta, not the requested test voltage and "
                "not workbook E20."
            ),
            Severity.WARNING,
        )
    )
    caveats.append(
        Caveat(
            "REDUCED_ORDER_NOT_FULL_CIRCUIT",
            "model_kind is reduced_order_double_exponential. Topology is not "
            "specified well enough for an ODE Marx model. This is not a "
            "laboratory waveform.",
            Severity.WARNING,
        )
    )
    if impulse_type is ImpulseType.SWITCHING:
        caveats.append(
            Caveat(
                "SWITCHING_E18_IS_NOT_TP",
                "Workbook E18 still applies the lightning 1.67 construction to "
                "switching scalars. Extracted switching time on this curve is Tp.",
                Severity.WARNING,
            )
        )

    chopped = None
    if chopping_time_s is not None:
        if collapse_duration_s is None or collapse_shape is None:
            raise ValueError(
                "Chopped lightning requires chopping_time_s, collapse_duration_s "
                "and collapse_shape. They are not defaulted."
            )
        chopped = build_chopped_lightning(
            shape,
            chopping_time_s=chopping_time_s,
            collapse_duration_s=collapse_duration_s,
            collapse_shape=collapse_shape,
        )
        caveats.extend(chopped.caveats)

    duration = max(prediction.tail_time_s * 3.0, shape.time_to_peak_s * 20.0, 1e-6)
    voltage_at = shape.voltage_at
    if chopped is not None:
        duration = max(duration, chopped.collapse_end_s * 1.5)
        time_s, voltage_v = chopped.sample(duration, points=sample_points)
        voltage_at = chopped.voltage_at
    else:
        time_s, voltage_v = shape.sample(duration, points=sample_points)

    t30 = _crossing_on_rise(shape, 0.30 * shape.peak_voltage_v)
    t90 = _crossing_on_rise(shape, 0.90 * shape.peak_voltage_v)
    t50 = _crossing_on_tail(shape, 0.50 * shape.peak_voltage_v)
    origin = _virtual_origin(shape)
    inject = [shape.time_to_peak_s, t30, t90, t50]
    if chopped is not None:
        inject.extend([chopping_time_s, chopped.collapse_end_s])
    time_s, voltage_v = _inject_times(time_s, voltage_v, inject, voltage_at)
    time_s, voltage_v = _downsample(
        time_s, voltage_v, api_points, keep_times_s=tuple(t for t in inject if t is not None)
    )
    sample_max = float(np.max(voltage_v))
    peak_index = int(np.argmax(voltage_v))
    peak_time_s = float(time_s[peak_index])

    unavailable: list[str] = []
    crest_v = prediction.crest_voltage_v
    front_time_s: float | None
    tail_time_s: float | None

    if chopped is not None and chopping_time_s < shape.time_to_peak_s:
        crest_v = chopped.voltage_at_chop_v
        if chopping_time_s < _crossing_on_rise(shape, 0.90 * shape.peak_voltage_v):
            unavailable.extend(["front_time", "tail_time"])
            front_time_s = None
            tail_time_s = None
        else:
            front_time_s = extract_front_time(shape)
            unavailable.append("tail_time")
            tail_time_s = None
    elif chopped is not None:
        front_time_s = _extract_front_for(shape, impulse_type)
        tail_half = _crossing_on_tail(shape, 0.50 * shape.peak_voltage_v)
        if chopping_time_s < tail_half:
            unavailable.append("tail_time")
            tail_time_s = None
        else:
            tail_time_s = extract_tail_time(shape)
        crest_v = shape.peak_voltage_v
    else:
        front_time_s = _extract_front_for(shape, impulse_type)
        tail_time_s = extract_tail_time(shape)
        t30 = _crossing_on_rise(shape, 0.30 * shape.peak_voltage_v)
        t90 = _crossing_on_rise(shape, 0.90 * shape.peak_voltage_v)
        t50 = _crossing_on_tail(shape, 0.50 * shape.peak_voltage_v)
        origin = _virtual_origin(shape)

    sample_front = None
    sample_tail = None
    if "front_time" not in unavailable:
        if impulse_type is ImpulseType.SWITCHING and chopped is None:
            sample_front = peak_time_s
        else:
            s30 = _interpolated_crossing(time_s, voltage_v, 0.30 * crest_v, rising=True)
            s90 = _interpolated_crossing(time_s, voltage_v, 0.90 * crest_v, rising=True)
            if s30 is None or s90 is None:
                unavailable.append("front_time")
                front_time_s = None
            else:
                sample_front = FRONT_TIME_DEFINITION_FACTOR * (s90 - s30)
                t30 = t30 if t30 is not None else s30
                t90 = t90 if t90 is not None else s90
    if "tail_time" not in unavailable:
        s50 = _interpolated_crossing(time_s, voltage_v, 0.50 * crest_v, rising=False)
        if s50 is None:
            unavailable.append("tail_time")
            tail_time_s = None
        else:
            origin_s = origin if origin is not None else (
                (t30 - 0.5 * (t90 - t30)) if t30 is not None and t90 is not None else 0.0
            )
            sample_tail = s50 - origin_s
            t50 = s50

    checks = {
        "sample_max_vs_crest": _rel_err(sample_max, crest_v),
        "extracted_front_vs_workbook": (
            _rel_err(front_time_s, prediction.front_time_s)
            if front_time_s is not None and chopped is None
            else 0.0
        ),
        "extracted_tail_vs_workbook": (
            _rel_err(tail_time_s, prediction.tail_time_s)
            if tail_time_s is not None and chopped is None
            else 0.0
        ),
        "sample_front_vs_extracted": (
            _rel_err(sample_front, front_time_s)
            if sample_front is not None and front_time_s is not None
            else 0.0
        ),
        "sample_tail_vs_extracted": (
            _rel_err(sample_tail, tail_time_s)
            if sample_tail is not None and tail_time_s is not None
            else 0.0
        ),
    }
    consistency_ok = all(float(v) <= _CONSISTENCY_REL for v in checks.values()) and fitted.converged
    if not consistency_ok:
        caveats.append(
            Caveat(
                "WAVEFORM_INCONSISTENT",
                "Sampled crest or extracted times disagree with the reported "
                f"values beyond {_CONSISTENCY_REL:.2%} relative. The curve is "
                "not returned as a consistent prediction.",
                Severity.BLOCKING,
            )
        )

    return RealizedWaveform(
        time_s=time_s,
        voltage_v=voltage_v,
        model_kind=MODEL_KIND,
        model_id=model_id,
        crest_v=crest_v,
        peak_time_s=peak_time_s if chopped is None else (
            shape.time_to_peak_s if chopping_time_s > shape.time_to_peak_s else chopping_time_s
        ),
        front_time_s=front_time_s,
        tail_time_s=tail_time_s,
        workbook_front_time_s=prediction.front_time_s,
        workbook_tail_time_s=prediction.tail_time_s,
        t30_s=t30,
        t90_s=t90,
        t50_s=t50,
        virtual_origin_s=origin,
        unavailable=tuple(dict.fromkeys(unavailable)),
        consistency_ok=consistency_ok,
        consistency={**checks, "ok": consistency_ok, "fit_converged": fitted.converged},
        caveats=caveats,
        chopped=chopped is not None,
        impulse_type=impulse_type,
    )


def serialize_realized_waveform(realized: RealizedWaveform) -> dict:
    from backend.units import v_to_kv

    return {
        "model_kind": realized.model_kind,
        "model_id": realized.model_id,
        "time_s": [float(x) for x in realized.time_s],
        "voltage_v": [float(x) for x in realized.voltage_v],
        "time_us": [s_to_us(float(x)) for x in realized.time_s],
        "voltage_kv": [v_to_kv(float(x)) for x in realized.voltage_v],
        "resolution": {
            "points": int(len(realized.time_s)),
            "duration_s": float(realized.time_s[-1]) if len(realized.time_s) else 0.0,
        },
        "units": {"time": "s", "voltage": "V"},
        "assumptions": [
            "Peak is N*V_stage*eta, not the requested voltage.",
            "Front/tail fit targets are workbook E18/E19 approximations.",
            "Not a full Marx transient and not a measured CPRI waveform.",
        ],
        "metrics": {
            "crest_v": realized.crest_v,
            "crest_kv": v_to_kv(realized.crest_v),
            "peak_time_s": realized.peak_time_s,
            "peak_time_us": s_to_us(realized.peak_time_s),
            "front_time_s": realized.front_time_s,
            "front_time_us": None if realized.front_time_s is None else s_to_us(realized.front_time_s),
            "tail_time_s": realized.tail_time_s,
            "tail_time_us": None if realized.tail_time_s is None else s_to_us(realized.tail_time_s),
            "unavailable": list(realized.unavailable),
        },
        "thresholds": {
            "thirty_percent_v": 0.30 * realized.crest_v,
            "ninety_percent_v": 0.90 * realized.crest_v,
            "fifty_percent_v": 0.50 * realized.crest_v,
            "label": "Fractions of the predicted loaded crest, not of the request.",
        },
        "consistency": realized.consistency,
        "chopped": realized.chopped,
        "disclaimer": "reduced-order approximation, not a laboratory waveform",
        "warnings": [
            {"code": item.code, "message": item.message, "severity": item.severity.value}
            for item in realized.caveats
        ],
    }
