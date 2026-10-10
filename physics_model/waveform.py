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
* Its peak is not a prediction. The peak is an *input*. Callers may pass the
  preliminary loaded estimate ``N*V_stage*eta`` from
  :func:`physics_model.calculations.loaded_crest_estimate`, or the request.
  Workbook E20 is the request echoed back and is not a crest model. A
  reconstruction cannot invent a circuit solution the source lacked.

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
    Severity,
)
from physics_model.inputs import ImpulseType

__all__ = [
    "IS_WORKBOOK_OUTPUT",
    "DoubleExponential",
    "WaveformParameters",
    "fit_double_exponential",
    "extract_front_time",
    "extract_time_to_peak",
    "extract_tail_time",
    "extract_peak",
]

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
            "The peak of this curve was supplied, not predicted. The workbook has "
            "no crest-voltage model, so the reconstruction cannot estimate one.",
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
