"""The workbook's physics equations, ported to Python without alteration.

Every function below names the workbook cell it came from. Where the workbook's
formula is an approximation rather than a circuit solution, the docstring says so
and says which part is empirical. Nothing here was re-derived or "improved": if a
formula is crude, it is reproduced crudely and flagged, because the synthetic
dataset and the ML residuals were both generated from these exact expressions and
changing them would silently invalidate the ML layer.

What is genuinely physics
-------------------------
``total_load_capacitance``, ``required_stages``, ``charging_voltage_per_stage``,
``stage_utilisation``, ``generator_capacitance`` and ``tail_time`` are standard,
exact given their stated idealisations.

What is an empirical approximation
----------------------------------
``front_time`` combines a resistive term and an inductive term in quadrature with a
coefficient of 2.5. The 1.67 factor is the IEC definition of front time from the
30%/90% points; the quadrature combination and the 2.5 are a smoothing heuristic,
not a solution of the series RLC loop. It is not a simulation of the generator.

What is not modelled at all
---------------------------
``crest_voltage`` is workbook cell ``E20``, whose formula is ``=B7`` - the requested
test voltage echoed back. The workbook contains no voltage-efficiency physics, so
peak-voltage error is structurally invisible to the physics layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from backend.units import s_to_us, v_to_kv
from physics_model.inputs import (
    GeneratorConfiguration,
    ImpulseTarget,
    ImpulseType,
    TestCircuit,
    Veracity,
)

__all__ = [
    "FRONT_TIME_DEFINITION_FACTOR",
    "INDUCTANCE_COEFFICIENT",
    "TAIL_DECAY_FACTOR",
    "RESISTOR_SNAP_GRID_OHM",
    "Severity",
    "Caveat",
    "Quantity",
    "PhysicsPrediction",
    "total_load_capacitance",
    "required_stages",
    "charging_voltage_per_stage",
    "ideal_output_voltage",
    "stage_utilisation",
    "generator_capacitance",
    "series_equivalent_capacitance",
    "stored_energy",
    "front_time",
    "tail_time",
    "crest_voltage",
    "front_time_floor",
    "front_time_is_reachable",
    "theoretical_total_front_resistance",
    "theoretical_total_tail_resistance",
    "snap_to_grid",
    "predict",
]

# Workbook E18/E12: the IEC 60060-1 front-time definition factor, T1 = 1.67*(t90-t30).
FRONT_TIME_DEFINITION_FACTOR = 1.67

# Workbook E18/E12: weight on the inductive term. Empirical; no derivation was
# supplied and none is obvious from the series RLC loop.
INDUCTANCE_COEFFICIENT = 2.5

# Workbook E19/E15: ln(2), i.e. decay to half value in a first-order discharge.
TAIL_DECAY_FACTOR = 0.693

# Workbook E14/E17 round the theoretical resistance to the nearest 5 ohm.
RESISTOR_SNAP_GRID_OHM = 5.0


class Severity(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warning"
    NOTICE = "notice"


@dataclass(frozen=True)
class Caveat:
    """A machine-readable reason not to trust a number at face value."""

    code: str
    message: str
    severity: Severity = Severity.WARNING

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True)
class Quantity:
    """A number that carries its unit and where it came from."""

    value: float
    unit: str
    source: str
    veracity: Veracity = Veracity.SPECIFIED

    def __str__(self) -> str:
        return f"{self.value:.6g} {self.unit} ({self.source})"


# --------------------------------------------------------------------------- #
# Capacitance and voltage. Exact given the stated idealisations.
# --------------------------------------------------------------------------- #


def total_load_capacitance(
    load_f: float, divider_f: float, stray_f: float
) -> float:
    """Workbook ``E7``: ``SUM(B8:B10)``.

    The test object, the measuring divider and the stray capacitance all appear
    across the generator output, so they add. Exact for parallel capacitors.
    """
    return load_f + divider_f + stray_f


def required_stages(
    test_voltage_v: float, max_stage_voltage_v: float, efficiency: float
) -> int:
    """Workbook ``E8``: ``CEILING(B7/(B14*B12),1)``.

    Fewest stages whose combined rated voltage, after the efficiency derating,
    reaches the requested test voltage. Exact.
    """
    if efficiency <= 0.0:
        raise ValueError(f"efficiency must be positive, got {efficiency}")
    if max_stage_voltage_v <= 0.0:
        raise ValueError(
            f"max_stage_voltage_v must be positive, got {max_stage_voltage_v}"
        )
    return math.ceil(test_voltage_v / (max_stage_voltage_v * efficiency))


def charging_voltage_per_stage(
    test_voltage_v: float, efficiency: float, stages: int
) -> float:
    """Workbook ``E9``: ``B7/(B12*E8)``.

    The DC voltage each stage must be charged to. Exact inverse of
    :func:`ideal_output_voltage`.
    """
    if stages <= 0:
        raise ValueError(f"stages must be positive, got {stages}")
    if efficiency <= 0.0:
        raise ValueError(f"efficiency must be positive, got {efficiency}")
    return test_voltage_v / (efficiency * stages)


def ideal_output_voltage(
    stages: int, charging_voltage_per_stage_v: float, efficiency: float
) -> float:
    """Open-circuit output voltage: ``n * Vcharge * efficiency``.

    The workbook never writes this directly; it is the exact algebraic inverse of
    ``E9``, included because the optimiser needs the forward direction. Note the
    efficiency is an operator-supplied assumption (``B12``), not a derived quantity,
    so this inherits that assumption's uncertainty.
    """
    return stages * charging_voltage_per_stage_v * efficiency


def stage_utilisation(
    charging_voltage_per_stage_v: float, max_stage_voltage_v: float
) -> float:
    """Workbook ``E10``: ``E9/B14``. Fraction of each stage's rating in use."""
    if max_stage_voltage_v <= 0.0:
        raise ValueError(
            f"max_stage_voltage_v must be positive, got {max_stage_voltage_v}"
        )
    return charging_voltage_per_stage_v / max_stage_voltage_v


def generator_capacitance(stage_capacitance_f: float, stages: int) -> float:
    """Workbook ``E11``: ``B15/E8``.

    Once the spark gaps fire the stage capacitors are in series, so ``n`` equal
    capacitors of ``C`` give ``C/n``. Exact for identical capacitors.
    """
    if stages <= 0:
        raise ValueError(f"stages must be positive, got {stages}")
    return stage_capacitance_f / stages


def series_equivalent_capacitance(
    generator_c_f: float, load_c_f: float
) -> float:
    """``C1*C2/(C1+C2)``. Standard, and **not** taken from the workbook.

    Included because it justifies an approximation the workbook relies on: the true
    front time constant is ``Rf * C1*C2/(C1+C2)``, which tends to ``Rf * C2`` when
    ``C1 >> C2``. For the worked example C1 is 333 nF against C2 of 1.5 nF, a ratio
    of 222, so the workbook's use of ``Rf * C2`` in ``E18`` is sound to ~0.4%.
    """
    total = generator_c_f + load_c_f
    if total <= 0.0:
        raise ValueError("capacitances must sum to a positive value")
    return (generator_c_f * load_c_f) / total


def stored_energy(capacitance_f: float, voltage_v: float) -> float:
    """``E = 0.5 * C * V**2`` in joules.

    Standard textbook relation. The workbook does not compute energy anywhere, so
    this is an addition rather than a port.
    """
    if capacitance_f < 0.0:
        raise ValueError(f"capacitance_f must not be negative, got {capacitance_f}")
    return 0.5 * capacitance_f * voltage_v * voltage_v


# --------------------------------------------------------------------------- #
# Waveform timing. Forward direction.
# --------------------------------------------------------------------------- #


def front_time(
    front_resistance_total_ohm: float,
    total_load_c_f: float,
    stray_inductance_h: float,
) -> float:
    """Workbook ``E18``, in seconds.

    ``T1 = 1.67 * sqrt((Rf*C2)^2 + 2.5*L*C2)``

    An empirical approximation, not a circuit solution. The resistive term
    ``Rf*C2`` is the front time constant in the ``C1 >> C2`` limit; the inductive
    term is added in quadrature with a coefficient of 2.5 for which the supplied
    material gives no derivation. Treat the inductive contribution as a tuned
    correction whose coefficient needs verification against measured shots.
    """
    resistive = front_resistance_total_ohm * total_load_c_f
    inductive = INDUCTANCE_COEFFICIENT * stray_inductance_h * total_load_c_f
    return FRONT_TIME_DEFINITION_FACTOR * math.sqrt(resistive * resistive + inductive)


def tail_time(
    tail_resistance_total_ohm: float, generator_c_f: float, total_load_c_f: float
) -> float:
    """Workbook ``E19``, in seconds.

    ``T2 = 0.693 * Rt * (C1 + C2)``

    The standard first-order result: the tail resistor discharges the generator and
    load capacitances in parallel, and ``ln(2)`` of that time constant is the time
    to half value. Exact for a single-pole discharge; it ignores the front
    resistor's contribution to the discharge loop, which is small when
    ``Rf << Rt``. That inequality does **not** hold for this inventory on lightning
    impulses (50 ohm front against 25 ohm tail per stage), so this is one of the
    places where measured data is most likely to disagree with the model.
    """
    return TAIL_DECAY_FACTOR * tail_resistance_total_ohm * (generator_c_f + total_load_c_f)


def crest_voltage(test_voltage_v: float) -> float:
    """Workbook ``E20``, whose formula is literally ``=B7``.

    This is not a prediction. The workbook returns the requested voltage as the
    predicted crest, so the physics layer has zero crest-voltage error by
    construction and carries no information about what the generator would actually
    deliver. Any peak-voltage accuracy figure computed against this is meaningless.
    Callers get :data:`Severity.WARNING` caveat ``CREST_NOT_MODELLED`` from
    :func:`predict`.
    """
    return test_voltage_v


# --------------------------------------------------------------------------- #
# Waveform timing. Inverse direction: the basis of the fast optimiser.
# --------------------------------------------------------------------------- #


def front_time_floor(total_load_c_f: float, stray_inductance_h: float) -> float:
    """Shortest front time the inductive term alone permits, in seconds.

    Setting ``Rf = 0`` in ``E18`` leaves ``1.67 * sqrt(2.5*L*C2)``. No positive
    front resistance can produce a front time below this, because ``E18`` is
    strictly increasing in ``Rf``. The workbook hides this: ``E12`` wraps the
    discriminant in ``MAX(1E-18, ...)``, so an unreachable target silently yields a
    front resistance of ~0 instead of an error.
    """
    return FRONT_TIME_DEFINITION_FACTOR * math.sqrt(
        INDUCTANCE_COEFFICIENT * stray_inductance_h * total_load_c_f
    )


def front_time_is_reachable(
    target_front_time_s: float, total_load_c_f: float, stray_inductance_h: float
) -> bool:
    """Whether any non-negative front resistance can hit the target front time."""
    return target_front_time_s > front_time_floor(total_load_c_f, stray_inductance_h)


def theoretical_total_front_resistance(
    target_front_time_s: float,
    total_load_c_f: float,
    stray_inductance_h: float,
) -> float:
    """Workbook ``E12``, in ohms. The exact algebraic inverse of :func:`front_time`.

    ``Rf = sqrt((T1/1.67)^2 - 2.5*L*C2) / C2``

    The ``MAX(1E-18, ...)`` floor on the discriminant is reproduced exactly so this
    stays numerically identical to the spreadsheet and to the dataset generator.
    When the floor engages the answer is meaningless; test with
    :func:`front_time_is_reachable` first. :func:`predict` does that for you.
    """
    if total_load_c_f <= 0.0:
        raise ValueError(f"total_load_c_f must be positive, got {total_load_c_f}")
    reduced = target_front_time_s / FRONT_TIME_DEFINITION_FACTOR
    discriminant = max(
        1e-18,
        reduced * reduced
        - INDUCTANCE_COEFFICIENT * stray_inductance_h * total_load_c_f,
    )
    return math.sqrt(discriminant) / total_load_c_f


def theoretical_total_tail_resistance(
    target_tail_time_s: float, generator_c_f: float, total_load_c_f: float
) -> float:
    """Workbook ``E15``, in ohms. The exact algebraic inverse of :func:`tail_time`.

    ``Rt = T2 / (0.693 * (C1 + C2))``

    Always solvable for a positive target, since the map is linear and increasing.
    """
    denominator = TAIL_DECAY_FACTOR * (generator_c_f + total_load_c_f)
    if denominator <= 0.0:
        raise ValueError("generator and load capacitance must sum to a positive value")
    return target_tail_time_s / denominator


def snap_to_grid(ohms: float, grid: float = RESISTOR_SNAP_GRID_OHM) -> float:
    """Workbook ``E14``/``E17``: ``ROUND(x/5,0)*5``.

    A spreadsheet convenience, not engineering. It rounds to a 5 ohm grid without
    consulting the inventory sheet, so it can return a resistance the laboratory
    does not own. :mod:`physics_model.optimiser` replaces this with a search over
    combinations that actually exist; this function is kept for workbook parity.

    Uses banker's-rounding-free half-up behaviour to match Excel's ``ROUND``, which
    differs from Python's :func:`round` on exact halves.
    """
    if grid <= 0.0:
        raise ValueError(f"grid must be positive, got {grid}")
    scaled = ohms / grid
    return math.floor(scaled + 0.5) * grid


# --------------------------------------------------------------------------- #
# Structured prediction
# --------------------------------------------------------------------------- #


@dataclass
class PhysicsPrediction:
    """Everything the physics layer can say about one configuration.

    Times are seconds and voltages volts, as everywhere in this package. The
    ``*_us`` and ``*_kv`` properties exist for display and for the ML feature row,
    which follows the workbook's laboratory units.
    """

    configuration: GeneratorConfiguration
    front_time_s: float
    tail_time_s: float
    crest_voltage_v: float
    requested_voltage_v: float
    generator_capacitance_f: float
    total_load_capacitance_f: float
    stage_utilisation: float
    stored_energy_j: float
    caveats: list[Caveat] = field(default_factory=list)

    @property
    def front_time_us(self) -> float:
        return s_to_us(self.front_time_s)

    @property
    def tail_time_us(self) -> float:
        return s_to_us(self.tail_time_s)

    @property
    def crest_voltage_kv(self) -> float:
        return v_to_kv(self.crest_voltage_v)

    @property
    def blocking(self) -> list[Caveat]:
        return [c for c in self.caveats if c.severity is Severity.BLOCKING]

    def quantities(self) -> dict[str, Quantity]:
        """Named results with units and the workbook cell each came from."""
        return {
            "front_time": Quantity(
                self.front_time_us, "us", "workbook E18 (empirical approximation)"
            ),
            "tail_time": Quantity(
                self.tail_time_us, "us", "workbook E19 (first-order discharge)"
            ),
            "crest_voltage": Quantity(
                self.crest_voltage_kv,
                "kV",
                "workbook E20 (=B7, not a prediction)",
                Veracity.UNKNOWN,
            ),
            "generator_capacitance": Quantity(
                self.generator_capacitance_f * 1e6, "uF", "workbook E11"
            ),
            "total_load_capacitance": Quantity(
                self.total_load_capacitance_f * 1e12, "pF", "workbook E7"
            ),
            "stage_utilisation": Quantity(
                self.stage_utilisation, "fraction", "workbook E10"
            ),
            "stored_energy": Quantity(
                self.stored_energy_j, "J", "0.5*C*V^2, added (not in workbook)"
            ),
        }


def predict(
    configuration: GeneratorConfiguration, requested_voltage_v: float
) -> PhysicsPrediction:
    """Run the full physics layer for one configuration.

    ``requested_voltage_v`` is kept separate from the configuration because the
    workbook's crest "prediction" is just that number echoed back, and keeping them
    distinct makes the tautology visible rather than hiding it inside the result.
    """
    circuit: TestCircuit = configuration.circuit
    c_load = circuit.total_load_capacitance_f
    c_gen = configuration.generator_capacitance_f

    caveats: list[Caveat] = [
        Caveat(
            "CREST_NOT_MODELLED",
            "Predicted crest voltage is the requested voltage echoed back "
            "(workbook E20 is '=B7'). The physics layer models no voltage "
            "efficiency, so it cannot estimate peak-voltage error.",
            Severity.WARNING,
        ),
        Caveat(
            "FRONT_TIME_IS_EMPIRICAL",
            "Front time uses the workbook's quadrature approximation with an "
            "inductive coefficient of 2.5, for which no derivation was supplied. "
            "It is not a transient simulation of the generator.",
            Severity.NOTICE,
        ),
        Caveat(
            "NO_MEASURED_VALIDATION",
            "No measured waveform data has been supplied, so none of these "
            "numbers have been checked against the physical generator.",
            Severity.NOTICE,
        ),
    ]

    target: ImpulseTarget = configuration.target
    if not front_time_is_reachable(
        target.front_time_s, c_load, circuit.stray_inductance_h
    ):
        floor_us = s_to_us(front_time_floor(c_load, circuit.stray_inductance_h))
        caveats.append(
            Caveat(
                "FRONT_TIME_BELOW_INDUCTIVE_FLOOR",
                f"The {s_to_us(target.front_time_s):.3g} us target front time is "
                f"below the {floor_us:.3g} us floor set by the stray inductance "
                f"alone. No front resistance can reach it; reduce stray inductance "
                f"or load capacitance.",
                Severity.BLOCKING,
            )
        )

    if circuit.veracity is Veracity.ILLUSTRATIVE:
        caveats.append(
            Caveat(
                "CIRCUIT_VALUES_ILLUSTRATIVE",
                "The test-circuit capacitances and inductance are the workbook's "
                "single worked example, not measurements of this laboratory.",
                Severity.NOTICE,
            )
        )

    t1 = front_time(
        configuration.front_resistance_total_ohm, c_load, circuit.stray_inductance_h
    )
    t2 = tail_time(configuration.tail_resistance_total_ohm, c_gen, c_load)

    # Energy at the moment of charging: n stage capacitors each at the charging
    # voltage. Equals 0.5*C1*Vgen^2 on the discharge side; see the test suite.
    energy = configuration.stages * stored_energy(
        configuration.ratings.stage_capacitance_f,
        configuration.charging_voltage_per_stage_v,
    )

    return PhysicsPrediction(
        configuration=configuration,
        front_time_s=t1,
        tail_time_s=t2,
        crest_voltage_v=crest_voltage(requested_voltage_v),
        requested_voltage_v=requested_voltage_v,
        generator_capacitance_f=c_gen,
        total_load_capacitance_f=c_load,
        stage_utilisation=stage_utilisation(
            configuration.charging_voltage_per_stage_v,
            configuration.ratings.max_stage_voltage_v,
        ),
        stored_energy_j=energy,
        caveats=caveats,
    )
