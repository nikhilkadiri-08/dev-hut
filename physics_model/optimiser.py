"""Selects the best buildable configuration, cheaply and provably.

Why this is not a search
------------------------
The obvious approach is to enumerate every (stages, front resistor, tail resistor)
triple, predict each, and keep the best. For the workbook's worked example that is
7 feasible stage counts times 25 distinct buildable front resistances times 24
tail resistances, so 4,200 evaluations per request, and it grows with the inventory.

It is also unnecessary, because the workbook's timing equations are **analytically
invertible**. Cells ``E12`` and ``E15`` are the exact algebraic inverses of ``E18``
and ``E19``. So rather than searching for a resistance that happens to produce the
right time, we solve for the resistance that produces it exactly, then take the
nearest value the laboratory can actually build.

The optimum is then provably one of two values, by this chain of reasoning:

1. ``front_time`` is strictly increasing in the front resistance and ``tail_time``
   is strictly increasing in the tail resistance. Both are continuous.
2. So over a sorted set of buildable resistances, the one minimising
   ``|achieved - target|`` is adjacent to the exact analytic solution: it is one of
   the two values bracketing it. Nothing further away can be closer.
3. The acceptance window centre equals the target for all four cases in use
   (lightning 0.84-1.56 around 1.2, switching 200-300 around 250, and both tails
   being symmetric +/-20%), so minimising ``|achieved - target|`` is the same thing
   as maximising the margin inside the window. ``test_physics_model`` asserts this
   rather than assuming it.
4. Front time depends only on the front resistance and tail time only on the tail
   resistance in this model, so the joint objective separates. Maximising a
   minimum of two independent functions is achieved by maximising each separately,
   because ``min`` is monotone in both arguments.

That makes the cost ``O(m log m)`` once per inventory plus ``O(S log m)`` per
request, with ``S`` the number of candidate stage counts and ``m`` the inventory
size: at most ``4*S`` evaluations, so 28 instead of 4,200 for the worked example,
and the answer is the true optimum rather than an approximation to it.
``test_fast_search_finds_the_same_optimum_as_exhaustive_search`` checks that
against brute force over the whole space.

The separability in step 4 is a property of the *workbook's approximation*, not of
a real impulse generator. A true double-exponential solution couples the front and
tail through both resistors, and if the physics layer is ever replaced with a
transient solver this derivation stops holding and the search must be revisited.

Minimising physical reconfiguration
-----------------------------------
The operational cost that matters is not CPU time, it is how many times somebody
has to climb the generator. That shapes the ranking in two ways.

First, candidates are ranked by how *centred* they sit in the acceptance window,
not merely by whether they fall inside it. A configuration at the window edge
passes the model and then fails the real shot under a small model error, which
costs a climb. A centred one absorbs the same error.

Second, the parameters split into ones that cost a climb and one that does not.
Stage count and resistors are physical. The charging voltage is set from the
control desk. So where a correction can be absorbed by charging voltage it should
be, and :func:`physics_model.ml_interface.charging_voltage_trim` exists for exactly
that. Candidates that keep the currently installed resistors are credited for it.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

from backend.units import v_to_kv
from physics_model.calculations import (
    Caveat,
    PhysicsPrediction,
    Severity,
    charging_voltage_per_stage,
    generator_capacitance,
    predict,
    required_stages,
    theoretical_total_front_resistance,
    theoretical_total_tail_resistance,
    total_load_capacitance,
)
from physics_model.inputs import (
    IMPULSE_TARGETS,
    GeneratorConfiguration,
    GeneratorRatings,
    ImpulseType,
    ResistorInventory,
    ResistorStock,
    TestCircuit,
)
from physics_model.validation import (
    ValidationReport,
    achievable_per_stage_values,
    validate_configuration,
)

__all__ = [
    "ResistorBuild",
    "InstalledSetup",
    "RankingWeights",
    "OptimisationRequest",
    "Candidate",
    "OptimisationResult",
    "buildable_resistances",
    "bracketing_values",
    "optimise",
]


@dataclass(frozen=True)
class ResistorBuild:
    """How to physically make one per-stage resistance from stocked parts."""

    ohms_per_stage: float
    base_ohms: float
    units_in_series: int

    def __str__(self) -> str:
        if self.units_in_series == 1:
            return f"{self.ohms_per_stage:g} ohm/stage (1 x {self.base_ohms:g} ohm)"
        return (
            f"{self.ohms_per_stage:g} ohm/stage "
            f"({self.units_in_series} x {self.base_ohms:g} ohm in series)"
        )


@dataclass(frozen=True)
class InstalledSetup:
    """What is currently fitted to the generator, if anything.

    Supplying this lets the optimiser prefer a configuration that needs no physical
    change. Per-stage resistances, matching the rest of the package.
    """

    stages: int
    front_resistance_per_stage_ohm: float
    tail_resistance_per_stage_ohm: float


@dataclass(frozen=True)
class RankingWeights:
    """How to trade waveform margin against physical reconfiguration effort.

    These are an operational preference, not physics. The defaults put margin
    first and use the climb penalties to break ties between configurations of
    similar quality, which is the behaviour that avoids pointless re-work without
    ever recommending a worse waveform to save a climb.
    """

    margin: float = 1.0
    stage_change_penalty: float = 0.10
    front_swap_penalty: float = 0.05
    tail_swap_penalty: float = 0.05
    units_penalty: float = 0.002


@dataclass(frozen=True)
class OptimisationRequest:
    """Everything needed to choose a configuration."""

    requested_voltage_v: float
    impulse_type: ImpulseType
    circuit: TestCircuit
    ratings: GeneratorRatings
    inventory: ResistorInventory
    installed: InstalledSetup | None = None
    shots_per_hour: float | None = None
    weights: RankingWeights = field(default_factory=RankingWeights)

    def __post_init__(self) -> None:
        if self.requested_voltage_v <= 0.0:
            raise ValueError(
                f"requested_voltage_v must be positive, got {self.requested_voltage_v}"
            )


@dataclass
class Candidate:
    """One ranked configuration, with its reasoning attached."""

    configuration: GeneratorConfiguration
    prediction: PhysicsPrediction
    report: ValidationReport
    front_build: ResistorBuild
    tail_build: ResistorBuild
    climb_operations: int
    climb_detail: tuple[str, ...]
    score: float

    @property
    def stages(self) -> int:
        return self.configuration.stages

    @property
    def buildable(self) -> bool:
        return self.report.buildable

    @property
    def compliant(self) -> bool:
        return self.report.quality_verdict.value == "compliant"

    @property
    def worst_margin(self) -> float:
        return self.report.worst_margin

    def describe(self) -> str:
        return (
            f"{self.stages} stages at "
            f"{v_to_kv(self.configuration.charging_voltage_per_stage_v):.1f} kV/stage, "
            f"front {self.front_build}, tail {self.tail_build} -> "
            f"front {self.prediction.front_time_us:.4g} us, "
            f"tail {self.prediction.tail_time_us:.4g} us "
            f"(margin {self.worst_margin:+.3f}, "
            f"{self.climb_operations} physical change(s))"
        )


@dataclass
class OptimisationResult:
    """Ranked candidates plus an account of what was rejected and why."""

    request: OptimisationRequest
    candidates: list[Candidate]
    rejected: list[tuple[str, str]] = field(default_factory=list)
    caveats: list[Caveat] = field(default_factory=list)
    evaluations: int = 0
    exhaustive_evaluations: int = 0

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def recommended(self) -> Candidate | None:
        """Highest-ranked candidate that is both buildable and predicted compliant."""
        for candidate in self.candidates:
            if candidate.buildable and candidate.compliant:
                return candidate
        return None


def buildable_resistances(
    stock: tuple[ResistorStock, ...],
) -> tuple[list[float], dict[float, ResistorBuild]]:
    """Every per-stage resistance the stock can make, sorted, with how to make it.

    Where two stockings give the same resistance (for example 2 x 10 ohm and
    1 x 20 ohm), the one using fewer units is kept, since it is less to fit.

    Cost ``O(m log m)``, done once per inventory.
    """
    builds: dict[float, ResistorBuild] = {}
    for item in stock:
        for units in range(1, item.units_per_stage + 1):
            value = round(item.ohms * units, 9)
            existing = builds.get(value)
            if existing is None or units < existing.units_in_series:
                builds[value] = ResistorBuild(value, item.ohms, units)
    return sorted(builds), builds


def bracketing_values(values: list[float], target: float) -> list[float]:
    """The at-most-two sorted values that bracket ``target``.

    ``O(log m)`` by binary search. These are the only candidates that can be
    closest to ``target``, which is what makes the search cheap; see the module
    docstring for why that is sufficient.
    """
    if not values:
        return []
    index = bisect_left(values, target)
    chosen: list[float] = []
    if index > 0:
        chosen.append(values[index - 1])
    if index < len(values):
        chosen.append(values[index])
    return chosen


def _climb_cost(
    candidate: GeneratorConfiguration,
    front: ResistorBuild,
    tail: ResistorBuild,
    installed: InstalledSetup | None,
    weights: RankingWeights,
) -> tuple[int, tuple[str, ...], float]:
    """Count the physical interventions this configuration needs, and penalise them.

    Changing the charging voltage is free: it is a control-desk setting. Changing
    stage count or resistors is not.
    """
    units = (front.units_in_series + tail.units_in_series) * candidate.stages
    if installed is None:
        return (
            0,
            ("No installed setup supplied, so physical change cannot be counted.",),
            weights.units_penalty * units,
        )

    operations = 0
    detail: list[str] = []
    penalty = weights.units_penalty * units

    if candidate.stages != installed.stages:
        operations += 1
        penalty += weights.stage_change_penalty
        detail.append(
            f"Change the active stage count from {installed.stages} to "
            f"{candidate.stages}."
        )
    if abs(
        candidate.front_resistance_per_stage_ohm
        - installed.front_resistance_per_stage_ohm
    ) > 1e-6:
        operations += 1
        penalty += weights.front_swap_penalty
        detail.append(
            f"Swap the front resistors from "
            f"{installed.front_resistance_per_stage_ohm:g} to "
            f"{candidate.front_resistance_per_stage_ohm:g} ohm/stage."
        )
    if abs(
        candidate.tail_resistance_per_stage_ohm
        - installed.tail_resistance_per_stage_ohm
    ) > 1e-6:
        operations += 1
        penalty += weights.tail_swap_penalty
        detail.append(
            f"Swap the tail resistors from "
            f"{installed.tail_resistance_per_stage_ohm:g} to "
            f"{candidate.tail_resistance_per_stage_ohm:g} ohm/stage."
        )

    if not detail:
        detail.append(
            "No physical change needed; the installed resistors and stage count "
            "already suit this test. Set the charging voltage from the desk."
        )
    return operations, tuple(detail), penalty


def optimise(request: OptimisationRequest) -> OptimisationResult:
    """Rank the buildable configurations for one test request.

    Returns candidates ordered best-first. A candidate that violates a hard safety
    constraint is excluded and recorded in ``rejected``; a candidate whose
    component checks could not be evaluated is kept, because unevaluated is not the
    same as unsafe, and the unevaluated checks travel with it in its report.
    """
    circuit = request.circuit
    ratings = request.ratings
    c_load = total_load_capacitance(
        circuit.load_capacitance_f,
        circuit.divider_capacitance_f,
        circuit.stray_capacitance_f,
    )
    target = IMPULSE_TARGETS[request.impulse_type]

    front_values, front_builds = buildable_resistances(request.inventory.front)
    tail_values, tail_builds = buildable_resistances(request.inventory.tail)

    candidates: list[Candidate] = []
    rejected: list[tuple[str, str]] = []
    caveats: list[Caveat] = []
    evaluations = 0

    minimum_stages = max(
        ratings.min_stages,
        required_stages(
            request.requested_voltage_v, ratings.max_stage_voltage_v, circuit.efficiency
        ),
    )
    if minimum_stages > ratings.max_stages:
        reachable_v = (
            ratings.max_stages * ratings.max_stage_voltage_v * circuit.efficiency
        )
        caveats.append(
            Caveat(
                "EXCEEDS_GENERATOR_CEILING",
                f"{v_to_kv(request.requested_voltage_v):.4g} kV needs "
                f"{minimum_stages} stages but the generator has "
                f"{ratings.max_stages}. At {circuit.efficiency:.0%} efficiency it "
                f"tops out at {v_to_kv(reachable_v):.4g} kV.",
                Severity.BLOCKING,
            )
        )
        return OptimisationResult(
            request=request,
            candidates=[],
            rejected=rejected,
            caveats=caveats,
            evaluations=0,
            exhaustive_evaluations=0,
        )

    stage_options = range(minimum_stages, ratings.max_stages + 1)

    for stages in stage_options:
        stage_v = charging_voltage_per_stage(
            request.requested_voltage_v, circuit.efficiency, stages
        )
        if stage_v > ratings.max_stage_voltage_v:
            rejected.append(
                (
                    f"{stages} stages",
                    f"would need {v_to_kv(stage_v):.4g} kV/stage, above the "
                    f"{v_to_kv(ratings.max_stage_voltage_v):.4g} kV rating",
                )
            )
            continue

        c_gen = generator_capacitance(ratings.stage_capacitance_f, stages)

        # Closed-form inverses: workbook E12 and E15. O(1).
        front_total_target = theoretical_total_front_resistance(
            target.front_time_s, c_load, circuit.stray_inductance_h
        )
        tail_total_target = theoretical_total_tail_resistance(
            target.tail_time_s, c_gen, c_load
        )
        front_stage_target = front_total_target / stages
        tail_stage_target = tail_total_target / stages

        # O(log m) each, at most two values each, plus whatever is already fitted.
        front_choices = set(bracketing_values(front_values, front_stage_target))
        tail_choices = set(bracketing_values(tail_values, tail_stage_target))
        if request.installed is not None:
            if request.installed.front_resistance_per_stage_ohm in front_builds:
                front_choices.add(request.installed.front_resistance_per_stage_ohm)
            if request.installed.tail_resistance_per_stage_ohm in tail_builds:
                tail_choices.add(request.installed.tail_resistance_per_stage_ohm)

        for front_ohms in sorted(front_choices):
            for tail_ohms in sorted(tail_choices):
                evaluations += 1
                configuration = GeneratorConfiguration(
                    stages=stages,
                    charging_voltage_per_stage_v=stage_v,
                    front_resistance_per_stage_ohm=front_ohms,
                    tail_resistance_per_stage_ohm=tail_ohms,
                    impulse_type=request.impulse_type,
                    circuit=circuit,
                    ratings=ratings,
                )
                prediction = predict(configuration, request.requested_voltage_v)
                report = validate_configuration(
                    configuration,
                    prediction,
                    inventory=request.inventory,
                    shots_per_hour=request.shots_per_hour,
                )
                if not report.buildable:
                    rejected.append(
                        (
                            f"{stages} stages, front {front_ohms:g}, "
                            f"tail {tail_ohms:g} ohm/stage",
                            "; ".join(
                                c.description for c in report.violations()
                            ),
                        )
                    )
                    continue

                operations, detail, penalty = _climb_cost(
                    configuration,
                    front_builds[front_ohms],
                    tail_builds[tail_ohms],
                    request.installed,
                    request.weights,
                )
                score = request.weights.margin * report.worst_margin - penalty
                candidates.append(
                    Candidate(
                        configuration=configuration,
                        prediction=prediction,
                        report=report,
                        front_build=front_builds[front_ohms],
                        tail_build=tail_builds[tail_ohms],
                        climb_operations=operations,
                        climb_detail=detail,
                        score=score,
                    )
                )

    # Compliant candidates first, then by score. Sorting on `compliant` keeps a
    # non-compliant but high-margin candidate from outranking a compliant one.
    candidates.sort(key=lambda c: (not c.compliant, -c.score))

    if not candidates:
        caveats.append(
            Caveat(
                "NO_BUILDABLE_CONFIGURATION",
                "No configuration satisfied the hard constraints. The rejection "
                "list records why each option failed.",
                Severity.BLOCKING,
            )
        )
    elif candidates[0].report.unevaluated():
        caveats.append(
            Caveat(
                "RANKED_WITH_UNEVALUATED_LIMITS",
                f"The recommended configuration has "
                f"{len(candidates[0].report.unevaluated())} component check(s) that "
                f"could not be evaluated because the ratings were not supplied. "
                f"Ranking reflects waveform quality only.",
                Severity.WARNING,
            )
        )

    return OptimisationResult(
        request=request,
        candidates=candidates,
        rejected=rejected,
        caveats=caveats,
        evaluations=evaluations,
        exhaustive_evaluations=len(stage_options) * len(front_values) * len(tail_values),
    )
