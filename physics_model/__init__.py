"""Physics model for a Marx-type high-voltage impulse generator.

Ported from the physics layer of
``data/reference/Hybrid_Physics_ML_Impulse_Generator_Optimiser.xlsx``, with every
equation traced back to the cell it came from and every approximation labelled as
one. Independent of the ML model and of the web frontend.

Units are SI throughout: volts, farads, henries, ohms, seconds, joules. Laboratory
units appear only at the boundaries, via the ``from_lab_units`` constructors in
:mod:`physics_model.inputs` and the dataset-facing row in
:mod:`physics_model.ml_interface`.

Typical use::

    from physics_model import (
        ImpulseType, OptimisationRequest, optimise,
        WORKBOOK_EXAMPLE_CIRCUIT, WORKBOOK_EXAMPLE_RATINGS,
        WORKBOOK_EXAMPLE_INVENTORY,
    )
    from backend.units import kv_to_v

    result = optimise(OptimisationRequest(
        requested_voltage_v=kv_to_v(1425.0),
        impulse_type=ImpulseType.LIGHTNING,
        circuit=WORKBOOK_EXAMPLE_CIRCUIT,
        ratings=WORKBOOK_EXAMPLE_RATINGS,
        inventory=WORKBOOK_EXAMPLE_INVENTORY,
    ))
    chosen = result.recommended or result.best_unconfirmed
    print(chosen.describe() if chosen else result.caveats)

No part of this package has been checked against measured waveforms, because no
measured data was supplied. It is decision support for a qualified engineer, and
nothing it reports is a substitute for a verification test.
"""

from physics_model.calculations import (
    Caveat,
    PhysicsPrediction,
    Quantity,
    Severity,
    charging_voltage_per_stage,
    crest_voltage,
    front_time,
    front_time_floor,
    front_time_is_reachable,
    generator_capacitance,
    ideal_marx_voltage,
    ideal_output_voltage,
    loaded_crest_estimate,
    predict,
    required_stages,
    series_equivalent_capacitance,
    snap_to_grid,
    stage_utilisation,
    stored_energy,
    tail_time,
    theoretical_total_front_resistance,
    theoretical_total_tail_resistance,
    total_load_capacitance,
    workbook_echo_crest,
)
from physics_model.targets import (
    FRONT_TIME_LIGHTNING_DEFINITION,
    FRONT_TIME_SWITCHING_DEFINITION,
    PEAK_VOLTAGE_DEFINITION,
    TAIL_TIME_DEFINITION,
    WORKBOOK_BASELINE,
    TimingDefinition,
    timing_definition,
)
from physics_model.capacitor_stress import (
    CapacitorStressReport,
    CheckStatus,
    StressCheck,
    assess_capacitor_stress,
    individual_capacitor_voltage,
)
from physics_model.inputs import (
    IMPULSE_TARGETS,
    WORKBOOK_EXAMPLE_CIRCUIT,
    WORKBOOK_EXAMPLE_INVENTORY,
    WORKBOOK_EXAMPLE_RATINGS,
    CapacitorRatings,
    GeneratorConfiguration,
    GeneratorRatings,
    ImpulseTarget,
    ImpulseType,
    IntraStageTopology,
    ResistorInventory,
    ResistorStock,
    TestCircuit,
    Veracity,
)
from physics_model.literature import (
    Citation,
    InsulationLevel,
    LiteratureHit,
    LiteratureWindow,
    TypicalRange,
    impulse_window,
    insulation_level_for_um,
    literature_impulse_target,
    load_catalogue,
    lookup,
    suggest_test_voltage_kv,
    typical_ranges,
    workbook_conflicts,
)
from physics_model.ml_interface import (
    DATASET_FEATURE_COLUMNS,
    DATASET_PHYSICS_COLUMNS,
    DATASET_RESIDUAL_COLUMNS,
    KNN_NEIGHBOURS,
    MLFeatures,
    MLHandoff,
    ResidualCorrection,
    VoltageTrim,
    apply_residuals,
    build_handoff,
    charging_voltage_trim,
    knn_distance,
    training_domain,
)
from physics_model.optimiser import (
    Candidate,
    InstalledSetup,
    OptimisationRequest,
    OptimisationResult,
    RankingWeights,
    ResistorBuild,
    SequenceOptimisation,
    ShotOptimisation,
    bracketing_values,
    buildable_resistances,
    optimise,
    optimise_sequence,
)
from physics_model.test_sequence import (
    CitationStatus,
    ConstraintStatus,
    ShotEvaluation,
    ShotSpec,
    StandardCitation,
    SequenceEvaluation,
    TestSequence,
    WaveformKind,
    evaluate_sequence,
    evaluate_shot,
    impulse_type_for,
)
from physics_model.validation import (
    QualityObjective,
    SafetyConstraint,
    ValidationReport,
    Verdict,
    achievable_per_stage_values,
    is_realisable,
    validate_configuration,
)
from physics_model.waveform import (
    ChoppedLightning,
    DoubleExponential,
    WaveformParameters,
    build_chopped_lightning,
    extract_front_time,
    extract_peak,
    extract_tail_time,
    extract_time_to_peak,
    fit_double_exponential,
)

__all__ = [
    # inputs
    "ImpulseType",
    "ImpulseTarget",
    "IMPULSE_TARGETS",
    "IntraStageTopology",
    "Veracity",
    "CapacitorRatings",
    "GeneratorRatings",
    "GeneratorConfiguration",
    "ResistorStock",
    "ResistorInventory",
    "TestCircuit",
    "WORKBOOK_EXAMPLE_CIRCUIT",
    "WORKBOOK_EXAMPLE_RATINGS",
    "WORKBOOK_EXAMPLE_INVENTORY",
    # calculations
    "Caveat",
    "Severity",
    "Quantity",
    "PhysicsPrediction",
    "total_load_capacitance",
    "required_stages",
    "charging_voltage_per_stage",
    "ideal_marx_voltage",
    "ideal_output_voltage",
    "loaded_crest_estimate",
    "workbook_echo_crest",
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
    # targets
    "FRONT_TIME_LIGHTNING_DEFINITION",
    "FRONT_TIME_SWITCHING_DEFINITION",
    "PEAK_VOLTAGE_DEFINITION",
    "TAIL_TIME_DEFINITION",
    "WORKBOOK_BASELINE",
    "TimingDefinition",
    "timing_definition",
    # waveform
    "DoubleExponential",
    "ChoppedLightning",
    "WaveformParameters",
    "fit_double_exponential",
    "extract_front_time",
    "extract_time_to_peak",
    "extract_tail_time",
    "extract_peak",
    "build_chopped_lightning",
    # stress
    "CheckStatus",
    "StressCheck",
    "CapacitorStressReport",
    "individual_capacitor_voltage",
    "assess_capacitor_stress",
    # validation
    "Verdict",
    "SafetyConstraint",
    "QualityObjective",
    "ValidationReport",
    "achievable_per_stage_values",
    "is_realisable",
    "validate_configuration",
    # optimiser
    "ResistorBuild",
    "InstalledSetup",
    "RankingWeights",
    "OptimisationRequest",
    "Candidate",
    "OptimisationResult",
    "buildable_resistances",
    "bracketing_values",
    "optimise",
    "optimise_sequence",
    "SequenceOptimisation",
    "ShotOptimisation",
    # sequences
    "WaveformKind",
    "CitationStatus",
    "ConstraintStatus",
    "StandardCitation",
    "ShotSpec",
    "TestSequence",
    "ShotEvaluation",
    "SequenceEvaluation",
    "evaluate_shot",
    "evaluate_sequence",
    "impulse_type_for",
    # ml handoff
    "DATASET_FEATURE_COLUMNS",
    "DATASET_PHYSICS_COLUMNS",
    "DATASET_RESIDUAL_COLUMNS",
    "KNN_NEIGHBOURS",
    "MLFeatures",
    "MLHandoff",
    "ResidualCorrection",
    "VoltageTrim",
    "knn_distance",
    "training_domain",
    "apply_residuals",
    "charging_voltage_trim",
    "build_handoff",
    # literature
    "Citation",
    "InsulationLevel",
    "LiteratureHit",
    "LiteratureWindow",
    "TypicalRange",
    "impulse_window",
    "insulation_level_for_um",
    "literature_impulse_target",
    "load_catalogue",
    "lookup",
    "suggest_test_voltage_kv",
    "typical_ranges",
    "workbook_conflicts",
]
