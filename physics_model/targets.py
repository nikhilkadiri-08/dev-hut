"""What this physics package is trying to predict.

Intended outputs
----------------
For one generator configuration the package is meant to predict three *circuit*
quantities, not to echo the request or the spreadsheet cells:

* **peak voltage** — the crest the output node would actually reach, in volts;
* **front time** — lightning: IEC front time ``T1 = 1.67 * (t90 - t30)``;
  switching: time-to-peak ``Tp``, which is a different definition;
* **tail time** — time to half-value ``T2``.

Those are the quantities a residual ML model (when one exists) should correct, and
they are the quantities an optimiser should rank. They are **not** yet available
from a solved circuit, because the supplied material does not define a complete
Marx topology.

Workbook baseline (what the equations currently compute)
--------------------------------------------------------
Until a topology is confirmed, the implemented numbers are the workbook ports:

* ``E18`` — empirical front/peak-time approximation. For lightning it is shaped
  like ``T1``. For switching the workbook still applies the same 1.67 formula,
  which is **not** IEC time-to-peak.
* ``E19`` — first-order tail-time approximation ``0.693 * Rt * (C1 + C2)``.
* ``E20`` — ``=B7``, the requested voltage echoed back. **Not** an independent
  crest prediction. Kept only so the synthetic dataset's ``Physics_Crest_kV``
  column can be reproduced.

A preliminary loaded-voltage estimate ``N * V_stage * eta`` is computed from the
configuration and is independent of the request. It is not a measured crest and
not a complete Marx simulation. The ideal open-circuit voltage ``N * V_stage``
is reported beside it so the efficiency factor is visible rather than baked in.
"""

from __future__ import annotations

from dataclasses import dataclass

from physics_model.inputs import ImpulseType

__all__ = [
    "FRONT_TIME_LIGHTNING_DEFINITION",
    "FRONT_TIME_SWITCHING_DEFINITION",
    "TAIL_TIME_DEFINITION",
    "PEAK_VOLTAGE_DEFINITION",
    "WORKBOOK_BASELINE",
    "TimingDefinition",
    "timing_definition",
]

FRONT_TIME_LIGHTNING_DEFINITION = (
    "IEC front time T1 = 1.67 * (t90 - t30), measured on the rising edge of V(t)."
)
FRONT_TIME_SWITCHING_DEFINITION = (
    "IEC switching time-to-peak Tp: interval from the origin to the actual crest. "
    "This is not the 30/90 construction and is not what workbook E18 computes."
)
TAIL_TIME_DEFINITION = (
    "Time to half-value T2, from the virtual origin to the 50% point on the tail."
)
PEAK_VOLTAGE_DEFINITION = (
    "Crest of the output voltage. Intended to come from the circuit. Workbook E20 "
    "is the requested voltage echoed back and is not this quantity."
)


@dataclass(frozen=True)
class TimingDefinition:
    """Which standardised timing a prediction is supposed to represent."""

    impulse_type: ImpulseType
    front_or_peak: str
    tail: str
    workbook_front_cell: str = "E18"
    workbook_tail_cell: str = "E19"
    workbook_matches_definition: bool = False


def timing_definition(impulse_type: ImpulseType) -> TimingDefinition:
    """The intended IEC definition versus what the workbook formula actually is."""
    if impulse_type is ImpulseType.SWITCHING:
        return TimingDefinition(
            impulse_type=impulse_type,
            front_or_peak=FRONT_TIME_SWITCHING_DEFINITION,
            tail=TAIL_TIME_DEFINITION,
            workbook_matches_definition=False,
        )
    return TimingDefinition(
        impulse_type=impulse_type,
        front_or_peak=FRONT_TIME_LIGHTNING_DEFINITION,
        tail=TAIL_TIME_DEFINITION,
        # E18 uses the 1.67 factor, so it is *shaped* like T1, but it is not
        # extracted from a V(t). The inversion is exact for E18, not for IEC T1.
        workbook_matches_definition=False,
    )


WORKBOOK_BASELINE = {
    "front_time": "E18 empirical approximation; not extracted from a simulated V(t)",
    "tail_time": "E19 first-order 0.693*Rt*(C1+C2); not a full-circuit T2",
    "peak_voltage": "E20 is =B7 and is not a prediction; use loaded_crest_estimate",
}
