# Requirements to implementation

Honest status against what this repository actually contains. Unit tests check
internal consistency. They are not laboratory validation.

| Feature | Status | Notes |
| --- | --- | --- |
| Workbook scalars E7–E11, E12/E15, E18/E19 | implemented+tested | Approximations labelled. Not replaced. |
| Loaded crest `N*V_stage*η` vs request vs E20 | implemented+tested | Distinct quantities. |
| Confirmed-safe gate | implemented+tested | Unknown ratings stay unknown. |
| Lightning T1 = 1.67*(t90−t30) extractor | implemented+tested | Bisection on the reconstruction, not a measured V(t). |
| Switching Tp extractor | implemented+tested | Waveform uses Tp. E18 still uses 1.67; conflict documented. |
| Tail T2 from virtual origin | implemented+tested | Reconstruction only. |
| Configurable test sequence | implemented+tested | User-configured. IEC clause unresolved unless supplied. |
| Chopped lightning | partial | Pre-chop reconstruction + assumed linear/exponential collapse. Not a circuit chop. |
| Voltage level ≠ tail 50% crossing | implemented+tested | Different quantities. |
| Capacitor / resistor stress | partial | Evaluable only when ratings are supplied. |
| Intra-stage topology | partial | Unspecified → per-capacitor checks UNKNOWN. |
| Modelling C energy (3 µF → 60 kJ/stage at 200 kV) | implemented+tested | Conflict with an unsourced 2.5 kJ / 0.125 µF figure is recorded, not resolved. |
| Residual ML | missing | Hook only. kNN documents the spreadsheet. No weights. |
| Measured waveforms | missing | No CPRI data in the repo. |
| IEC edition / chopped clauses | blocked | No standard PDF in the repository. |
| Marx topology / transient solve | blocked | Topology not specified. |
| Hardware control | missing (intentional) | Decision support only. |
| `POST /api/optimise` physics-only | implemented+tested | Labelled `physics_only` when no model is loaded. |
| Sequence on predict/optimise | implemented+tested | Optional body field. Single-config body still works. |

The problem-statement PDF and the source workbook are referenced by name; they are
not present under `data/reference/` in this checkout.
