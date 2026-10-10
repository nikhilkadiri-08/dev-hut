# Impulse Generator Setting Optimiser

Decision-support prototype for configuring a multi-stage high-voltage impulse
generator used to test 400 kV-class insulators. The engineer enters the test
requirement and the test-circuit parameters; the application is intended to recommend
feasible generator configurations for review before any physical shot is taken.

**It does not operate any equipment, and a predicted pass is not evidence that a
physical test will pass.** Every recommendation remains subject to review by a
qualified engineer under the laboratory's approved test procedure.

## Status

| Module | State |
| --- | --- |
| Input schema, validation, normalisation | implemented |
| Admin generator setup, persisted | implemented |
| Operator test-request interface | implemented |
| Assumption and provenance register | implemented |
| ML training-domain warnings | implemented |
| Physics model (`physics_model/`) | implemented |
| Configuration optimiser (`physics_model/optimiser.py`) | implemented |
| Capacitor stress and safety checks | implemented, limited by missing ratings |
| Waveform reconstruction | implemented, preliminary and labelled as such |
| ML handoff contract (`physics_model/ml_interface.py`) | implemented, no model trained |
| Residual ML model | not started |
| Wiring the physics model into `POST /api/optimise` | not started |
| Measured waveform import | not started |
| Open-literature reference catalogue | implemented |

`POST /api/optimise` exists and deliberately returns HTTP 501. The physics model is
implemented but not yet wired to it, so nobody can mistake a placeholder for a
recommendation.

## The physics model

`physics_model/` is a standalone package, independent of the ML layer and the
frontend. Every equation is a port of a specific cell in the supplied workbook, and
the docstring of each function names that cell.

| Workbook cell | Function | Nature |
| --- | --- | --- |
| `E7` | `total_load_capacitance` | physics, exact |
| `E8` | `required_stages` | physics, exact |
| `E9` | `charging_voltage_per_stage` | physics, exact |
| `E10` | `stage_utilisation` | physics, exact |
| `E11` | `generator_capacitance` | physics, exact for a series Marx |
| `E12` | `theoretical_total_front_resistance` | algebraic inverse of `E18` |
| `E15` | `theoretical_total_tail_resistance` | algebraic inverse of `E19` |
| `E14`, `E17` | `snap_to_grid` | spreadsheet convenience, superseded |
| `E18` | `front_time` | **empirical approximation** |
| `E19` | `tail_time` | physics, standard first-order discharge |
| `E20` | `crest_voltage` | **tautology, `=B7`, not a prediction** |
| `H12` | `validation.safety_verdict` | hard constraint |
| `H13` | `validation.quality_verdict` | tolerance windows, edition unconfirmed |
| `H6`-`H8` | `ml_interface.apply_residuals` | hooks only, no model trained |

Not in the workbook and added here: `stored_energy` (0.5·C·V²),
`series_equivalent_capacitance`, the waveform reconstruction, the capacitor stress
checks and the optimiser.

### Why the optimiser is fast

`E12` and `E15` are the exact algebraic inverses of `E18` and `E19`, so the required
resistance is solved in closed form rather than searched for. Because both timing
functions are strictly monotone in their resistance, the best buildable resistor is
provably one of the two values bracketing that exact solution, found by binary
search. Front time depends only on the front resistor and tail time only on the tail
resistor in this model, so the two choices are independent.

That gives `O(m log m)` once per inventory plus `O(S log m)` per request. On the
worked example it evaluates 63 candidates where exhaustive enumeration needs 4,200,
and `test_fast_search_finds_the_same_optimum_as_exhaustive_search` confirms the
answer is identical to brute force.

The independence of front and tail is a property of the *workbook's approximation*,
not of a real generator. If the physics layer is ever replaced by a transient
solver, that derivation stops holding and the search must be revisited.

### Ranking for the fewest trips up the generator

Candidates are ranked by how *centred* they sit in the acceptance window rather than
merely by whether they fall inside it, because a configuration at the window edge
passes the model and then fails the real shot, which costs a climb.

The parameters also split by cost. Stage count and resistors are physical; the
charging voltage is a control-desk setting. So `charging_voltage_trim` turns a
predicted crest error into a revised charging voltage, which needs no physical
change, and candidates that keep the currently installed resistors are credited for
it. Climb penalties break ties and never override waveform quality, which
`test_margin_is_never_sacrificed_to_avoid_a_climb_by_default` enforces.

### What the ML layer receives

`physics_model/ml_interface.py` emits a row in the exact schema of
`data/synthetic_dataset.csv`: the nine features of the workbook's `ML Helper`
columns A-I, plus the three physics baselines the residual model corrects. No
adapter is needed and no model is trained here.

```
Impulse_Type Lightning   Test_kV 1425   Load_C_pF 850   Divider_C_pF 500
Stray_C_pF 150   L_uH 18.5   Efficiency 0.82   Front_R_Stage 50   Tail_R_Stage 25
Physics_FrontPeak_us 1.21003   Physics_Tail_us 52.2089   Physics_Crest_kV 1425
```

One asymmetry matters. Two of the three residuals correct a real model. The crest
residual does not: because `E20` is `=B7`, it *is* the entire voltage-efficiency
model, learned from data, and cannot be cross-checked against physics.

### Open literature, for objects the workbook does not cover

`data/literature/open_references.json` holds cited public values: IEC 60060-1:2010
windows (lightning 1.2/50 ±30%/±20%, switching 250/2500 ±20%/±60%), the 2025
edition change list, IEC 60071-1 Um tables including the 1425 kV lightning level
for 420 kV equipment, published bushing C1 examples (370–489 pF), per-disc
insulator capacitances, and typical Marx efficiency / inductance bands.

`GET /api/literature/lookup?q=420%20kV%20bushing` matches a new description
against that file. A miss returns no number. Literature never overrides a
workbook formula or a value the operator typed. Add a new paper by appending
JSON, not by hardcoding in Python.

### Tests

93 tests in `physics_model/test_physics_model.py`. No expectation is produced by
calling the function under test. They come from the workbook's own cached cell
values, from reproducing all 2000 rows of the synthetic dataset to within 1e-12,
from mathematical properties (exact inverse round-trips, monotonicity, the `L -> 0`
limit, and stored energy agreeing across the charge and discharge routes), and from
brute force for the optimiser.

None of this is evidence of real-world accuracy. The "observed" columns in the
dataset were synthesised by perturbing the physics baseline, so the R² figures on
the workbook's 'Model Validation' sheet measure self-consistency, not accuracy, and
are not reproduced here as if they did.

## Two interfaces

The dashboard has two views, switched from the header. The split follows the
distinction the problem statement itself draws between fixed generator
parameters/constraints and variable test-circuit parameters.

**Generator setup** is for whoever maintains the equipment. It records the ratings
(stages, stage voltage, stage capacitance), the resistor store with how many units of
each value can go in series on one stage, any external resistors, and the laboratory's
default efficiency and stray inductance. It is entered once and saved to
`backend/data/generator_profile.json`. The panel beside the form shows what the setup
can reach: highest achievable peak voltage and the front and tail resistance ceilings.
A profile with blocking errors is refused rather than stored, so an operator can never
inherit a setup that exceeds the equipment's physical ratings.

**Test request** is for the engineer running a shot. It asks only what is being tested
and what result is needed: test object, impulse type, polarity, required peak voltage
and the three test-circuit capacitances. It carries no generator fields at all —
`OperatorRequestIn` has no way to express them — so the operator cannot accidentally
describe hardware the laboratory does not have. Stray inductance and efficiency are
inherited from the admin defaults, with a per-test override available behind a
checkbox.

The consequence is that the admin decides what is possible. Reduce the generator to 10
stages and a 2000 kV request is refused with "the generator tops out at 1640 kV",
where on the full 15-stage setup the same request is accepted.

There is **no authentication**. This is a workflow split so that equipment
configuration is entered once by the right person; it is not access control, and the
admin view says so on screen.

## Running it

Backend:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

Frontend (Node 22):

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to port 8000
```

Tests:

```bash
.venv/bin/python -m pytest
```

Regenerate the ML training-domain file after changing the dataset:

```bash
.venv/bin/python scripts/extract_ml_domain.py
```

## Layout

```
backend/
  config.py       Every engineering constant, each one provenance-tagged
  provenance.py   Source and verification-status types, assumption register
  units.py        The only place units are converted
  schemas.py      Request/response contracts shared by all five workstreams
  validation.py   Input checking, normalisation, out-of-domain warnings
  store.py        Persistence for the saved generator setup
  main.py         FastAPI app
  data/           ml_domain.json, generated from the training split
frontend/src/     React dashboard: admin and operator views, validation results
data/
  reference/      The supplied problem statement and workbook, unmodified
  synthetic_dataset.csv   The workbook's dataset, exported for the ML workstream
physics_model/    Workbook-ported physics, optimiser, ML handoff, literature lookup
data/literature/  Cited open-source typical values (not CPRI measurements)
scripts/          Data-extraction utilities
tests/            Validation and API tests
```

## Where the numbers come from

The repository started empty. Everything here is derived from two supplied files, now
vendored under `data/reference/`:

- `HV IG Problem Statement.pdf`
- `Hybrid_Physics_ML_Impulse_Generator_Optimiser.xlsx`

Because an unsourced number in a high-voltage tool is a liability, every constant in
`backend/config.py` carries a `Provenance` tag recording its source and one of four
statuses: `supplied`, `needs_verification`, `conflict` or `placeholder`. The full
register is served from `GET /api/metadata` and shown in the dashboard.

### Taken from the supplied material

- Generator ratings: 15 stages, 200 kV per stage, 3 µF per stage
  (workbook `Hybrid Calculator!B13:B15`).
- Test range approximately 50–2400 kV with a minimum of two active stages
  (problem statement, Additional Guidance).
- Capacity arithmetic reproduced in `validation.py`, from workbook cells `E7:E11`:
  total load capacitance is the sum of test-object, divider and stray capacitance;
  stages required is `ceil(V / (V_stage × η))`; charging voltage per stage is
  `V / (η × stages)`; generator capacitance is `C_stage / stages`.
- Resistor inventory: front 10–500 Ω, tail 5–200 Ω, four units per stage
  (workbook `Resistor Inventory`).
- Dataset: 2000 rows, split 1400 train / 300 validation / 300 hidden test, evenly
  divided between lightning and switching. Targets are three scalar residuals
  (front/peak time, tail time, crest voltage), not waveforms.

### Problems found in the supplied material

These are recorded in the assumption register and surfaced in the UI. They are not
bugs in this code; they are properties of the material we were given.

1. **No crest-voltage physics.** `Physics_Crest_kV` equals `Test_kV` exactly in all
   2000 rows, and the workbook cell is literally `=B7`. The ML "crest residual" is
   therefore synthetic scatter around a tautology, and its reported R² of 0.9994 is
   not measuring anything useful.
2. **Training domain far narrower than the operating range.** The generator is
   specified for 50–2400 kV, but the dataset only covers 1350–1500 kV for lightning
   and 950–1100 kV for switching. Outside those bands the residual correction is an
   extrapolation, which is why the out-of-domain warning is mandatory rather than
   optional.
3. **The inventory cannot produce a switching impulse.** Switching rows need
   9370–21810 Ω per stage on the front and ~1200 Ω on the tail; the stocked parts
   reach 2000 Ω and 800 Ω respectively. The problem statement mentions external
   front/tail resistors, which is the likely resolution, but supplies no values, so
   the engineer must enter them.
4. **The inventory sheet is decorative.** No workbook formula references it. The
   calculator rounds the theoretical resistance to the nearest 5 Ω, producing values
   such as 35 Ω and 55 Ω that are not stocked. The optimiser must select from real
   stock instead of rounding.
5. **The switching half-value tolerance is probably wrong.** The workbook applies
   ±20% to the time to half-value for both impulse types. For lightning that matches
   the figure usually quoted for IEC 60060-1; for switching the standard is generally
   understood to allow a much wider band, so the workbook window will reject
   configurations a real laboratory would accept.
6. **Efficiency is an input, not a result.** Nothing in the supplied material relates
   efficiency to the circuit, so the engineer's estimate propagates directly into the
   stage count and charging voltage.

### Requires verification before the compliance checker can be trusted

- The applicable edition of IEC 60060-1. Tolerance values and the definition of the
  front time have changed between editions, and no edition was supplied.
- All tolerance windows, currently taken from the workbook's acceptance gate
  (`Hybrid Calculator!H13`) rather than from the standard text.
- Test-object capacitances. The catalogue in `config.py` contains placeholders with no
  measured basis, because the supplied material gives only one worked value (850 pF).
  Only the "workbook worked example" entry is sourced.

## Conventions

The API speaks laboratory units, with the unit named in every field: `kV`, `pF`, `uF`,
`uH`, `ohm`, `us`. Conversion to SI happens only in `backend/units.py`, and the physics
engine is expected to accept SI exclusively. Mixing the two is the easiest way to get a
plausible but wrong answer out of an impulse calculation.

Validation findings come in three severities. `error` blocks the request. `warning`
means the request will run but the result is less trustworthy than it looks. `notice`
is context worth reading once. Warnings never block.

## Interfaces for the remaining workstreams

`backend/schemas.py` is the contract. `TestRequestIn` is the validated input, and
`validate_request()` returns a `ValidationResponse` whose `normalized` field carries
the target waveform, the acceptance windows and the capacity arithmetic each downstream
module needs. Build against those types rather than passing dictionaries around, and
keep each module independently testable.
