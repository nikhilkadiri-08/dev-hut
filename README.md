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
| Physics engine (`backend/physics.py`) | not started |
| Residual ML model (`backend/ml_model.py`) | not started |
| Candidate optimiser (`backend/optimizer.py`) | not started |
| Compliance checker (`backend/compliance.py`) | not started |
| Robustness analysis (`backend/robustness.py`) | not started |
| Measured waveform import | not started |

`POST /api/optimise` exists and deliberately returns HTTP 501 so the frontend can
integrate against the real endpoint without anyone mistaking a placeholder for a
recommendation.

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
