import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, runOptimisation, validateTestRequest } from "../api";
import {
  buildOperatorPayload,
  initialOperatorForm,
  type OperatorForm,
} from "../formState";
import { Field, NumberField, TextField } from "./Field";
import { IssueList } from "./IssueList";
import {
  AssumptionsPanel,
  CapacitySummary,
  DomainPanel,
  TargetSummary,
} from "./ResultsPanel";
import type { Issue, Metadata, ProfileResponse, ValidationResponse } from "../types";

const VALIDATE_DEBOUNCE_MS = 400;

interface OperatorViewProps {
  metadata: Metadata;
  profile: ProfileResponse;
  onOpenAdmin: () => void;
}

export function OperatorView({ metadata, profile, onOpenAdmin }: OperatorViewProps) {
  const [form, setForm] = useState<OperatorForm>(() =>
    initialOperatorForm(metadata, profile.stored),
  );
  const [result, setResult] = useState<ValidationResponse | null>(null);
  const [validating, setValidating] = useState(false);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [optimiseMessage, setOptimiseMessage] = useState<string | null>(null);

  const payload = useMemo(() => buildOperatorPayload(form), [form]);
  const payloadKey = useMemo(() => JSON.stringify(payload), [payload]);
  const latestPayload = useRef(payload);
  latestPayload.current = payload;

  // Revalidate as the engineer types, but only after they pause. Also reruns when the
  // admin profile changes, since that can make a request feasible or infeasible.
  const profileStamp = profile.stored.updated_at ?? "default";
  useEffect(() => {
    setValidating(true);
    const timer = setTimeout(() => {
      validateTestRequest(latestPayload.current)
        .then((response) => {
          setResult(response);
          setTransportError(null);
        })
        .catch((error: unknown) =>
          setTransportError(
            error instanceof ApiError ? error.message : "Validation failed.",
          ),
        )
        .finally(() => setValidating(false));
    }, VALIDATE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [payloadKey, profileStamp]);

  const update = useCallback(
    <K extends keyof OperatorForm>(key: K) =>
      (value: OperatorForm[K]) =>
        setForm((current) => ({ ...current, [key]: value })),
    [],
  );

  const selectTestObject = useCallback(
    (id: string) => {
      setForm((current) => {
        const chosen = metadata.test_objects.find((item) => item.id === id);
        if (!chosen || chosen.id === "custom") {
          return { ...current, testObjectTypeId: id };
        }
        return {
          ...current,
          testObjectTypeId: id,
          loadCapacitancePf: String(chosen.default_load_capacitance_pf),
        };
      });
    },
    [metadata],
  );

  const onOptimise = useCallback(async () => {
    setOptimiseMessage(null);
    try {
      const outcome = await runOptimisation(latestPayload.current);
      setOptimiseMessage(outcome.detail);
    } catch {
      setOptimiseMessage("Could not reach the optimisation endpoint.");
    }
  }, []);

  const issues: Issue[] = result
    ? [...result.errors, ...result.warnings, ...result.notices]
    : [];
  const selectedObject = metadata.test_objects.find(
    (item) => item.id === form.testObjectTypeId,
  );
  const lab = profile.stored.profile;

  return (
    <div className="layout">
      <form className="panel" onSubmit={(event) => event.preventDefault()}>
        <section className="card card--quiet">
          <h3 className="card__title">
            Generator in use
            {profile.stored.is_factory_default ? (
              <span className="pill pill--warning">unconfirmed</span>
            ) : null}
          </h3>
          <p className="card__body">
            {lab.max_stages} stages at {lab.max_stage_voltage_kv} kV,{" "}
            {lab.stage_capacitance_uf} &micro;F per stage,{" "}
            {lab.front_resistors.length} front and {lab.tail_resistors.length} tail
            resistor values in store.
          </p>
          <button type="button" className="button button--ghost" onClick={onOpenAdmin}>
            Change in generator setup
          </button>
        </section>

        <fieldset className="group">
          <legend className="group__legend">What are you testing?</legend>
          <Field label="Test object">
            <select
              className="input"
              value={form.testObjectTypeId}
              onChange={(event) => selectTestObject(event.target.value)}
            >
              {metadata.test_objects.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </Field>
          {selectedObject ? (
            <p className="group__note">
              {selectedObject.description}
              {selectedObject.provenance.status === "placeholder" ? (
                <em className="tag tag--placeholder">placeholder capacitance</em>
              ) : null}
            </p>
          ) : null}
          <TextField
            label="Reference or serial number"
            value={form.testObjectLabel}
            onChange={update("testObjectLabel")}
            placeholder="optional, for traceability"
            path="test_object_label"
            issues={issues}
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">What result do you need?</legend>
          <Field label="Standard" path="standard_id" issues={issues}>
            <select
              className="input"
              value={form.standardId}
              onChange={(event) => update("standardId")(event.target.value)}
            >
              <option value={metadata.standard.id}>{metadata.standard.label}</option>
            </select>
          </Field>
          <p className="group__note">{metadata.standard.edition_note}</p>

          <Field label="Impulse type" path="impulse_type" issues={issues}>
            <div className="segmented">
              {metadata.impulse_types.map((spec) => (
                <button
                  key={spec.impulse_type}
                  type="button"
                  className={`segment${
                    form.impulseType === spec.impulse_type ? " segment--active" : ""
                  }`}
                  onClick={() => update("impulseType")(spec.impulse_type)}
                >
                  {spec.label}
                  <small>{spec.designation}</small>
                </button>
              ))}
            </div>
          </Field>

          <Field label="Polarity" path="polarity" issues={issues}>
            <div className="segmented">
              {(["positive", "negative"] as const).map((polarity) => (
                <button
                  key={polarity}
                  type="button"
                  className={`segment${
                    form.polarity === polarity ? " segment--active" : ""
                  }`}
                  onClick={() => update("polarity")(polarity)}
                >
                  {polarity}
                </button>
              ))}
            </div>
          </Field>

          <NumberField
            label="Required peak voltage"
            unit="kV"
            value={form.requiredPeakKv}
            onChange={update("requiredPeakKv")}
            step="5"
            path="required_peak_kv"
            issues={issues}
            hint={`This generator reaches about ${(
              lab.max_stages *
              lab.max_stage_voltage_kv *
              lab.default_efficiency
            ).toFixed(0)} kV`}
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Test circuit</legend>
          <p className="group__note">
            These change with every setup, which is why they are entered per test
            rather than in the generator setup.
          </p>
          <NumberField
            label="Test object capacitance"
            unit="pF"
            value={form.loadCapacitancePf}
            onChange={update("loadCapacitancePf")}
            path="load_capacitance_pf"
            issues={issues}
          />
          <NumberField
            label="Measuring divider capacitance"
            unit="pF"
            value={form.dividerCapacitancePf}
            onChange={update("dividerCapacitancePf")}
            path="divider_capacitance_pf"
            issues={issues}
          />
          <NumberField
            label="Stray capacitance"
            unit="pF"
            value={form.strayCapacitancePf}
            onChange={update("strayCapacitancePf")}
            path="stray_capacitance_pf"
            issues={issues}
            hint="Leads, layout and earth return path"
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Laboratory defaults</legend>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={form.overrideLabDefaults}
              onChange={(event) =>
                update("overrideLabDefaults")(event.target.checked)
              }
            />
            <span>Override for this test</span>
          </label>
          {form.overrideLabDefaults ? (
            <div className="grid-2">
              <NumberField
                label="Stray inductance"
                unit="&micro;H"
                value={form.strayInductanceUh}
                onChange={update("strayInductanceUh")}
                step="0.5"
                path="stray_inductance_uh"
                issues={issues}
              />
              <NumberField
                label="Expected efficiency"
                unit="fraction"
                value={form.expectedEfficiency}
                onChange={update("expectedEfficiency")}
                step="0.01"
                path="expected_efficiency"
                issues={issues}
              />
            </div>
          ) : (
            <p className="group__note">
              Using {lab.default_stray_inductance_uh} &micro;H stray inductance and{" "}
              {lab.default_efficiency} efficiency from the generator setup.
            </p>
          )}
        </fieldset>

        <div className="actions">
          <button
            type="button"
            className="button button--primary"
            onClick={onOptimise}
            disabled={!result?.ok}
          >
            Run optimisation
          </button>
          <p className="actions__note">
            {result?.ok
              ? "Inputs are accepted. The optimisation pipeline is still being built."
              : "Fix the blocking problems before running the optimisation."}
          </p>
        </div>
        {optimiseMessage ? (
          <section className="card card--pending">
            <h3 className="card__title">Not implemented yet</h3>
            <p className="card__body">{optimiseMessage}</p>
          </section>
        ) : null}
      </form>

      <div className="panel panel--results">
        {transportError ? (
          <section className="card card--error">
            <p>{transportError}</p>
          </section>
        ) : null}
        {result ? (
          <>
            <IssueList severity="error" issues={result.errors} />
            <IssueList severity="warning" issues={result.warnings} />
            {result.normalized ? (
              <>
                <TargetSummary normalized={result.normalized} />
                <CapacitySummary normalized={result.normalized} />
              </>
            ) : null}
            <DomainPanel checks={result.domain_checks} />
            <IssueList severity="notice" issues={result.notices} />
            <details className="disclosure disclosure--block">
              <summary>
                Assumptions and provenance ({result.relevant_assumptions.length})
              </summary>
              <AssumptionsPanel entries={result.relevant_assumptions} />
            </details>
            <details className="disclosure disclosure--block">
              <summary>Implementation status</summary>
              <ul className="capabilities">
                {Object.entries(metadata.capabilities).map(([name, status]) => (
                  <li key={name} className={`capability capability--${status}`}>
                    <span>{name.replace(/_/g, " ")}</span>
                    <em>{status.replace(/_/g, " ")}</em>
                  </li>
                ))}
              </ul>
            </details>
          </>
        ) : (
          <p className="loading">{validating ? "Validating\u2026" : "No result yet."}</p>
        )}
      </div>
    </div>
  );
}
