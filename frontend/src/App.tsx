import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, fetchMetadata, runOptimisation, validateRequest } from "./api";
import { Field, NumberField, TextField } from "./components/Field";
import { InventoryPicker } from "./components/InventoryPicker";
import { IssueList } from "./components/IssueList";
import {
  AssumptionsPanel,
  CapacitySummary,
  DomainPanel,
  TargetSummary,
} from "./components/ResultsPanel";
import { buildPayload, initialForm, type FormState } from "./formState";
import type { Issue, Metadata, ValidationResponse } from "./types";

const VALIDATE_DEBOUNCE_MS = 400;

export function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [result, setResult] = useState<ValidationResponse | null>(null);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  const [optimiseMessage, setOptimiseMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchMetadata()
      .then((loaded) => {
        setMetadata(loaded);
        setForm(initialForm(loaded));
      })
      .catch((error: unknown) =>
        setTransportError(
          error instanceof ApiError ? error.message : "Could not load configuration.",
        ),
      );
  }, []);

  const payload = useMemo(() => (form ? buildPayload(form) : null), [form]);
  const payloadKey = useMemo(() => JSON.stringify(payload), [payload]);

  // Revalidate as the engineer types, but only after they pause.
  const latestPayload = useRef(payload);
  latestPayload.current = payload;

  useEffect(() => {
    if (!latestPayload.current) return;
    setValidating(true);
    const timer = setTimeout(() => {
      validateRequest(latestPayload.current)
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
  }, [payloadKey]);

  const update = useCallback(
    <K extends keyof FormState>(key: K) =>
      (value: FormState[K]) =>
        setForm((current) => (current ? { ...current, [key]: value } : current)),
    [],
  );

  const selectTestObject = useCallback(
    (id: string) => {
      setForm((current) => {
        if (!current || !metadata) return current;
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
    if (!payload) return;
    setOptimiseMessage(null);
    try {
      const outcome = await runOptimisation(payload);
      setOptimiseMessage(outcome.detail);
    } catch {
      setOptimiseMessage("Could not reach the optimisation endpoint.");
    }
  }, [payload]);

  if (transportError && !metadata) {
    return (
      <main className="shell shell--centred">
        <section className="card card--error">
          <h2 className="card__title">Backend unavailable</h2>
          <p>{transportError}</p>
        </section>
      </main>
    );
  }

  if (!metadata || !form) {
    return (
      <main className="shell shell--centred">
        <p className="loading">Loading generator configuration&hellip;</p>
      </main>
    );
  }

  const allIssues: Issue[] = result
    ? [...result.errors, ...result.warnings, ...result.notices]
    : [];
  const selectedObject = metadata.test_objects.find(
    (item) => item.id === form.testObjectTypeId,
  );
  const impulseSpec = metadata.impulse_types.find(
    (item) => item.impulse_type === form.impulseType,
  );

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1 className="masthead__title">Impulse Generator Setting Optimiser</h1>
          <p className="masthead__subtitle">
            Decision support for high-voltage impulse testing. Recommendations are for
            engineer review and never operate equipment.
          </p>
        </div>
        <div className="masthead__status">
          {validating ? (
            <span className="pill">checking&hellip;</span>
          ) : result?.ok ? (
            <span className="pill pill--ok">inputs accepted</span>
          ) : (
            <span className="pill pill--error">inputs incomplete</span>
          )}
        </div>
      </header>

      {transportError ? (
        <section className="card card--error">
          <p>{transportError}</p>
        </section>
      ) : null}

      <div className="layout">
        <form className="panel" onSubmit={(event) => event.preventDefault()}>
          <fieldset className="group">
            <legend className="group__legend">Test object</legend>
            <Field label="Type">
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
              issues={allIssues}
            />
          </fieldset>

          <fieldset className="group">
            <legend className="group__legend">Test requirement</legend>
            <Field label="Standard" path="standard_id" issues={allIssues}>
              <select
                className="input"
                value={form.standardId}
                onChange={(event) => update("standardId")(event.target.value)}
              >
                <option value={metadata.standard.id}>{metadata.standard.label}</option>
              </select>
            </Field>
            <p className="group__note">{metadata.standard.edition_note}</p>

            <Field label="Impulse type" path="impulse_type" issues={allIssues}>
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

            <Field label="Polarity" path="polarity" issues={allIssues}>
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
              issues={allIssues}
              hint={`Generator range ${metadata.generator.min_test_voltage_kv}\u2013${metadata.generator.max_test_voltage_kv} kV`}
            />
          </fieldset>

          <fieldset className="group">
            <legend className="group__legend">Test circuit</legend>
            <p className="group__note">
              These change with every setup, unlike the generator ratings below.
            </p>
            <NumberField
              label="Test object capacitance"
              unit="pF"
              value={form.loadCapacitancePf}
              onChange={update("loadCapacitancePf")}
              path="circuit.load_capacitance_pf"
              issues={allIssues}
            />
            <NumberField
              label="Measuring divider capacitance"
              unit="pF"
              value={form.dividerCapacitancePf}
              onChange={update("dividerCapacitancePf")}
              path="circuit.divider_capacitance_pf"
              issues={allIssues}
            />
            <NumberField
              label="Stray capacitance"
              unit="pF"
              value={form.strayCapacitancePf}
              onChange={update("strayCapacitancePf")}
              path="circuit.stray_capacitance_pf"
              issues={allIssues}
              hint="Leads, layout and earth return path"
            />
            <NumberField
              label="Stray inductance"
              unit="&micro;H"
              value={form.strayInductanceUh}
              onChange={update("strayInductanceUh")}
              step="0.5"
              path="circuit.stray_inductance_uh"
              issues={allIssues}
            />
            <NumberField
              label="Expected efficiency"
              unit="fraction"
              value={form.expectedEfficiency}
              onChange={update("expectedEfficiency")}
              step="0.01"
              path="circuit.expected_efficiency"
              issues={allIssues}
              hint="Engineer's estimate. Nothing in the supplied material calculates this."
            />
          </fieldset>

          <fieldset className="group">
            <legend className="group__legend">Generator configuration</legend>
            <div className="grid-2">
              <NumberField
                label="Stages available"
                value={form.maxStages}
                onChange={update("maxStages")}
                path="generator.max_stages"
                issues={allIssues}
              />
              <NumberField
                label="Minimum active stages"
                value={form.minActiveStages}
                onChange={update("minActiveStages")}
                path="generator.min_active_stages"
                issues={allIssues}
              />
              <NumberField
                label="Stage voltage"
                unit="kV"
                value={form.maxStageVoltageKv}
                onChange={update("maxStageVoltageKv")}
                path="generator.max_stage_voltage_kv"
                issues={allIssues}
              />
              <NumberField
                label="Stage capacitance"
                unit="&micro;F"
                value={form.stageCapacitanceUf}
                onChange={update("stageCapacitanceUf")}
                step="0.1"
                path="generator.stage_capacitance_uf"
                issues={allIssues}
              />
            </div>
          </fieldset>

          <fieldset className="group">
            <legend className="group__legend">Resistor inventory</legend>
            <NumberField
              label="Units available per stage"
              value={form.unitsPerStage}
              onChange={update("unitsPerStage")}
              issues={allIssues}
            />
            <InventoryPicker
              legend="Front resistors"
              stock={metadata.inventory.front}
              selected={form.frontOhmsAvailable}
              onChange={update("frontOhmsAvailable")}
              maxPerStage={metadata.inventory.max_front_ohms_per_stage}
            />
            <InventoryPicker
              legend="Tail resistors"
              stock={metadata.inventory.tail}
              selected={form.tailOhmsAvailable}
              onChange={update("tailOhmsAvailable")}
              maxPerStage={metadata.inventory.max_tail_ohms_per_stage}
            />
            <TextField
              label="External front resistors"
              unit="&#8486;, comma separated"
              value={form.externalFrontOhms}
              onChange={update("externalFrontOhms")}
              placeholder="e.g. 10000, 12000"
              path="inventory.external_front_ohms"
              issues={allIssues}
              hint={
                form.impulseType === "switching"
                  ? "Required for switching impulses: the stocked values cannot reach the needed resistance."
                  : "Leave empty unless external resistors are fitted."
              }
            />
            <TextField
              label="External tail resistors"
              unit="&#8486;, comma separated"
              value={form.externalTailOhms}
              onChange={update("externalTailOhms")}
              placeholder="e.g. 1200"
              path="inventory.external_tail_ohms"
              issues={allIssues}
            />
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
              <p>{optimiseMessage}</p>
            </section>
          ) : null}
        </form>

        <div className="panel panel--results">
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
            <p className="loading">Validating&hellip;</p>
          )}
          {impulseSpec ? (
            <p className="footnote">
              Showing {impulseSpec.label.toLowerCase()} targets (
              {impulseSpec.designation}). Waveform plots appear once the physics engine
              is implemented; nothing here is a predicted waveform.
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
