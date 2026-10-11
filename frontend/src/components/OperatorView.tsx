import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  fetchMlResiduals,
  runOptimisation,
  validateTestRequest,
  type CorrectedMetrics,
  type MlResidualReport,
  type SequenceShotOut,
  type WaveformOverlay,
  type WaveformPayload,
} from "../api";
import { AnnInferenceDiagram } from "./AnnInferenceDiagram";
import { WaveformChart } from "./WaveformChart";
import {
  buildOperatorPayload,
  initialOperatorForm,
  type OperatorForm,
} from "../formState";
import { Field, NumberField, TextField } from "./Field";
import { LiteratureHint } from "./LiteratureHint";
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
  const [optimiseImplemented, setOptimiseImplemented] = useState(false);
  const [optimiseMode, setOptimiseMode] = useState<string | null>(null);
  const [requestedCrestKv, setRequestedCrestKv] = useState<number | null>(null);
  const [predictedCrestKv, setPredictedCrestKv] = useState<number | null>(null);
  const [confirmedSafe, setConfirmedSafe] = useState<boolean | null>(null);
  const [sequenceShots, setSequenceShots] = useState<SequenceShotOut[] | null>(null);
  const [waveform, setWaveform] = useState<WaveformPayload | null>(null);
  const [correctedWaveform, setCorrectedWaveform] = useState<WaveformPayload | null>(null);
  const [requestedWaveform, setRequestedWaveform] = useState<WaveformPayload | null>(null);
  const [overlay, setOverlay] = useState<WaveformOverlay | null>(null);
  const [correctedMetrics, setCorrectedMetrics] = useState<CorrectedMetrics | null>(null);
  const [physicsFrontUs, setPhysicsFrontUs] = useState<number | null>(null);
  const [physicsTailUs, setPhysicsTailUs] = useState<number | null>(null);
  const [frontBuild, setFrontBuild] = useState<string | null>(null);
  const [tailBuild, setTailBuild] = useState<string | null>(null);
  const [presentation, setPresentation] = useState<string | null>(null);
  const [optimiseWarnings, setOptimiseWarnings] = useState<Array<{ code: string; message: string }>>([]);
  const [annPhase, setAnnPhase] = useState<"idle" | "running">("idle");
  const [annReport, setAnnReport] = useState<MlResidualReport | null>(null);
  const [annError, setAnnError] = useState<string | null>(null);

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

  const optimiseRequest = useRef(0);
  const onOptimise = useCallback(async () => {
    const request = ++optimiseRequest.current;
    setOptimiseMessage(null);
    setOptimiseImplemented(false);
    setOptimiseMode(null);
    setRequestedCrestKv(null);
    setPredictedCrestKv(null);
    setConfirmedSafe(null);
    setSequenceShots(null);
    setWaveform(null);
    setCorrectedWaveform(null);
    setRequestedWaveform(null);
    setOverlay(null);
    setCorrectedMetrics(null);
    setPhysicsFrontUs(null);
    setPhysicsTailUs(null);
    setFrontBuild(null);
    setTailBuild(null);
    setPresentation(null);
    setOptimiseWarnings([]);
    setAnnPhase("running");
    setAnnError(null);
    const residualPromise = fetchMlResiduals(latestPayload.current).then(
      (report) => ({ report, error: null as string | null }),
      (reason: unknown) => ({
        report: null,
        error: reason instanceof ApiError ? reason.message : "Could not reach the residual endpoint.",
      }),
    );
    try {
      const outcome = await runOptimisation(latestPayload.current);
      if (request !== optimiseRequest.current) return;
      setOptimiseMessage(outcome.detail);
      setOptimiseImplemented(outcome.implemented);
      setOptimiseMode(outcome.mode ?? null);
      setRequestedCrestKv(outcome.requestedPeakKv ?? null);
      setPredictedCrestKv(outcome.predictedCrestKv ?? null);
      setConfirmedSafe(outcome.confirmedSafe ?? null);
      setSequenceShots(outcome.sequence ?? null);
      setWaveform(outcome.waveform ?? null);
      setCorrectedWaveform(outcome.correctedWaveform ?? null);
      setRequestedWaveform(outcome.requestedWaveform ?? null);
      setOverlay(outcome.overlay ?? null);
      setCorrectedMetrics(outcome.correctedMetrics ?? null);
      setPhysicsFrontUs(outcome.physicsMetrics?.front_time_us ?? null);
      setPhysicsTailUs(outcome.physicsMetrics?.tail_time_us ?? null);
      setFrontBuild(outcome.frontBuild ?? null);
      setTailBuild(outcome.tailBuild ?? null);
      setPresentation(outcome.presentation ?? null);
      setOptimiseWarnings(outcome.warnings ?? []);
    } catch (error: unknown) {
      setOptimiseMessage(
        error instanceof ApiError ? error.message : "Could not reach the optimisation endpoint.",
      );
      setOptimiseImplemented(false);
    }
    const residual = await residualPromise;
    if (request !== optimiseRequest.current) return;
    setAnnReport(residual.report);
    setAnnError(residual.error);
    setAnnPhase("idle");
  }, []);

  // A previous lightning plot must not stay on screen after the impulse type
  // changes. Ask for the waveform that matches the selection.
  const displayedWaveform = useRef(false);
  displayedWaveform.current = waveform !== null;
  const seenImpulseType = useRef(form.impulseType);
  useEffect(() => {
    if (seenImpulseType.current === form.impulseType) return;
    seenImpulseType.current = form.impulseType;
    if (displayedWaveform.current) {
      void onOptimise();
    } else {
      setAnnReport(null);
      setAnnError(null);
    }
  }, [form.impulseType, onOptimise]);

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
            Generator
            {profile.stored.is_factory_default ? (
              <span className="pill pill--warning">unconfirmed</span>
            ) : null}
          </h3>
          <p className="card__body">
            {lab.max_stages} stages · {lab.max_stage_voltage_kv} kV ·{" "}
            {lab.stage_capacitance_uf} &micro;F · {lab.front_resistors.length} front ·{" "}
            {lab.tail_resistors.length} tail
          </p>
          <button type="button" className="button button--ghost" onClick={onOpenAdmin}>
            Generator setup
          </button>
        </section>

        <fieldset className="group">
          <legend className="group__legend">Test object</legend>
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
          {selectedObject?.provenance.status === "placeholder" ? (
            <em className="tag tag--placeholder">Placeholder capacitance</em>
          ) : null}
          <LiteratureHint
            selected={selectedObject}
            ranges={metadata.literature?.ranges ?? []}
            allowSearch={form.testObjectTypeId === "custom" || Boolean(selectedObject?.literature_kind)}
          />
          <TextField
            label="Reference"
            value={form.testObjectLabel}
            onChange={update("testObjectLabel")}
            placeholder="serial number"
            path="test_object_label"
            issues={issues}
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Target</legend>
          <Field label="Standard" path="standard_id" issues={issues}>
            <select
              className="input"
              value={form.standardId}
              onChange={(event) => update("standardId")(event.target.value)}
            >
              <option value={metadata.standard.id}>{metadata.standard.label}</option>
            </select>
          </Field>

          <Field label="Impulse" path="impulse_type" issues={issues}>
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
            label="Crest"
            unit="kV"
            value={form.requiredPeakKv}
            onChange={update("requiredPeakKv")}
            step="5"
            path="required_peak_kv"
            issues={issues}
            hint={`Up to ${(
              lab.max_stages *
              lab.max_stage_voltage_kv *
              lab.default_efficiency
            ).toFixed(0)} kV`}
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Circuit</legend>
          <NumberField
            label="Load capacitance"
            unit="pF"
            value={form.loadCapacitancePf}
            onChange={update("loadCapacitancePf")}
            path="load_capacitance_pf"
            issues={issues}
          />
          <NumberField
            label="Divider capacitance"
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
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Defaults</legend>
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
                label="Efficiency"
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
              {lab.default_stray_inductance_uh} &micro;H · efficiency {lab.default_efficiency}
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
            Run
          </button>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={form.includeLevelSequence}
              onChange={(event) =>
                update("includeLevelSequence")(event.target.checked)
              }
            />
            <span>Also run 50% then 100%</span>
          </label>
          {result && !result.ok ? (
            <p className="field__issue field__issue--error">Fix the errors, then Run.</p>
          ) : null}
        </div>
        {optimiseMessage ? (
          <section className={`card ${optimiseImplemented ? "card--quiet" : "card--pending"}`}>
            <h3 className="card__title">
              {optimiseImplemented
                ? optimiseMode === "physics_only"
                  ? "Physics"
                  : optimiseMode === "physics_plus_ann"
                    ? "Physics + ANN"
                    : "Result"
                : "Could not run"}
            </h3>
            {optimiseImplemented ? null : <p className="card__body">{optimiseMessage}</p>}
            {optimiseImplemented && requestedCrestKv != null && predictedCrestKv != null ? (
              <p className="card__body">
                Crest {requestedCrestKv.toFixed(1)} kV requested · {predictedCrestKv.toFixed(1)} kV predicted
              </p>
            ) : null}
            {confirmedSafe === false ? <p className="card__body">Not confirmed safe.</p> : null}
            {form.impulseType === "switching" &&
            (presentation === "closest_stocked_noncompliant" ||
              presentation === "buildable_unconfirmed_noncompliant") ? (
              <p className="card__body">Switching resistors cannot meet this window.</p>
            ) : null}
            {sequenceShots && sequenceShots.length > 0 ? (
              <ul className="card__body">
                {sequenceShots.map((shot, index) => (
                  <li key={`${shot.waveform_kind}-${index}`}>
                    {shot.level_fraction}× · {shot.requested_peak_kv.toFixed(1)} kV requested ·{" "}
                    {shot.predicted_crest_kv ?? "—"} kV · {shot.constraint_status} · {shot.safety_label}
                  </li>
                ))}
              </ul>
            ) : null}
            {frontBuild || tailBuild ? (
              <p className="card__body">
                Front {frontBuild ?? "—"} · Tail {tailBuild ?? "—"}
              </p>
            ) : null}
            {optimiseWarnings.some((item) => item.code.includes("SYNTHETIC")) ? (
              <p className="card__body">Synthetic training replay.</p>
            ) : null}
            {optimiseWarnings.some(
              (item) => item.code.includes("TAUTOLOGY") && /not applied/i.test(item.message),
            ) ? (
              <p className="card__body">Crest residual not added.</p>
            ) : null}
            {optimiseWarnings
              .filter((item) =>
                [
                  "ANN_PREDICT_FAILED",
                  "ANN_WAVEFORM_FIT_FAILED",
                  "ANN_WAVEFORM_INCONSISTENT",
                  "ANN_RESIDUAL_REJECTED",
                ].includes(item.code),
              )
              .map((item) => (
                <p key={item.code} className="card__body">
                  {item.message}
                </p>
              ))}
            {optimiseImplemented && waveform ? (
              <WaveformChart
                waveform={waveform}
                requestedPeakKv={requestedCrestKv}
                overlay={overlay}
                corrected={correctedWaveform}
                requested={requestedWaveform}
                physicsFrontUs={physicsFrontUs}
                physicsTailUs={physicsTailUs}
                physicsCrestKv={predictedCrestKv}
                correctedMetrics={correctedMetrics}
                requestedFrontUs={result?.normalized?.target_front_time_us ?? null}
                requestedTailUs={result?.normalized?.target_half_value_time_us ?? null}
                impulseType={form.impulseType}
              />
            ) : null}
            {optimiseImplemented && !waveform ? (
              <p className="card__body">No waveform returned.</p>
            ) : null}
          </section>
        ) : null}
      </form>

      <div className="panel panel--results">
        <AnnInferenceDiagram
          impulseType={form.impulseType}
          phase={annPhase}
          report={annReport}
          error={annError}
        />
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
                Assumptions ({result.relevant_assumptions.length})
              </summary>
              <AssumptionsPanel entries={result.relevant_assumptions} />
            </details>
            <details className="disclosure disclosure--block">
              <summary>Status</summary>
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
          <p className="loading">{validating ? "Checking\u2026" : "No result yet."}</p>
        )}
      </div>
    </div>
  );
}
