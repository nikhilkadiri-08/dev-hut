import { useCallback, useMemo, useState } from "react";
import { ApiError, resetProfile, saveProfile } from "../api";
import { buildProfilePayload, profileToForm, type ProfileForm } from "../formState";
import { NumberField, TextField } from "./Field";
import { IssueList } from "./IssueList";
import { ResistorTable } from "./ResistorTable";
import type { GeneratorProfile, Issue, Metadata, ProfileResponse } from "../types";

interface AdminViewProps {
  metadata: Metadata;
  profile: ProfileResponse;
  onProfileChange: (next: ProfileResponse) => void;
}

export function AdminView({ metadata, profile, onProfileChange }: AdminViewProps) {
  const [form, setForm] = useState<ProfileForm>(() => profileToForm(profile.stored));
  const [result, setResult] = useState<ProfileResponse>(profile);
  const [busy, setBusy] = useState(false);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const update = useCallback(
    <K extends keyof ProfileForm>(key: K) =>
      (value: ProfileForm[K]) =>
        setForm((current) => ({ ...current, [key]: value })),
    [],
  );

  const dirty = useMemo(
    () => JSON.stringify(form) !== JSON.stringify(profileToForm(profile.stored)),
    [form, profile.stored],
  );

  const issues: Issue[] = [...result.errors, ...result.warnings, ...result.notices];

  const onSave = useCallback(async () => {
    setBusy(true);
    setTransportError(null);
    setSavedAt(null);
    try {
      const payload = buildProfilePayload(form) as unknown as GeneratorProfile;
      const response = await saveProfile(payload, form.updatedBy);
      setResult(response);
      if (response.ok) {
        onProfileChange(response);
        setForm(profileToForm(response.stored));
        setSavedAt(response.stored.updated_at);
      }
    } catch (error: unknown) {
      setTransportError(error instanceof ApiError ? error.message : "Saving failed.");
    } finally {
      setBusy(false);
    }
  }, [form, onProfileChange]);

  const onReset = useCallback(async () => {
    setBusy(true);
    setTransportError(null);
    setSavedAt(null);
    try {
      const response = await resetProfile();
      setResult(response);
      onProfileChange(response);
      setForm(profileToForm(response.stored));
    } catch (error: unknown) {
      setTransportError(error instanceof ApiError ? error.message : "Reset failed.");
    } finally {
      setBusy(false);
    }
  }, [onProfileChange]);

  const reach = result.reach;

  return (
    <div className="layout">
      <form className="panel" onSubmit={(event) => event.preventDefault()}>
        <fieldset className="group">
          <legend className="group__legend">Ratings</legend>
          <p className="group__note">
            Limit {metadata.generator.max_stages} stages · {metadata.generator.max_stage_voltage_kv} kV
          </p>
          <div className="grid-2">
            <NumberField
              label="Stages available"
              value={form.maxStages}
              onChange={update("maxStages")}
              path="profile.max_stages"
              issues={issues}
            />
            <NumberField
              label="Minimum active stages"
              value={form.minActiveStages}
              onChange={update("minActiveStages")}
              path="profile.min_active_stages"
              issues={issues}
            />
            <NumberField
              label="Stage voltage"
              unit="kV"
              value={form.maxStageVoltageKv}
              onChange={update("maxStageVoltageKv")}
              path="profile.max_stage_voltage_kv"
              issues={issues}
            />
            <NumberField
              label="Stage capacitance"
              unit="&micro;F"
              value={form.stageCapacitanceUf}
              onChange={update("stageCapacitanceUf")}
              step="0.1"
              path="profile.stage_capacitance_uf"
              issues={issues}
            />
            <NumberField
              label="Capacitors per stage"
              value={form.capacitorsPerStage}
              onChange={update("capacitorsPerStage")}
              path="profile.capacitors_per_stage"
              issues={issues}
            />
          </div>
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Defaults</legend>
          <div className="grid-2">
            <NumberField
              label="Stray inductance"
              unit="&micro;H"
              value={form.defaultStrayInductanceUh}
              onChange={update("defaultStrayInductanceUh")}
              step="0.5"
              path="profile.default_stray_inductance_uh"
              issues={issues}
            />
            <NumberField
              label="Efficiency"
              unit="fraction"
              value={form.defaultEfficiency}
              onChange={update("defaultEfficiency")}
              step="0.01"
              path="profile.default_efficiency"
              issues={issues}
            />
          </div>
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Resistors</legend>
          <ResistorTable
            legend="Front resistors"
            rows={form.frontResistors}
            onChange={update("frontResistors")}
            path="profile.front_resistors"
            issues={issues}
            reachOhms={reach?.max_front_ohms_per_stage}
          />
          <ResistorTable
            legend="Tail resistors"
            rows={form.tailResistors}
            onChange={update("tailResistors")}
            path="profile.tail_resistors"
            issues={issues}
            reachOhms={reach?.max_tail_ohms_per_stage}
          />
          <TextField
            label="External front resistors"
            unit="&#8486;"
            value={form.externalFrontOhms}
            onChange={update("externalFrontOhms")}
            placeholder="10000, 12000"
            path="profile.external_front_ohms"
            issues={issues}
          />
          <TextField
            label="External tail resistors"
            unit="&#8486;"
            value={form.externalTailOhms}
            onChange={update("externalTailOhms")}
            placeholder="e.g. 1200"
            path="profile.external_tail_ohms"
            issues={issues}
          />
        </fieldset>

        <fieldset className="group">
          <legend className="group__legend">Save</legend>
          <TextField
            label="Changed by"
            value={form.updatedBy}
            onChange={update("updatedBy")}
            placeholder="name"
          />
          <div className="actions actions--row">
            <button
              type="button"
              className="button button--primary"
              onClick={onSave}
              disabled={busy || !dirty}
            >
              {busy ? "Saving\u2026" : "Save"}
            </button>
            <button
              type="button"
              className="button"
              onClick={onReset}
              disabled={busy || profile.stored.is_factory_default}
            >
              Restore defaults
            </button>
          </div>
          <p className="actions__note">
            {dirty ? "Unsaved changes." : savedAt ? "Saved." : "No unsaved changes."}
          </p>
        </fieldset>
      </form>

      <div className="panel panel--results">
        {transportError ? (
          <section className="card card--error">
            <p>{transportError}</p>
          </section>
        ) : null}

        <section className="card">
          <h3 className="card__title">
            Saved setup
            {profile.stored.is_factory_default ? (
              <span className="pill pill--warning">supplied defaults</span>
            ) : (
              <span className="pill pill--ok">configured</span>
            )}
          </h3>
          <p className="card__subtitle">
            {profile.stored.is_factory_default
              ? "Supplied defaults."
              : `Changed by ${profile.stored.updated_by}.`}
          </p>
          {reach ? (
            <dl className="metrics">
              <div className="metric">
                <dt>Max crest</dt>
                <dd>{reach.max_achievable_peak_kv.toLocaleString(undefined, { maximumFractionDigits: 0 })} kV</dd>
                <span className="metric__note">
                  at {(result.stored.profile.default_efficiency * 100).toFixed(0)}%
                  efficiency
                </span>
              </div>
              <div className="metric">
                <dt>Total charge voltage</dt>
                <dd>{reach.max_charge_voltage_kv.toLocaleString()} kV</dd>
                <span className="metric__note">all stages</span>
              </div>
              <div className="metric">
                <dt>Front</dt>
                <dd>{reach.max_front_ohms_per_stage.toLocaleString()} &#8486;</dd>
                <span className="metric__note">{reach.front_values} values</span>
              </div>
              <div className="metric">
                <dt>Tail</dt>
                <dd>{reach.max_tail_ohms_per_stage.toLocaleString()} &#8486;</dd>
                <span className="metric__note">{reach.tail_values} values</span>
              </div>
            </dl>
          ) : null}
        </section>

        <IssueList severity="error" issues={result.errors} />
        <IssueList severity="warning" issues={result.warnings} />
        <IssueList severity="notice" issues={result.notices} />
      </div>
    </div>
  );
}
