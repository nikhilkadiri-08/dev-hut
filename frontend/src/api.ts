import type {
  GeneratorProfile,
  Metadata,
  ProfileResponse,
  ValidationResponse,
} from "./types";

const UNREACHABLE =
  "Cannot reach the backend. Start it with: uvicorn backend.main:app --reload";

export class ApiError extends Error {}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError(
      `The backend returned a response that is not JSON (HTTP ${response.status}).`,
    );
  }
}

async function send(path: string, method: string, payload?: unknown): Promise<Response> {
  try {
    return await fetch(path, {
      method,
      headers: payload === undefined ? undefined : { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(UNREACHABLE);
  }
}

export async function fetchMetadata(): Promise<Metadata> {
  const response = await send("/api/metadata", "GET");
  if (!response.ok) {
    throw new ApiError(`Loading configuration failed (HTTP ${response.status}).`);
  }
  return readJson<Metadata>(response);
}

/**
 * Validation findings are a normal result, not a transport failure. The backend
 * returns HTTP 422 with the same body shape for schema problems, so both are parsed
 * the same way.
 */
async function readValidation(response: Response): Promise<ValidationResponse> {
  if (response.status !== 200 && response.status !== 422) {
    throw new ApiError(`Validation request failed (HTTP ${response.status}).`);
  }
  return readJson<ValidationResponse>(response);
}

/** Fully specified request, carrying its own generator ratings and inventory. */
export async function validateRequest(payload: unknown): Promise<ValidationResponse> {
  return readValidation(await send("/api/validate", "POST", payload));
}

/** Operator request: generator ratings come from the stored admin profile. */
export async function validateTestRequest(payload: unknown): Promise<ValidationResponse> {
  return readValidation(await send("/api/test-request/validate", "POST", payload));
}

async function readProfile(response: Response): Promise<ProfileResponse> {
  if (response.status !== 200 && response.status !== 422) {
    throw new ApiError(`Generator setup request failed (HTTP ${response.status}).`);
  }
  return readJson<ProfileResponse>(response);
}

export async function fetchProfile(): Promise<ProfileResponse> {
  return readProfile(await send("/api/admin/profile", "GET"));
}

export async function saveProfile(
  profile: GeneratorProfile,
  updatedBy: string,
): Promise<ProfileResponse> {
  return readProfile(
    await send("/api/admin/profile", "PUT", { profile, updated_by: updatedBy }),
  );
}

export async function resetProfile(): Promise<ProfileResponse> {
  return readProfile(await send("/api/admin/profile/reset", "POST"));
}

export interface WaveformPayload {
  model_kind: string;
  model_id: string;
  time_us: number[];
  voltage_kv: number[];
  time_s?: number[];
  voltage_v?: number[];
  resolution: { points: number; duration_s: number };
  units?: { time: string; voltage: string };
  assumptions?: string[];
  metrics: {
    crest_kv: number;
    peak_time_us: number;
    front_time_us: number | null;
    tail_time_us: number | null;
    unavailable: string[];
  };
  thresholds?: {
    thirty_percent_v: number;
    ninety_percent_v: number;
    fifty_percent_v: number;
    label: string;
  };
  consistency?: { ok?: boolean };
  chopped?: boolean;
  disclaimer: string;
}

export interface WaveformOverlay {
  time_us: number[];
  physics_kv: number[];
  corrected_kv?: number[] | null;
  requested_kv?: number[] | null;
  units?: { time: string; voltage: string };
  labels?: { physics?: string; corrected?: string; requested?: string };
}

export interface CorrectedMetrics {
  front_time_us: number;
  tail_time_us: number;
  crest_kv: number;
  peak_time_us?: number;
  extracted_front_time_us?: number | null;
  extracted_tail_time_us?: number | null;
}

export interface OptimiseCandidate {
  stages: number;
  charging_voltage_per_stage_kv: number;
  front_resistance_per_stage_ohm: number;
  tail_resistance_per_stage_ohm: number;
  front_build?: string;
  tail_build?: string;
  front_time_us: number;
  tail_time_us: number;
  crest_voltage_kv: number;
  confirmed_safe: boolean;
  safety_label: string;
  presentation: string;
  waveform?: WaveformPayload;
}

export interface SequenceShotOut {
  waveform_kind: string;
  level_fraction: number;
  requested_peak_kv: number;
  predicted_crest_kv?: number | null;
  tail_half_crossing_kv?: number | null;
  constraint_status: string;
  confirmed_safe: boolean;
  safety_label: string;
}

export interface OptimiseOutcome {
  implemented: boolean;
  detail: string;
  success?: boolean;
  mode?: string;
  candidate?: OptimiseCandidate | null;
  requestedPeakKv?: number | null;
  predictedCrestKv?: number | null;
  safetyLabel?: string | null;
  confirmedSafe?: boolean | null;
  sequence?: SequenceShotOut[] | null;
  warnings?: Array<{ code: string; message: string }>;
  waveform?: WaveformPayload | null;
  correctedWaveform?: WaveformPayload | null;
  requestedWaveform?: WaveformPayload | null;
  overlay?: WaveformOverlay | null;
  correctedMetrics?: CorrectedMetrics | null;
  physicsMetrics?: { front_time_us?: number; tail_time_us?: number; crest_voltage_kv?: number; peak_time_us?: number };
  inventoryNote?: string | null;
  frontBuild?: string | null;
  tailBuild?: string | null;
  presentation?: string | null;
}

export async function lookupLiterature(query: string): Promise<{
  query: string;
  matched: boolean;
  ranges: Array<{
    id: string;
    kind: string;
    label: string;
    quantity: string;
    typical_min: number;
    typical_max: number;
    unit: string;
    note: string;
    sources: Array<{ title: string; url: string; year: number | null }>;
  }>;
  insulation_levels: Array<{
    um_kv: number;
    standard_lightning_impulse_kv: number[];
    standard_switching_impulse_kv: number[];
  }>;
  windows: Array<{ id: string; designation: string; edition: string }>;
  caveats: Array<{ code: string; message: string }>;
}> {
  const response = await send(`/api/literature/lookup?q=${encodeURIComponent(query)}`, "GET");
  if (!response.ok) {
    throw new ApiError(`Literature lookup failed (HTTP ${response.status}).`);
  }
  return readJson(response);
}

export async function runOptimisation(payload: unknown): Promise<OptimiseOutcome> {
  const response = await send("/api/optimise", "POST", payload);
  const body = await readJson<{
    detail?: string;
    success?: boolean;
    mode?: string;
    candidate?: OptimiseCandidate | null;
    recommended?: OptimiseCandidate | null;
    best_unconfirmed?: OptimiseCandidate | null;
    predictions?: {
      requested_peak_kv?: number;
      crest_voltage_kv?: number;
      front_time_us?: number;
      tail_time_us?: number;
      peak_time_us?: number;
      waveform?: WaveformPayload;
      corrected_waveform?: WaveformPayload | null;
      requested_waveform?: WaveformPayload | null;
      waveform_overlay?: WaveformOverlay;
      corrected_metrics?: CorrectedMetrics | null;
    };
    corrected_waveform?: WaveformPayload | null;
    inventory_note?: string;
    sequence?: { shots?: SequenceShotOut[] };
    warnings?: Array<{ code: string; message: string }>;
    metadata?: { presentation?: string; residual_reason?: string | null };
    error?: { message?: string };
    errors?: Array<{ message?: string }>;
  }>(response);
  if (response.ok && body.success !== false) {
    const candidate = body.candidate ?? body.recommended ?? body.best_unconfirmed ?? null;
    const waveform = body.predictions?.waveform ?? candidate?.waveform ?? null;
    return {
      implemented: true,
      detail: body.detail ?? "Physics-only optimisation completed.",
      success: true,
      mode: body.mode,
      candidate,
      requestedPeakKv: body.predictions?.requested_peak_kv ?? null,
      predictedCrestKv: body.predictions?.crest_voltage_kv ?? candidate?.crest_voltage_kv ?? null,
      safetyLabel: candidate?.safety_label ?? null,
      confirmedSafe: candidate?.confirmed_safe ?? null,
      sequence: body.sequence?.shots ?? null,
      warnings: body.warnings,
      waveform,
      correctedWaveform: body.predictions?.corrected_waveform ?? body.corrected_waveform ?? null,
      requestedWaveform: body.predictions?.requested_waveform ?? null,
      overlay: body.predictions?.waveform_overlay ?? null,
      correctedMetrics: body.predictions?.corrected_metrics ?? null,
      physicsMetrics: {
        front_time_us: body.predictions?.front_time_us,
        tail_time_us: body.predictions?.tail_time_us,
        crest_voltage_kv: body.predictions?.crest_voltage_kv,
        peak_time_us: body.predictions?.peak_time_us,
      },
      inventoryNote: body.inventory_note ?? null,
      frontBuild: candidate?.front_build ?? null,
      tailBuild: candidate?.tail_build ?? null,
      presentation: body.metadata?.presentation ?? candidate?.presentation ?? null,
    };
  }
  const validationMessage = body.errors?.map((item) => item.message).filter(Boolean).join(" ");
  return {
    implemented: false,
    detail:
      body.detail ??
      body.error?.message ??
      validationMessage ??
      `Unexpected response (HTTP ${response.status}).`,
    success: false,
    mode: body.mode,
  };
}

export interface AnnEpoch {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_mae: number;
  val_mae: number;
  weights?: number[][][];
  biases?: number[][];
}

export interface AnnTrainingExample {
  id: number;
  split: string;
  impulse: string;
  features: number[];
  label: string;
}

export interface AnnTrainingHead {
  output: "front_us" | "tail_us" | "crest_kv" | string;
  impulse: string;
  active: boolean;
  reason: string;
  units: string;
  architecture: {
    input_size: number;
    input_features: string[];
    hidden_sizes: number[];
    hidden_activation: string;
    output_size: number;
    output_name: string;
  };
  epochs_executed: number;
  best_epoch: number;
  zero_residual_validation_mae: number;
  feature_mean: number[];
  feature_scale: number[];
  target_mean: number;
  target_scale: number;
  example: AnnTrainingExample;
  history: AnnEpoch[];
}

export interface AnnTrainingHistory {
  available: boolean;
  synthetic_only: boolean;
  laboratory_accuracy: boolean;
  disclaimer: string;
  loss_definition: string;
  weight_scale: string;
  weights_recorded_per_epoch: boolean;
  version: string;
  epoch_count: number;
  active_heads: Record<string, Record<string, boolean>>;
  feature_order: string[];
  heads: AnnTrainingHead[];
}

export async function fetchAnnTraining(): Promise<AnnTrainingHistory> {
  const response = await send("/api/ann/training", "GET");
  if (!response.ok) {
    throw new ApiError(`ANN training history failed to load (HTTP ${response.status}).`);
  }
  return readJson<AnnTrainingHistory>(response);
}

export interface MlDiagramMatrix {
  rows: number;
  cols: number;
  max_abs_weight: number;
  normalized: number[][];
}

export interface MlDiagramHead {
  output: "front_us" | "tail_us" | "crest_kv";
  label: string;
  unit: string;
  active: boolean;
  reason: string;
  hidden_activation: string;
  output_activation: string;
  layer_sizes: number[];
  hidden_sizes: number[];
  matrices: MlDiagramMatrix[];
}

export interface MlDiagram {
  impulse: string;
  input_features: Array<{ name: string; group: string; label: string }>;
  input_count: number;
  heads: MlDiagramHead[];
  shared_hidden_layers: boolean;
  hidden_activation: string;
}

export interface MlStatus {
  status: "ready" | "unavailable" | "error";
  available: boolean;
  synthetic_only: boolean;
  laboratory_validated: boolean;
  architecture_matches_reference: boolean;
  reference: {
    input_nodes: number;
    hidden_layers: number[];
    hidden_activation: string;
    outputs: string[];
    shared_network: boolean;
  };
  discrepancy: string;
  detail?: string | null;
  diagram: MlDiagram | null;
}

export interface MlResidualValue {
  label: string;
  unit: string;
  raw: number;
  added: number;
  applied: boolean;
  head_active: boolean;
  reason: string;
}

export interface MlResidualReport {
  status: "finished" | "unavailable" | "error" | "ready";
  available?: boolean;
  synthetic_only: boolean;
  laboratory_validated: boolean;
  architecture_matches_reference?: boolean;
  discrepancy?: string;
  impulse?: string;
  in_domain?: boolean | null;
  mode?: string | null;
  message?: string;
  physics?: { front_time_us: number; tail_time_us: number; crest_kv: number } | null;
  corrected?: { front_time_us: number; tail_time_us: number; crest_kv: number } | null;
  requested?: { peak_kv: number; front_time_us: number; tail_time_us: number } | null;
  features?: Array<{ name: string; group: string; label: string; value: number }> | null;
  residuals?: {
    front_us: MlResidualValue;
    tail_us: MlResidualValue;
    crest_kv: MlResidualValue;
  } | null;
}

export async function fetchMlStatus(impulse: "lightning" | "switching"): Promise<MlStatus> {
  const response = await send(`/api/ml/status?impulse=${impulse}`, "GET");
  if (!response.ok) {
    throw new ApiError(`ANN status failed to load (HTTP ${response.status}).`);
  }
  return readJson<MlStatus>(response);
}

export async function fetchMlResiduals(payload: unknown): Promise<MlResidualReport> {
  const response = await send("/api/ml/residuals", "POST", payload);
  const body = await readJson<
    MlResidualReport & {
      detail?: string;
      errors?: Array<{ message?: string }>;
    }
  >(response);
  if (!response.ok) {
    const validation = body.errors?.map((item) => item.message).filter(Boolean).join(" ");
    throw new ApiError(
      validation || body.message || body.detail || `Residual request failed (HTTP ${response.status}).`,
    );
  }
  return body;
}
