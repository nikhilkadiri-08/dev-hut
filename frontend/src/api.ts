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

export interface OptimiseCandidate {
  stages: number;
  charging_voltage_per_stage_kv: number;
  front_resistance_per_stage_ohm: number;
  tail_resistance_per_stage_ohm: number;
  front_time_us: number;
  tail_time_us: number;
  crest_voltage_kv: number;
  confirmed_safe: boolean;
  safety_label: string;
  presentation: string;
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
    predictions?: { requested_peak_kv?: number; crest_voltage_kv?: number };
    sequence?: { shots?: SequenceShotOut[] };
    warnings?: Array<{ code: string; message: string }>;
    error?: { message?: string };
    errors?: Array<{ message?: string }>;
  }>(response);
  if (response.ok && body.success !== false) {
    const candidate = body.candidate ?? body.recommended ?? body.best_unconfirmed ?? null;
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
