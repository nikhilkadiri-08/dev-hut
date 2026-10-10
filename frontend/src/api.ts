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

export interface OptimiseOutcome {
  implemented: boolean;
  detail: string;
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
  const body = await readJson<{ detail?: string }>(response);
  return {
    implemented: response.ok,
    detail: body.detail ?? `Unexpected response (HTTP ${response.status}).`,
  };
}
