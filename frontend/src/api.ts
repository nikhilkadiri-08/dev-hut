import type { Metadata, ValidationResponse } from "./types";

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

export async function fetchMetadata(): Promise<Metadata> {
  let response: Response;
  try {
    response = await fetch("/api/metadata");
  } catch {
    throw new ApiError(
      "Cannot reach the backend. Start it with: uvicorn backend.main:app --reload",
    );
  }
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
export async function validateRequest(payload: unknown): Promise<ValidationResponse> {
  let response: Response;
  try {
    response = await fetch("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(
      "Cannot reach the backend. Start it with: uvicorn backend.main:app --reload",
    );
  }
  if (response.status !== 200 && response.status !== 422) {
    throw new ApiError(`Validation request failed (HTTP ${response.status}).`);
  }
  return readJson<ValidationResponse>(response);
}

export interface OptimiseOutcome {
  implemented: boolean;
  detail: string;
}

export async function runOptimisation(payload: unknown): Promise<OptimiseOutcome> {
  const response = await fetch("/api/optimise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await readJson<{ detail?: string }>(response);
  return {
    implemented: response.ok,
    detail: body.detail ?? `Unexpected response (HTTP ${response.status}).`,
  };
}
