import type { ImpulseType, Metadata, Polarity } from "./types";

/**
 * Numeric fields are held as strings so that a half-typed or empty entry survives in
 * the input box. The payload builder passes an unparseable string straight through to
 * the backend, which owns every error message the engineer sees.
 */
export interface FormState {
  testObjectTypeId: string;
  testObjectLabel: string;
  standardId: string;
  impulseType: ImpulseType;
  polarity: Polarity;
  requiredPeakKv: string;
  loadCapacitancePf: string;
  dividerCapacitancePf: string;
  strayCapacitancePf: string;
  strayInductanceUh: string;
  expectedEfficiency: string;
  maxStages: string;
  minActiveStages: string;
  maxStageVoltageKv: string;
  stageCapacitanceUf: string;
  frontOhmsAvailable: number[];
  tailOhmsAvailable: number[];
  unitsPerStage: string;
  externalFrontOhms: string;
  externalTailOhms: string;
}

export function initialForm(metadata: Metadata): FormState {
  const workbookObject =
    metadata.test_objects.find((item) => item.id === "workbook_example") ??
    metadata.test_objects[0];

  return {
    testObjectTypeId: workbookObject.id,
    testObjectLabel: "",
    standardId: metadata.standard.id,
    impulseType: "lightning",
    polarity: "positive",
    requiredPeakKv: String(metadata.defaults.required_peak_kv),
    loadCapacitancePf: String(workbookObject.default_load_capacitance_pf),
    dividerCapacitancePf: String(metadata.defaults.divider_capacitance_pf),
    strayCapacitancePf: String(metadata.defaults.stray_capacitance_pf),
    strayInductanceUh: String(metadata.defaults.stray_inductance_uh),
    expectedEfficiency: String(metadata.defaults.expected_efficiency),
    maxStages: String(metadata.generator.max_stages),
    minActiveStages: String(metadata.generator.min_active_stages),
    maxStageVoltageKv: String(metadata.generator.max_stage_voltage_kv),
    stageCapacitanceUf: String(metadata.generator.stage_capacitance_uf),
    frontOhmsAvailable: metadata.inventory.front.map((item) => item.ohms),
    tailOhmsAvailable: metadata.inventory.tail.map((item) => item.ohms),
    unitsPerStage: String(metadata.inventory.front[0]?.units_per_stage ?? 4),
    externalFrontOhms: "",
    externalTailOhms: "",
  };
}

/** A number when the text parses, otherwise the raw text so the backend can complain. */
function numeric(raw: string): number | string {
  const trimmed = raw.trim();
  if (trimmed === "") return "";
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : trimmed;
}

function integer(raw: string): number | string {
  const value = numeric(raw);
  return typeof value === "number" ? Math.round(value) : value;
}

function ohmList(raw: string): (number | string)[] {
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .map((part) => {
      const parsed = Number(part);
      return Number.isFinite(parsed) ? parsed : part;
    });
}

export function buildPayload(form: FormState) {
  const units = integer(form.unitsPerStage);
  return {
    test_object_type_id: form.testObjectTypeId,
    test_object_label: form.testObjectLabel.trim() || null,
    standard_id: form.standardId,
    impulse_type: form.impulseType,
    polarity: form.polarity,
    required_peak_kv: numeric(form.requiredPeakKv),
    circuit: {
      load_capacitance_pf: numeric(form.loadCapacitancePf),
      divider_capacitance_pf: numeric(form.dividerCapacitancePf),
      stray_capacitance_pf: numeric(form.strayCapacitancePf),
      stray_inductance_uh: numeric(form.strayInductanceUh),
      expected_efficiency: numeric(form.expectedEfficiency),
    },
    generator: {
      max_stages: integer(form.maxStages),
      min_active_stages: integer(form.minActiveStages),
      max_stage_voltage_kv: numeric(form.maxStageVoltageKv),
      stage_capacitance_uf: numeric(form.stageCapacitanceUf),
    },
    inventory: {
      front: form.frontOhmsAvailable.map((ohms) => ({ ohms, units_per_stage: units })),
      tail: form.tailOhmsAvailable.map((ohms) => ({ ohms, units_per_stage: units })),
      external_front_ohms: ohmList(form.externalFrontOhms),
      external_tail_ohms: ohmList(form.externalTailOhms),
    },
  };
}
