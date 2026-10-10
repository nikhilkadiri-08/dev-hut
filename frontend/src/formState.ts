import type {
  GeneratorProfile,
  ImpulseType,
  Metadata,
  Polarity,
  StoredProfile,
} from "./types";

/**
 * Numeric fields are held as strings so that a half-typed or empty entry survives in
 * the input box. The payload builders pass an unparseable string straight through to
 * the backend, which owns every error message the engineer sees.
 */

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

// ---------------------------------------------------------------------------
// Operator interface: one test request
// ---------------------------------------------------------------------------

export interface OperatorForm {
  testObjectTypeId: string;
  testObjectLabel: string;
  standardId: string;
  impulseType: ImpulseType;
  polarity: Polarity;
  requiredPeakKv: string;
  loadCapacitancePf: string;
  dividerCapacitancePf: string;
  strayCapacitancePf: string;
  /** When false, the two fields below are left to the laboratory defaults. */
  overrideLabDefaults: boolean;
  strayInductanceUh: string;
  expectedEfficiency: string;
  includeLevelSequence: boolean;
}

export function initialOperatorForm(
  metadata: Metadata,
  stored: StoredProfile,
): OperatorForm {
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
    overrideLabDefaults: false,
    strayInductanceUh: String(stored.profile.default_stray_inductance_uh),
    expectedEfficiency: String(stored.profile.default_efficiency),
    includeLevelSequence: false,
  };
}

export function buildOperatorPayload(form: OperatorForm) {
  const peak = numeric(form.requiredPeakKv);
  const payload: Record<string, unknown> = {
    test_object_type_id: form.testObjectTypeId,
    test_object_label: form.testObjectLabel.trim() || null,
    standard_id: form.standardId,
    impulse_type: form.impulseType,
    polarity: form.polarity,
    required_peak_kv: peak,
    load_capacitance_pf: numeric(form.loadCapacitancePf),
    divider_capacitance_pf: numeric(form.dividerCapacitancePf),
    stray_capacitance_pf: numeric(form.strayCapacitancePf),
    stray_inductance_uh: form.overrideLabDefaults ? numeric(form.strayInductanceUh) : null,
    expected_efficiency: form.overrideLabDefaults
      ? numeric(form.expectedEfficiency)
      : null,
  };
  if (form.includeLevelSequence && typeof peak === "number") {
    payload.sequence = {
      reference_peak_kv: peak,
      name: "operator_50_100",
      shots: [
        {
          waveform_kind: form.impulseType === "switching" ? "switching" : "lightning_full",
          level_fraction: 0.5,
          label: "50%",
        },
        {
          waveform_kind: form.impulseType === "switching" ? "switching" : "lightning_full",
          level_fraction: 1.0,
          label: "100%",
        },
      ],
    };
  }
  return payload;
}

// ---------------------------------------------------------------------------
// Admin interface: the laboratory's generator setup
// ---------------------------------------------------------------------------

export interface ResistorRow {
  ohms: string;
  unitsPerStage: string;
}

export interface ProfileForm {
  maxStages: string;
  minActiveStages: string;
  maxStageVoltageKv: string;
  stageCapacitanceUf: string;
  capacitorsPerStage: string;
  frontResistors: ResistorRow[];
  tailResistors: ResistorRow[];
  externalFrontOhms: string;
  externalTailOhms: string;
  defaultStrayInductanceUh: string;
  defaultEfficiency: string;
  updatedBy: string;
}

export function profileToForm(stored: StoredProfile): ProfileForm {
  const profile = stored.profile;
  const rows = (stock: GeneratorProfile["front_resistors"]): ResistorRow[] =>
    stock.map((item) => ({
      ohms: String(item.ohms),
      unitsPerStage: String(item.units_per_stage),
    }));

  return {
    maxStages: String(profile.max_stages),
    minActiveStages: String(profile.min_active_stages),
    maxStageVoltageKv: String(profile.max_stage_voltage_kv),
    stageCapacitanceUf: String(profile.stage_capacitance_uf),
    capacitorsPerStage: String(profile.capacitors_per_stage),
    frontResistors: rows(profile.front_resistors),
    tailResistors: rows(profile.tail_resistors),
    externalFrontOhms: profile.external_front_ohms.join(", "),
    externalTailOhms: profile.external_tail_ohms.join(", "),
    defaultStrayInductanceUh: String(profile.default_stray_inductance_uh),
    defaultEfficiency: String(profile.default_efficiency),
    updatedBy: stored.updated_by && stored.updated_by !== "unidentified" ? stored.updated_by : "",
  };
}

export function buildProfilePayload(form: ProfileForm) {
  const stock = (rows: ResistorRow[]) =>
    rows.map((row) => ({
      ohms: numeric(row.ohms),
      units_per_stage: integer(row.unitsPerStage),
    }));

  return {
    max_stages: integer(form.maxStages),
    min_active_stages: integer(form.minActiveStages),
    max_stage_voltage_kv: numeric(form.maxStageVoltageKv),
    stage_capacitance_uf: numeric(form.stageCapacitanceUf),
    capacitors_per_stage: integer(form.capacitorsPerStage),
    front_resistors: stock(form.frontResistors),
    tail_resistors: stock(form.tailResistors),
    external_front_ohms: ohmList(form.externalFrontOhms),
    external_tail_ohms: ohmList(form.externalTailOhms),
    default_stray_inductance_uh: numeric(form.defaultStrayInductanceUh),
    default_efficiency: numeric(form.defaultEfficiency),
  };
}
