/** Mirrors the backend response models in backend/schemas.py. */

export type ImpulseType = "lightning" | "switching";
export type Polarity = "positive" | "negative";
export type Severity = "error" | "warning" | "notice";
export type ProvenanceStatus =
  | "supplied"
  | "needs_verification"
  | "conflict"
  | "placeholder";

export interface Provenance {
  source: string;
  locator: string;
  status: ProvenanceStatus;
  note: string;
}

export interface Issue {
  severity: Severity;
  code: string;
  message: string;
  field: string | null;
  suggestion: string | null;
}

export interface DomainCheck {
  field: string;
  label: string;
  value: number;
  training_min: number;
  training_max: number;
  in_domain: boolean;
  unit: string;
}

export interface AcceptanceWindow {
  nominal: number;
  min: number;
  max: number;
  unit: string;
  provenance_status: ProvenanceStatus;
  provenance_note: string;
}

export interface DerivedQuantities {
  total_load_capacitance_pf: number;
  stages_required: number;
  charging_voltage_per_stage_kv: number;
  stage_utilisation: number;
  generator_capacitance_uf: number;
  max_achievable_peak_kv: number;
  workbook_reference: string;
}

export interface NormalizedRequest {
  test_object_id: string;
  test_object_label: string | null;
  standard_id: string;
  standard_label: string;
  impulse_type: ImpulseType;
  impulse_designation: string;
  polarity: Polarity;
  required_peak_kv: number;
  target_front_time_us: number;
  target_half_value_time_us: number;
  front_time_definition: string;
  front_time_window: AcceptanceWindow;
  half_value_time_window: AcceptanceWindow;
  peak_relative_tolerance: number;
  derived: DerivedQuantities;
}

export interface AssumptionEntry {
  key: string;
  subject: string;
  value: string;
  impact: string;
  source: string;
  locator: string;
  status: ProvenanceStatus;
  note: string;
}

export interface ValidationResponse {
  ok: boolean;
  errors: Issue[];
  warnings: Issue[];
  notices: Issue[];
  normalized: NormalizedRequest | null;
  domain_checks: DomainCheck[];
  relevant_assumptions: AssumptionEntry[];
}

export interface ResistorStock {
  ohms: number;
  units_per_stage: number;
}

/** The laboratory's standing generator setup, owned by the admin interface. */
export interface GeneratorProfile {
  max_stages: number;
  min_active_stages: number;
  max_stage_voltage_kv: number;
  stage_capacitance_uf: number;
  capacitors_per_stage: number;
  front_resistors: ResistorStock[];
  tail_resistors: ResistorStock[];
  external_front_ohms: number[];
  external_tail_ohms: number[];
  default_stray_inductance_uh: number;
  default_efficiency: number;
}

export interface StoredProfile {
  profile: GeneratorProfile;
  updated_at: string | null;
  updated_by: string | null;
  is_factory_default: boolean;
}

export interface ProfileReach {
  max_front_ohms_per_stage: number;
  max_tail_ohms_per_stage: number;
  max_charge_voltage_kv: number;
  max_achievable_peak_kv: number;
  front_values: number;
  tail_values: number;
  has_external_front: boolean;
  has_external_tail: boolean;
}

export interface ProfileResponse {
  stored: StoredProfile;
  ok: boolean;
  errors: Issue[];
  warnings: Issue[];
  notices: Issue[];
  reach: ProfileReach | null;
}

export interface LiteratureSource {
  title: string;
  url: string;
  year: number | null;
  excerpt?: string;
}

export interface LiteratureRange {
  id: string;
  kind: string;
  label: string;
  quantity: string;
  typical_min: number;
  typical_max: number;
  unit: string;
  note: string;
  sources: LiteratureSource[];
}

export interface LiteratureCatalogue {
  disclaimer: string;
  retrieved: string;
  ranges: LiteratureRange[];
  workbook_conflicts: { id: string; edition: string; detail: string }[];
}

export interface TestObjectType {
  id: string;
  label: string;
  description: string;
  default_load_capacitance_pf: number;
  typical_min_pf: number;
  typical_max_pf: number;
  provenance: Provenance;
  literature_kind?: string | null;
}

export interface ImpulseSpecMeta {
  impulse_type: ImpulseType;
  label: string;
  designation: string;
  front_time: { nominal: number; min: number; max: number; unit: string };
  half_value_time: { nominal: number; min: number; max: number; unit: string };
  peak: { relative: number };
  front_time_definition: { text: string; provenance: Provenance };
}

export interface Metadata {
  generator: {
    max_stages: number;
    min_active_stages: number;
    max_stage_voltage_kv: number;
    stage_capacitance_uf: number;
    capacitors_per_stage: number;
    min_test_voltage_kv: number;
    max_test_voltage_kv: number;
    max_charge_voltage_kv: number;
  };
  standard: { id: string; label: string; edition_note: string };
  impulse_types: ImpulseSpecMeta[];
  test_objects: TestObjectType[];
  inventory: {
    front: ResistorStock[];
    tail: ResistorStock[];
    max_front_ohms_per_stage: number;
    max_tail_ohms_per_stage: number;
  };
  defaults: {
    required_peak_kv: number;
    divider_capacitance_pf: number;
    stray_capacitance_pf: number;
    stray_inductance_uh: number;
    expected_efficiency: number;
  };
  assumptions: AssumptionEntry[];
  assumptions_requiring_attention: AssumptionEntry[];
  capabilities: Record<string, string>;
  literature?: LiteratureCatalogue;
}
