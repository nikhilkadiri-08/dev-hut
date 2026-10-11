import type {
  AssumptionEntry,
  DomainCheck,
  NormalizedRequest,
  ProvenanceStatus,
} from "../types";

const STATUS_LABEL: Record<ProvenanceStatus, string> = {
  supplied: "from supplied material",
  needs_verification: "needs verification",
  conflict: "disputed",
  placeholder: "placeholder",
};

function format(value: number, digits = 2): string {
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

/** Trims the spurious precision that synthetic data carries, e.g. 1350.0007 -> 1,350. */
function concise(value: number): string {
  const magnitude = Math.abs(value);
  return format(value, magnitude >= 100 ? 0 : magnitude >= 1 ? 2 : 3);
}

export function TargetSummary({ normalized }: { normalized: NormalizedRequest }) {
  const { front_time_window: front, half_value_time_window: tail } = normalized;
  return (
    <section className="card">
      <h3 className="card__title">Target</h3>
      <p className="card__subtitle">
        {normalized.impulse_designation} · {normalized.polarity}
      </p>
      <dl className="metrics">
        <div className="metric">
          <dt>Crest</dt>
          <dd>{format(normalized.required_peak_kv)} kV</dd>
          <span className="metric__note">
            &plusmn;{format(normalized.peak_relative_tolerance * 100)}%
          </span>
        </div>
        <div className="metric">
          <dt>Front</dt>
          <dd>
            {format(front.nominal)} {front.unit}
          </dd>
          <span className="metric__note">
            {format(front.min)}&ndash;{format(front.max)} {front.unit}
            {front.provenance_status !== "supplied" ? (
              <em className={`tag tag--${front.provenance_status}`}>
                {STATUS_LABEL[front.provenance_status]}
              </em>
            ) : null}
          </span>
        </div>
        <div className="metric">
          <dt>Tail</dt>
          <dd>
            {format(tail.nominal)} {tail.unit}
          </dd>
          <span className="metric__note">
            {format(tail.min)}&ndash;{format(tail.max)} {tail.unit}
            {tail.provenance_status !== "supplied" ? (
              <em className={`tag tag--${tail.provenance_status}`}>
                {STATUS_LABEL[tail.provenance_status]}
              </em>
            ) : null}
          </span>
        </div>
      </dl>
    </section>
  );
}

export function CapacitySummary({ normalized }: { normalized: NormalizedRequest }) {
  const d = normalized.derived;
  return (
    <section className="card">
      <h3 className="card__title">Capacity</h3>
      <dl className="metrics">
        <div className="metric">
          <dt>Stages to use</dt>
          <dd>{d.stages_required}</dd>
        </div>
        <div className="metric">
          <dt>Charging voltage per stage</dt>
          <dd>{format(d.charging_voltage_per_stage_kv, 1)} kV</dd>
          <span className="metric__note">
            {format(d.stage_utilisation * 100, 1)}% of the stage rating
          </span>
        </div>
        <div className="metric">
          <dt>Generator capacitance</dt>
          <dd>{format(d.generator_capacitance_uf, 4)} &micro;F</dd>
        </div>
        <div className="metric">
          <dt>Total load capacitance</dt>
          <dd>{format(d.total_load_capacitance_pf)} pF</dd>
        </div>
        <div className="metric">
          <dt>Highest reachable peak</dt>
          <dd>{format(d.max_achievable_peak_kv)} kV</dd>
        </div>
      </dl>
    </section>
  );
}

export function DomainPanel({ checks }: { checks: DomainCheck[] }) {
  if (checks.length === 0) return null;
  const outside = checks.filter((check) => !check.in_domain).length;
  return (
    <section className="card">
      <h3 className="card__title">
        Training domain
        {outside > 0 ? (
          <span className="pill pill--warning">{outside} outside</span>
        ) : (
          <span className="pill pill--ok">all inside</span>
        )}
      </h3>
      <table className="table">
        <thead>
          <tr>
            <th>Input</th>
            <th>Your value</th>
            <th>Trained on</th>
            <th aria-label="status" />
          </tr>
        </thead>
        <tbody>
          {checks.map((check) => (
            <tr key={check.field} className={check.in_domain ? "" : "row--warning"}>
              <td>{check.label}</td>
              <td className="numeric">
                {concise(check.value)} {check.unit}
              </td>
              <td className="numeric">
                {concise(check.training_min)}&ndash;{concise(check.training_max)}{" "}
                {check.unit}
              </td>
              <td>{check.in_domain ? "inside" : "outside"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function AssumptionsPanel({ entries }: { entries: AssumptionEntry[] }) {
  if (entries.length === 0) return null;
  const unresolved = entries.filter((entry) => entry.status !== "supplied");
  return (
    <section className="card">
      <h3 className="card__title">
        Assumptions
        {unresolved.length > 0 ? (
          <span className="pill pill--warning">{unresolved.length} unverified</span>
        ) : null}
      </h3>
      <ul className="assumptions">
        {entries.map((entry) => (
          <li key={entry.key} className="assumption">
            <div className="assumption__head">
              <strong>{entry.subject}</strong>
              <em className={`tag tag--${entry.status}`}>{STATUS_LABEL[entry.status]}</em>
            </div>
            <p className="assumption__value">{entry.value}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
