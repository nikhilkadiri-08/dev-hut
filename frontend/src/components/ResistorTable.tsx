import type { ResistorRow } from "../formState";
import type { Issue } from "../types";

interface ResistorTableProps {
  legend: string;
  rows: ResistorRow[];
  onChange: (rows: ResistorRow[]) => void;
  /** Dotted backend path, e.g. "profile.front_resistors", for inline messages. */
  path: string;
  issues: Issue[];
  reachOhms?: number;
}

export function ResistorTable({
  legend,
  rows,
  onChange,
  path,
  issues,
  reachOhms,
}: ResistorTableProps) {
  const update = (index: number, patch: Partial<ResistorRow>) =>
    onChange(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  const remove = (index: number) => onChange(rows.filter((_, i) => i !== index));
  const add = () => onChange([...rows, { ohms: "", unitsPerStage: "4" }]);

  const own = issues.filter((issue) => issue.field?.startsWith(path));
  const rowIssue = (index: number) =>
    own.filter((issue) => issue.field?.startsWith(`${path}[${index}]`));

  return (
    <fieldset className="stock">
      <legend className="stock__legend">
        {legend}
        {reachOhms !== undefined ? (
          <span className="stock__reach">
            reaches {reachOhms.toLocaleString()} &#8486;/stage in series
          </span>
        ) : null}
      </legend>

      <table className="stock__table">
        <thead>
          <tr>
            <th>Value (&#8486;)</th>
            <th>Units per stage</th>
            <th aria-label="remove" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const problems = rowIssue(index);
            return (
              <tr key={index} className={problems.length > 0 ? "row--error" : ""}>
                <td>
                  <input
                    className="input input--compact"
                    type="number"
                    value={row.ohms}
                    onChange={(event) => update(index, { ohms: event.target.value })}
                  />
                </td>
                <td>
                  <input
                    className="input input--compact"
                    type="number"
                    value={row.unitsPerStage}
                    onChange={(event) =>
                      update(index, { unitsPerStage: event.target.value })
                    }
                  />
                </td>
                <td>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Remove ${row.ohms || "empty"} ohm resistor`}
                    onClick={() => remove(index)}
                  >
                    &times;
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <button type="button" className="button button--ghost" onClick={add}>
        Add resistor
      </button>

      {own
        .filter((issue) => issue.field === path)
        .map((issue) => (
          <p key={issue.code} className={`field__issue field__issue--${issue.severity}`}>
            {issue.message}
          </p>
        ))}
      {rows.flatMap((_, index) =>
        rowIssue(index).map((issue) => (
          <p
            key={`${index}-${issue.code}`}
            className={`field__issue field__issue--${issue.severity}`}
          >
            {issue.message}
          </p>
        )),
      )}
    </fieldset>
  );
}
