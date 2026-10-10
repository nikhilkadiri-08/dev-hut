import { useEffect, useState } from "react";
import { lookupLiterature } from "../api";
import type { LiteratureRange, TestObjectType } from "../types";

interface LiteratureHintProps {
  selected: TestObjectType | undefined;
  ranges: LiteratureRange[];
  allowSearch: boolean;
}

export function LiteratureHint({ selected, ranges, allowSearch }: LiteratureHintProps) {
  const kindRanges = ranges.filter((item) => item.kind === selected?.literature_kind);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<LiteratureRange[]>([]);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    setHits([]);
    setNote(null);
    setQuery("");
  }, [selected?.id]);

  const shown = hits.length > 0 ? hits : kindRanges;
  if (!allowSearch && shown.length === 0) {
    return null;
  }

  return (
    <div className="literature">
      <p className="literature__title">Published typical values</p>
      <p className="literature__disclaimer">
        Open-literature ranges for when this object is not in the workbook. They are
        not measurements of the test piece. Use the nameplate if you have one.
      </p>
      {shown.map((item) => (
        <p key={item.id} className="literature__row">
          <strong>
            {item.typical_min === item.typical_max
              ? `${item.typical_min} ${item.unit}`
              : `${item.typical_min}–${item.typical_max} ${item.unit}`}
          </strong>
          {` ${item.label}. ${item.note} `}
          {item.sources[0] ? (
            <a href={item.sources[0].url} target="_blank" rel="noreferrer">
              {item.sources[0].title}
            </a>
          ) : null}
        </p>
      ))}
      {allowSearch ? (
        <form
          className="literature__search"
          onSubmit={(event) => {
            event.preventDefault();
            if (!query.trim()) return;
            lookupLiterature(query.trim())
              .then((result) => {
                setHits(result.ranges);
                setNote(
                  result.matched
                    ? result.insulation_levels.length
                      ? `Standard rated withstand voltages for Um ≈ ${result.insulation_levels[0].um_kv} kV: lightning ${result.insulation_levels[0].standard_lightning_impulse_kv.join(", ")} kV.`
                      : null
                    : "No cited match. Measure the object or add a paper to data/literature/open_references.json.",
                );
              })
              .catch((error: Error) => setNote(error.message));
          }}
        >
          <label className="field">
            <span className="field__label">Look up a new object or voltage class</span>
            <input
              className="input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. 420 kV bushing, switching impulse, 300 kV"
            />
          </label>
          <button type="submit" className="button button--ghost">
            Search literature
          </button>
        </form>
      ) : null}
      {note ? <p className="literature__note">{note}</p> : null}
    </div>
  );
}
