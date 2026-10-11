import type { ResistorStock } from "../types";

interface InventoryPickerProps {
  legend: string;
  stock: ResistorStock[];
  selected: number[];
  onChange: (selected: number[]) => void;
  maxPerStage: number;
}

export function InventoryPicker({
  legend,
  stock,
  selected,
  onChange,
  maxPerStage,
}: InventoryPickerProps) {
  const toggle = (ohms: number) => {
    onChange(
      selected.includes(ohms)
        ? selected.filter((value) => value !== ohms)
        : [...selected, ohms].sort((a, b) => a - b),
    );
  };

  return (
    <fieldset className="inventory">
      <legend className="inventory__legend">
        {legend}
        <span className="inventory__reach">
          up to {maxPerStage.toLocaleString()} &#8486;/stage in series
        </span>
      </legend>
      <div className="inventory__grid">
        {stock.map((item) => {
          const active = selected.includes(item.ohms);
          return (
            <button
              key={item.ohms}
              type="button"
              className={`chip${active ? " chip--active" : ""}`}
              aria-pressed={active}
              onClick={() => toggle(item.ohms)}
            >
              {item.ohms} &#8486;
            </button>
          );
        })}
      </div>
      <p className="inventory__note">
        {selected.length} of {stock.length} selected.
      </p>
    </fieldset>
  );
}
