import type { ReactNode } from "react";
import type { Issue } from "../types";

interface FieldProps {
  label: string;
  unit?: string;
  hint?: ReactNode;
  /** Dotted backend path, used to attach server-side messages to this input. */
  path?: string;
  issues?: Issue[];
  children: ReactNode;
}

export function Field({ label, unit, hint, path, issues = [], children }: FieldProps) {
  const own = path ? issues.filter((issue) => issue.field === path) : [];
  const severity = own.some((i) => i.severity === "error")
    ? "error"
    : own.some((i) => i.severity === "warning")
      ? "warning"
      : undefined;

  return (
    <label className={`field${severity ? ` field--${severity}` : ""}`}>
      <span className="field__label">
        {label}
        {unit ? <span className="field__unit">{unit}</span> : null}
      </span>
      {children}
      {hint ? <span className="field__hint">{hint}</span> : null}
      {own.map((issue) => (
        <span key={issue.code} className={`field__issue field__issue--${issue.severity}`}>
          {issue.message}
          {issue.suggestion ? ` ${issue.suggestion}` : ""}
        </span>
      ))}
    </label>
  );
}

interface NumberFieldProps extends Omit<FieldProps, "children"> {
  value: string;
  onChange: (value: string) => void;
  step?: string;
  placeholder?: string;
}

export function NumberField({ value, onChange, step, placeholder, ...rest }: NumberFieldProps) {
  return (
    <Field {...rest}>
      <input
        className="input"
        type="number"
        inputMode="decimal"
        step={step}
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}

interface TextFieldProps extends Omit<FieldProps, "children"> {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function TextField({ value, onChange, placeholder, ...rest }: TextFieldProps) {
  return (
    <Field {...rest}>
      <input
        className="input"
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </Field>
  );
}
