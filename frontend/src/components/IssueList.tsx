import type { Issue, Severity } from "../types";

const HEADINGS: Record<Severity, string> = {
  error: "Blocking problems",
  warning: "Reduced confidence",
  notice: "Worth knowing",
};

export function IssueList({ severity, issues }: { severity: Severity; issues: Issue[] }) {
  if (issues.length === 0) return null;
  return (
    <section className={`issues issues--${severity}`}>
      <h3 className="issues__heading">
        {HEADINGS[severity]}
        <span className="issues__count">{issues.length}</span>
      </h3>
      <ul className="issues__list">
        {issues.map((issue) => (
          <li key={issue.code + (issue.field ?? "")} className="issues__item">
            {issue.field ? <code className="issues__field">{issue.field}</code> : null}
            <p className="issues__message">{issue.message}</p>
            {issue.suggestion ? (
              <p className="issues__suggestion">{issue.suggestion}</p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
