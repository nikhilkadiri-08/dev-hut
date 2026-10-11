import { useEffect, useState, type ReactElement } from "react";
import {
  ApiError,
  fetchMlStatus,
  type MlDiagram,
  type MlResidualReport,
  type MlResidualValue,
  type MlStatus,
} from "../api";

type Impulse = "lightning" | "switching";

const GROUP_COLOR: Record<string, string> = {
  test_request: "#3ddc84",
  circuit: "#e8a33d",
  physics: "#8aa394",
};

function formatSigned(value: number, unit: string): string {
  const digits = unit === "kV" ? 2 : 3;
  const text = `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
  return `${text} ${unit === "us" ? "µs" : unit}`;
}

function formatPlain(value: number, unit: string): string {
  const digits = unit === "kV" ? 2 : 3;
  return `${value.toFixed(digits)} ${unit === "us" ? "µs" : unit}`;
}

function statusLabel(
  phase: "idle" | "running",
  error: string | null,
  architecture: MlStatus | null,
  report: MlResidualReport | null,
): { id: string; text: string } {
  if (error) return { id: "error", text: "Error" };
  if (phase === "running") return { id: "running", text: "Running inference" };
  if (report?.status === "error") return { id: "error", text: "Error" };
  if (report?.status === "unavailable" || architecture?.status === "unavailable") {
    return { id: "unavailable", text: "Model unavailable" };
  }
  if (report?.status === "finished") return { id: "finished", text: "Finished" };
  if (architecture?.status === "ready") return { id: "ready", text: "Ready" };
  if (!architecture) return { id: "running", text: "Running inference" };
  return { id: "unavailable", text: "Model unavailable" };
}

function NetworkSvg({
  diagram,
  report,
  live,
}: {
  diagram: MlDiagram;
  report: MlResidualReport | null;
  live: boolean;
}) {
  const heads = diagram.heads;
  const pitch = 8;
  const groupGap = 28;
  const top = 52;
  const inputX = 136;
  const h1X = 340;
  const hasSecond = heads.some((head) => head.hidden_sizes.length > 1);
  const h2X = h1X + 170;
  const outX = (hasSecond ? h2X : h1X) + 210;

  const h1Groups = heads.map((head, headIndex) => {
    const count = head.hidden_sizes[0] ?? 1;
    const origin =
      top +
      heads.slice(0, headIndex).reduce((sum, item) => sum + (item.hidden_sizes[0] ?? 1) * pitch + groupGap, 0);
    const nodes = Array.from({ length: count }, (_, index) => ({
      x: h1X,
      y: origin + index * pitch,
    }));
    return { head, count, nodes };
  });
  const lastHidden = h1Groups[h1Groups.length - 1]?.nodes.at(-1);
  const firstHidden = h1Groups[0]?.nodes[0];
  const stackTop = firstHidden?.y ?? top;
  const stackBottom = lastHidden?.y ?? top;
  const h2Groups = heads.map((head, index) => {
    const count = head.hidden_sizes[1] ?? 0;
    const host = h1Groups[index]?.nodes ?? [];
    if (count === 0 || host.length === 0) return [];
    const mid = (host[0].y + host[host.length - 1].y) / 2;
    const span = (count - 1) * pitch;
    return Array.from({ length: count }, (_, neuron) => ({
      x: h2X,
      y: mid - span / 2 + neuron * pitch,
    }));
  });
  const outputCenter = (stackTop + stackBottom) / 2;
  const outputs = h1Groups.map((group, index) => ({
    x: outX,
    y: outputCenter + (index - (h1Groups.length - 1) / 2) * 78,
    head: group.head,
  }));
  const inputNodes = diagram.input_features.map((feature, index) => ({
    ...feature,
    x: inputX,
    y:
      diagram.input_features.length === 1
        ? (stackTop + stackBottom) / 2
        : stackTop + (index * (stackBottom - stackTop)) / (diagram.input_features.length - 1),
  }));
  const width = outX + 168;
  const height = stackBottom + 36;
  const hiddenSummary = h1Groups.map((group) => `${group.head.label} ${group.count}`).join(" · ");
  const secondSummary = heads
    .filter((head) => (head.hidden_sizes[1] ?? 0) > 0)
    .map((head) => `${head.label} ${head.hidden_sizes[1]}`)
    .join(" · ");
  const activation = diagram.hidden_activation;

  const featureValue = new Map((report?.features ?? []).map((feature) => [feature.name, feature.value]));
  const residualFor = (key: string): MlResidualValue | null =>
    report?.residuals ? report.residuals[key as keyof NonNullable<MlResidualReport["residuals"]>] : null;

  const lines = h1Groups.flatMap((group, headIndex) => {
    const drawn: ReactElement[] = [];
    group.head.matrices.forEach((matrix, layerIndex) => {
      const sources = layerIndex === 0 ? inputNodes : layerIndex === 1 ? group.nodes : h2Groups[headIndex];
      const targets =
        layerIndex === 0
          ? group.nodes
          : layerIndex < group.head.matrices.length - 1
            ? h2Groups[headIndex]
            : [outputs[headIndex]];
      for (let i = 0; i < matrix.rows; i += 1) {
        for (let j = 0; j < matrix.cols; j += 1) {
          const source = sources[i];
          const target = targets[j];
          const weight = matrix.normalized[i]?.[j];
          if (!source || !target || weight == null) continue;
          const magnitude = Math.min(1, Math.abs(weight));
          drawn.push(
            <line
              key={`${group.head.output}-${layerIndex}-${i}-${j}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={live ? "#8dffc0" : "#1e6a3c"}
              strokeWidth={0.45}
              strokeOpacity={(live ? 0.1 : 0.04) + magnitude * (live ? 0.78 : 0.24)}
            />,
          );
        }
      }
    });
    return drawn;
  });

  return (
    <svg
      className="nn-diagram__svg"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Input ${diagram.input_count}, hidden ${activation}, three residual outputs`}
    >
      <text className="nn-diagram__heading" x={inputX} y={22} textAnchor="middle">
        Input · {diagram.input_count}
      </text>
      <text className="nn-diagram__heading" x={h1X} y={18} textAnchor="middle">
        Hidden 1 · {activation}
      </text>
      <text className="nn-diagram__heading nn-diagram__heading--sub" x={h1X} y={34} textAnchor="middle">
        {hiddenSummary}
      </text>
      {hasSecond ? (
        <>
          <text className="nn-diagram__heading" x={h2X} y={18} textAnchor="middle">
            Hidden 2 · {activation}
          </text>
          <text className="nn-diagram__heading nn-diagram__heading--sub" x={h2X} y={34} textAnchor="middle">
            {secondSummary}
          </text>
        </>
      ) : null}
      <text className="nn-diagram__heading" x={outX} y={22} textAnchor="middle">
        Outputs · 3
      </text>
      {lines}
      {inputNodes.map((node) => (
        <g key={node.name}>
          <circle cx={node.x} cy={node.y} r={4.2} fill={GROUP_COLOR[node.group] ?? "#8aa394"} />
          <text className="nn-diagram__input-label" x={node.x - 10} y={node.y + 3} textAnchor="end">
            {node.label}
          </text>
          <title>
            {node.name}
            {featureValue.has(node.name) ? `: ${featureValue.get(node.name)}` : ""}
          </title>
        </g>
      ))}
      {h1Groups.map((group) => (
        <g key={`${group.head.output}-h1`}>
          <text className="nn-diagram__group" x={group.nodes[0].x - 14} y={group.nodes[0].y - 6} textAnchor="end">
            {group.head.label.toUpperCase()} · {group.count}
          </text>
          {group.nodes.map((node, index) => (
            <circle
              key={`${group.head.output}-h1-${index}`}
              cx={node.x}
              cy={node.y}
              r={3.3}
              fill={live ? "#143222" : "#0c1c14"}
              stroke="#3ddc84"
              strokeOpacity={live ? 0.95 : 0.4}
              strokeWidth={1}
            />
          ))}
        </g>
      ))}
      {h2Groups.map((nodes, headIndex) =>
        nodes.map((node, index) => (
          <circle
            key={`${heads[headIndex].output}-h2-${index}`}
            cx={node.x}
            cy={node.y}
            r={3.3}
            fill={live ? "#143222" : "#0c1c14"}
            stroke="#3ddc84"
            strokeOpacity={live ? 0.95 : 0.4}
            strokeWidth={1}
          />
        )),
      )}
      {outputs.map((output) => {
        const residual = residualFor(output.head.output);
        return (
          <g key={output.head.output}>
            <circle
              cx={output.x}
              cy={output.y}
              r={8}
              fill={live ? "#ff4d4d" : "#6a3030"}
              stroke={residual?.applied ? "#ffe08a" : "#ffb4a8"}
              strokeWidth={residual?.applied ? 2 : 1}
            />
            <text className="nn-diagram__output-label" x={output.x + 16} y={output.y - 8}>
              {output.head.label.toUpperCase()}
            </text>
            <text className="nn-diagram__output-value" x={output.x + 16} y={output.y + 8}>
              {residual ? formatSigned(residual.raw, residual.unit) : "—"}
            </text>
            <text className="nn-diagram__output-note" x={output.x + 16} y={output.y + 22}>
              {residual ? (residual.applied ? "added" : "not added") : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function EquationRow({
  title,
  physics,
  residual,
  corrected,
  unit,
  applied,
}: {
  title: string;
  physics: number;
  residual: MlResidualValue;
  corrected: number;
  unit: string;
  applied: boolean;
}) {
  return (
    <p className="nn-diagram__equation">
      <span>{title}</span> = {formatPlain(physics, unit)} + {formatSigned(residual.added, unit)} ={" "}
      {formatPlain(corrected, unit)}
      {applied ? "" : title === "Crest" ? " · crest residual not added" : " · not added"}
    </p>
  );
}

export function AnnInferenceDiagram({
  impulseType,
  phase,
  report,
  error,
}: {
  impulseType: Impulse;
  phase: "idle" | "running";
  report: MlResidualReport | null;
  error: string | null;
}) {
  const [architecture, setArchitecture] = useState<MlStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setArchitecture(null);
    setLoadError(null);
    fetchMlStatus(impulseType)
      .then((status) => {
        if (!cancelled) setArchitecture(status);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setLoadError(reason instanceof ApiError ? reason.message : "Could not load the ANN.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [impulseType]);

  const shownError = error ?? loadError;
  const badge = statusLabel(phase, shownError, architecture, report);
  const diagram = architecture?.diagram ?? null;
  const live = badge.id === "finished" && report?.residuals != null;
  const physics = report?.physics;
  const corrected = report?.corrected;
  const residuals = report?.residuals;

  return (
    <section className="nn-diagram" aria-label="ANN residual network">
      <div className="nn-diagram__bar">
        <h3 className="nn-diagram__title">Network</h3>
        <span className={`nn-diagram__status nn-diagram__status--${badge.id}`}>{badge.text}</span>
      </div>
      {shownError ? <p className="nn-diagram__error">{shownError}</p> : null}
      {!shownError && report?.status === "error" && report.message ? (
        <p className="nn-diagram__error">{report.message}</p>
      ) : null}
      {diagram ? (
        <div className="nn-diagram__canvas">
          <NetworkSvg diagram={diagram} report={report} live={live} />
        </div>
      ) : (
        <p className="nn-diagram__message">
          {badge.id === "unavailable" ? "Model unavailable." : "Loading…"}
        </p>
      )}
      <ul className="nn-diagram__legend">
        <li><i className="nn-swatch nn-swatch--test" /> Test request</li>
        <li><i className="nn-swatch nn-swatch--circuit" /> Circuit</li>
        <li><i className="nn-swatch nn-swatch--physics" /> Physics</li>
        <li><i className="nn-swatch nn-swatch--output" /> Residual</li>
      </ul>
      {physics && corrected && residuals ? (
        <div className="nn-diagram__equations">
          <EquationRow
            title="Front"
            physics={physics.front_time_us}
            residual={residuals.front_us}
            corrected={corrected.front_time_us}
            unit="us"
            applied={residuals.front_us.applied}
          />
          <EquationRow
            title="Tail"
            physics={physics.tail_time_us}
            residual={residuals.tail_us}
            corrected={corrected.tail_time_us}
            unit="us"
            applied={residuals.tail_us.applied}
          />
          <EquationRow
            title="Crest"
            physics={physics.crest_kv}
            residual={residuals.crest_kv}
            corrected={corrected.crest_kv}
            unit="kV"
            applied={residuals.crest_kv.applied}
          />
        </div>
      ) : (
        <p className="nn-diagram__equation">Run to see Front, Tail, and Crest.</p>
      )}
    </section>
  );
}
