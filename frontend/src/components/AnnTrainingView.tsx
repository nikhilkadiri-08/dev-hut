import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchAnnTraining,
  type AnnEpoch,
  type AnnTrainingHead,
  type AnnTrainingHistory,
} from "../api";
import { ApiError } from "../api";

const OUTPUT_LABEL: Record<string, string> = {
  front_us: "Front",
  tail_us: "Tail",
  crest_kv: "Crest",
};

function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

function axisDomain(values: number[]): [number, number] {
  const finite = values.filter((value) => Number.isFinite(value));
  const low = finite.length ? Math.min(...finite) : 0;
  const high = finite.length ? Math.max(...finite) : 1;
  const span = high - low;
  const pad = span > 0 ? span * 0.08 : Math.max(Math.abs(high) * 0.05, 1e-6);
  return [low - pad, high + pad];
}

const INPUT_GROUP: Record<string, string> = {
  is_switching: "#3ddc84",
  Test_kV: "#3ddc84",
  Load_C_pF: "#e8a33d",
  Divider_C_pF: "#e8a33d",
  Stray_C_pF: "#e8a33d",
  L_uH: "#e8a33d",
  Efficiency: "#e8a33d",
  Front_R_Stage: "#e8a33d",
  Tail_R_Stage: "#e8a33d",
  Physics_FrontPeak_us: "#8aa394",
  Physics_Tail_us: "#8aa394",
  Physics_Crest_kV: "#8aa394",
};

const OUTPUT_HEADING: Record<string, string> = {
  front_us: "FRONT",
  tail_us: "TAIL",
  crest_kv: "CREST",
};

function forwardPass(head: AnnTrainingHead, point: AnnEpoch): number[][] | null {
  if (!point.weights || !point.biases || !head.example) return null;
  let current = head.example.features.map(
    (value, index) => (value - head.feature_mean[index]) / head.feature_scale[index],
  );
  const activations = [current];
  point.weights.forEach((matrix, layer) => {
    const bias = point.biases?.[layer] ?? [];
    const next = bias.map((offset, outIndex) => {
      let sum = offset;
      current.forEach((input, inIndex) => {
        sum += input * (matrix[inIndex]?.[outIndex] ?? 0);
      });
      return layer < point.weights!.length - 1 ? Math.tanh(sum) : sum;
    });
    current = next;
    activations.push(current);
  });
  return activations;
}

function networkMaxAbs(head: AnnTrainingHead): number {
  let maxAbs = 0;
  head.history.forEach((row) => {
    row.weights?.forEach((matrix) => {
      matrix.forEach((inputs) => {
        inputs.forEach((weight) => {
          maxAbs = Math.max(maxAbs, Math.abs(weight));
        });
      });
    });
  });
  return maxAbs || 1;
}

function NetworkDiagram({
  head,
  iteration,
  point,
}: {
  head: AnnTrainingHead;
  iteration: number;
  point: AnnEpoch | undefined;
}) {
  const maxAbs = useMemo(() => networkMaxAbs(head), [head]);
  const activations = point ? forwardPass(head, point) : null;
  if (!point?.weights || !point.biases || !activations) {
    return <p className="ann-train__note">This iteration has no stored weight snapshot.</p>;
  }
  const sizes = [
    head.architecture.input_size,
    ...head.architecture.hidden_sizes,
    head.architecture.output_size,
  ];
  const titles = [
    `Input · ${head.architecture.input_size}`,
    ...head.architecture.hidden_sizes.map(
      (size, index) => `Hidden ${index + 1} · ${size} ${head.architecture.hidden_activation}`,
    ),
    OUTPUT_HEADING[head.output] ?? head.output,
  ];
  const rowHeight = 15;
  const height = Math.max(...sizes) * rowHeight + 78;
  const columnGap = 168;
  const left = 176;
  const width = left + (sizes.length - 1) * columnGap + 90;
  const radius = Math.max(...sizes) > 20 ? 4.2 : 6;
  const yOf = (index: number, count: number) => {
    const span = (count - 1) * rowHeight;
    return (height - span) / 2 + index * rowHeight;
  };
  const xs = sizes.map((_, index) => left + index * columnGap);
  const logit = activations[activations.length - 1]?.[0] ?? 0;
  const residual = logit * head.target_scale + head.target_mean;
  const kept = iteration === head.best_epoch;

  return (
    <svg
      className="ann-train__net"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${head.impulse} ${OUTPUT_LABEL[head.output] ?? head.output}, iteration ${iteration} of ${head.epochs_executed}`}
    >
      {point.weights.map((matrix, layer) =>
        matrix.map((row, inIndex) =>
          row.map((weight, outIndex) => {
            const scale = weight / maxAbs;
            const magnitude = Math.min(1, Math.abs(scale));
            return (
              <line
                key={`w-${iteration}-${layer}-${inIndex}-${outIndex}`}
                x1={xs[layer] + radius}
                y1={yOf(inIndex, sizes[layer])}
                x2={xs[layer + 1] - radius}
                y2={yOf(outIndex, sizes[layer + 1])}
                stroke={scale >= 0 ? "#7dffb2" : "#ff6b4a"}
                strokeWidth={0.35 + 3.1 * magnitude}
                strokeOpacity={0.22 + 0.78 * magnitude}
              />
            );
          }),
        ),
      )}
      {sizes.map((count, layer) =>
        Array.from({ length: count }, (_, index) => {
          const inputName = layer === 0 ? head.architecture.input_features[index] : undefined;
          const isOutput = layer === sizes.length - 1;
          const activation = activations[layer]?.[index] ?? 0;
          const tone = Math.max(-1, Math.min(1, Math.tanh(activation)));
          const hiddenFill =
            tone >= 0
              ? `rgb(12, ${Math.round(48 + 170 * tone)}, 32)`
              : `rgb(${Math.round(48 + 170 * -tone)}, 28, 24)`;
          return (
            <circle
              key={`n-${iteration}-${layer}-${index}`}
              cx={xs[layer]}
              cy={yOf(index, count)}
              r={isOutput ? radius + 2 : radius}
              fill={
                isOutput
                  ? "#ff4d4d"
                  : layer === 0
                    ? INPUT_GROUP[inputName ?? ""] ?? "#8aa394"
                    : hiddenFill
              }
              stroke={isOutput ? (kept ? "#ffe08a" : "#ffb4a8") : "#3ddc84"}
              strokeWidth={isOutput ? 1.6 : 0.8}
            >
              {layer === 0 ? <title>{head.architecture.input_features[index]}</title> : null}
            </circle>
          );
        }),
      )}
      {titles.map((title, layer) => (
        <text
          key={`t-${layer}`}
          x={xs[layer]}
          y={16}
          textAnchor="middle"
          className="ann-train__node-title"
        >
          {title}
        </text>
      ))}
      {head.architecture.input_features.map((name, index) => (
        <text
          key={name}
          x={xs[0] - radius - 6}
          y={yOf(index, sizes[0]) + 3}
          textAnchor="end"
          className="ann-train__node-detail"
        >
          {name}
        </text>
      ))}
      {point.weights.map((_, layer) => {
        const last = layer === point.weights!.length - 1;
        const hop =
          layer === 0
            ? "input → hidden"
            : last
              ? `hidden → ${OUTPUT_HEADING[head.output] ?? "output"}`
              : "hidden → hidden";
        return (
          <text
            key={`hop-${layer}`}
            x={(xs[layer] + xs[layer + 1]) / 2}
            y={height - 10}
            textAnchor="middle"
            className="ann-train__hop"
          >
            {hop}
          </text>
        );
      })}
      <text
        x={xs[xs.length - 1] + radius + 8}
        y={yOf(0, 1) - 4}
        className="ann-train__node-detail"
      >
        {OUTPUT_HEADING[head.output] ?? "output"} {formatMetric(residual)} {head.units}
      </text>
    </svg>
  );
}

function LossChart({
  rows,
  iteration,
  lossKey,
  maeKey,
  color,
  name,
  zeroMae,
}: {
  rows: AnnEpoch[];
  iteration: number;
  lossKey: "train_loss" | "val_loss";
  maeKey: "train_mae" | "val_mae";
  color: string;
  name: string;
  zeroMae?: number;
}) {
  const yLoss = axisDomain(rows.map((row) => row[lossKey]));
  const yMae = axisDomain(
    rows.map((row) => row[maeKey]).concat(zeroMae === undefined ? [] : [zeroMae]),
  );
  const axis = "#e4ece7";
  const grid = "rgba(255,255,255,0.22)";
  const tooltip = {
    contentStyle: { background: "#141414", border: "1px solid #3a3a3a", borderRadius: 8, color: "#f4f6f4" },
    labelStyle: { color: "#f4f6f4" },
    cursor: { stroke: "#8a8a8a" },
  };
  return (
    <div className="ann-train__charts">
      <div className="ann-train__plot">
        <p className="ann-train__plot-label">{name} loss</p>
        <div className="plot-surface">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={grid} />
              <XAxis
                dataKey="epoch"
                type="number"
                domain={[1, iteration]}
                allowDecimals={false}
                stroke={axis}
                tick={{ fill: axis, fontSize: 11 }}
              />
              <YAxis domain={yLoss} stroke={axis} tick={{ fill: axis, fontSize: 11 }} width={64} />
              <Tooltip {...tooltip} />
              <Line
                type="monotone"
                dataKey={lossKey}
                name={`${name} MSE`}
                stroke={color}
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="ann-train__plot">
        <p className="ann-train__plot-label">{name} MAE</p>
        <div className="plot-surface">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={grid} />
              <XAxis
                dataKey="epoch"
                type="number"
                domain={[1, iteration]}
                allowDecimals={false}
                stroke={axis}
                tick={{ fill: axis, fontSize: 11 }}
              />
              <YAxis domain={yMae} stroke={axis} tick={{ fill: axis, fontSize: 11 }} width={64} />
              <Tooltip {...tooltip} />
              {zeroMae !== undefined && name === "Validation" ? (
                <ReferenceLine
                  y={zeroMae}
                  stroke="#f0b429"
                  strokeDasharray="4 4"
                  label={{ value: "zero residual", fill: "#f0b429", fontSize: 11 }}
                />
              ) : null}
              <Line
                type="monotone"
                dataKey={maeKey}
                name={`${name} MAE`}
                stroke={color}
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export function AnnTrainingView() {
  const [payload, setPayload] = useState<AnnTrainingHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [headIndex, setHeadIndex] = useState(0);
  const [iteration, setIteration] = useState(1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    fetchAnnTraining()
      .then((loaded) => {
        setPayload(loaded);
        const crest = loaded.heads.findIndex(
          (item) => item.impulse === "Lightning" && item.output === "crest_kv",
        );
        if (crest >= 0) setHeadIndex(crest);
        setError(null);
      })
      .catch((caught: unknown) =>
        setError(caught instanceof ApiError ? caught.message : "Could not load ANN training history."),
      );
  }, []);

  const head = payload?.heads[headIndex];
  const total = head?.epochs_executed ?? 0;
  const point = head?.history[iteration - 1];
  const visible = useMemo(
    () => (head ? head.history.slice(0, iteration) : []),
    [head, iteration],
  );

  useEffect(() => {
    if (!playing || !head) return;
    if (iteration >= head.epochs_executed) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setIteration((current) => current + 1), 280);
    return () => window.clearTimeout(timer);
  }, [playing, iteration, head]);

  if (error) {
    return (
      <section className="card card--error">
        <h2 className="card__title">Training history unavailable</h2>
        <p className="card__body">{error}</p>
      </section>
    );
  }
  if (!payload || !head) {
    return <p className="loading">Loading&hellip;</p>;
  }

  return (
    <section className="ann-train">
      <header className="ann-train__header">
        <h2 className="card__title">Training</h2>
        <p className="ann-train__disclaimer">Synthetic training replay.</p>
        <p className="ann-train__meta">
          {head.impulse} · {OUTPUT_LABEL[head.output] ?? head.output}
        </p>
      </header>

      <div className="ann-train__controls">
        <label className="ann-train__field">
          Network
          <select
            value={headIndex}
            onChange={(event) => {
              setHeadIndex(Number(event.target.value));
              setIteration(1);
              setPlaying(false);
            }}
          >
            {payload.heads.map((item, index) => (
              <option key={`${item.impulse}-${item.output}`} value={index}>
                {item.impulse} {OUTPUT_LABEL[item.output] ?? item.output}
                {item.active ? "" : " · not applied"}
              </option>
            ))}
          </select>
        </label>
        <p className="ann-train__step">
          Iteration <strong>{iteration}</strong> of <strong>{total}</strong>
          {iteration === head.best_epoch ? " · kept" : ` · kept at ${head.best_epoch}`}
        </p>
        <div className="ann-train__transport">
          <button
            type="button"
            className="button button--primary"
            onClick={() => {
              if (playing) {
                setPlaying(false);
                return;
              }
              if (iteration >= total) setIteration(1);
              setPlaying(true);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="button button--ghost"
            onClick={() => {
              setPlaying(false);
              setIteration(1);
            }}
          >
            Restart
          </button>
          <label className="ann-train__slider">
            <span className="visually-hidden">Iteration</span>
            <input
              type="range"
              min={1}
              max={Math.max(total, 1)}
              value={Math.min(iteration, Math.max(total, 1))}
              onChange={(event) => {
                setPlaying(false);
                setIteration(Number(event.target.value));
              }}
            />
          </label>
        </div>
      </div>

      <p className="ann-train__status">
        {head.active ? "Applied." : "Not applied."}
        {!head.active && head.output === "crest_kv" ? " Crest residual not added." : ""}
      </p>

      <div className="ann-train__layout">
        <div>
          <NetworkDiagram head={head} iteration={iteration} point={point} />
        </div>
        <div>
          {point ? (
            <dl className="ann-train__readout">
              <div>
                <dt>Train loss</dt>
                <dd>{formatMetric(point.train_loss)}</dd>
              </div>
              <div>
                <dt>Validation loss</dt>
                <dd>{formatMetric(point.val_loss)}</dd>
              </div>
              <div>
                <dt>Train MAE</dt>
                <dd>
                  {formatMetric(point.train_mae)} {head.units}
                </dd>
              </div>
              <div>
                <dt>Validation MAE</dt>
                <dd>
                  {formatMetric(point.val_mae)} {head.units}
                </dd>
              </div>
            </dl>
          ) : null}
          <LossChart
            rows={visible}
            iteration={iteration}
            lossKey="train_loss"
            maeKey="train_mae"
            color="#6cb6ff"
            name="Train"
          />
          <LossChart
            rows={visible}
            iteration={iteration}
            lossKey="val_loss"
            maeKey="val_mae"
            color="#3ddc84"
            name="Validation"
            zeroMae={head.zero_residual_validation_mae}
          />
        </div>
      </div>
    </section>
  );
}
