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
import type { CorrectedMetrics, WaveformOverlay, WaveformPayload } from "../api";
import type { ImpulseType } from "../types";

interface WaveformChartProps {
  waveform: WaveformPayload;
  requestedPeakKv: number | null;
  overlay?: WaveformOverlay | null;
  corrected?: WaveformPayload | null;
  requested?: WaveformPayload | null;
  physicsFrontUs?: number | null;
  physicsTailUs?: number | null;
  physicsCrestKv?: number | null;
  correctedMetrics?: CorrectedMetrics | null;
  requestedFrontUs?: number | null;
  requestedTailUs?: number | null;
  impulseType?: ImpulseType;
}

interface PlotPoint {
  time_us: number;
  physics_kv: number | undefined;
  corrected_kv: number | undefined;
  requested_kv: number | undefined;
}

type SeriesKey = "physics_kv" | "corrected_kv" | "requested_kv";

function formatUs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(3)} µs`;
}

function formatKv(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(1)} kV`;
}

function formatTick(value: number, span: number): string {
  if (span < 20) return value.toFixed(2);
  if (span < 200) return value.toFixed(1);
  return value.toFixed(0);
}

function peakOf(points: PlotPoint[], key: SeriesKey): { timeUs: number; value: number } | null {
  let best: { timeUs: number; value: number } | null = null;
  for (const point of points) {
    const value = point[key];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (best === null || value > best.value) {
      best = { timeUs: point.time_us, value };
    }
  }
  return best;
}

function halfAfterPeak(points: PlotPoint[], key: SeriesKey, peak: { timeUs: number; value: number } | null): number | null {
  if (peak === null || peak.value <= 0) return null;
  const level = 0.5 * peak.value;
  for (const point of points) {
    const value = point[key];
    if (typeof value === "number" && point.time_us > peak.timeUs && value <= level) {
      return point.time_us;
    }
  }
  return null;
}

/**
 * Absolute microsecond windows. Not scaled to “N × this curve’s peak”, which
 * would redraw a 250/2500 µs switching impulse as the same silhouette as a
 * 1.2/50 µs lightning impulse.
 *
 * Lightning stays on a short axis: front detail about 0–8 µs, record through
 * the half-value on the order of tens of microseconds.
 * Switching stays on a long axis: front detail at least ~800 µs, record
 * through the half-value on the order of thousands of microseconds.
 */
function timeWindows(
  points: PlotPoint[],
  impulseType: ImpulseType,
): { frontEndUs: number; recordEndUs: number } {
  const last = points[points.length - 1]?.time_us ?? 0;
  const physicsPeak = peakOf(points, "physics_kv");
  const correctedPeak = peakOf(points, "corrected_kv");
  const crestTimes = [physicsPeak?.timeUs, correctedPeak?.timeUs].filter(
    (value): value is number => typeof value === "number" && value > 0,
  );
  const crestTime = crestTimes.length ? Math.max(...crestTimes) : 0;
  const halves = [
    halfAfterPeak(points, "physics_kv", physicsPeak),
    halfAfterPeak(points, "corrected_kv", correctedPeak),
    halfAfterPeak(points, "requested_kv", peakOf(points, "requested_kv")),
  ].filter((value): value is number => typeof value === "number");
  const halfTime = halves.length ? Math.max(...halves) : null;

  if (impulseType === "switching") {
    const recordEndUs = Math.min(
      last,
      Math.max(halfTime ? halfTime * 1.15 : last, Math.min(last, 2500)),
    );
    const frontEndUs = Math.min(
      recordEndUs,
      Math.max(crestTime * 4, Math.min(last, 800)),
    );
    return { frontEndUs, recordEndUs };
  }

  const recordEndUs = Math.min(
    last,
    Math.max(halfTime ? halfTime * 1.25 : last, Math.min(last, 80)),
  );
  const frontEndUs = Math.min(
    recordEndUs,
    Math.max(crestTime * 4, Math.min(last, 8)),
  );
  return { frontEndUs, recordEndUs };
}

function WaveformPanel({
  points,
  xMax,
  yMax,
  crestKv,
  peakTimeUs,
  caption,
}: {
  points: PlotPoint[];
  xMax: number;
  yMax: number;
  crestKv: number;
  peakTimeUs: number | null;
  caption: string;
}) {
  const xMin = Math.min(0, points[0]?.time_us ?? 0);
  const span = Math.max(xMax - xMin, 1e-9);
  const thirtyKv = 0.3 * crestKv;
  const ninetyKv = 0.9 * crestKv;
  const fiftyKv = 0.5 * crestKv;
  const showPeak = peakTimeUs != null && peakTimeUs >= xMin && peakTimeUs <= xMax;

  const axis = "#e4ece7";
  const grid = "rgba(255,255,255,0.22)";
  return (
    <figure className="waveform__panel">
      <figcaption className="waveform__panel-label">{caption}</figcaption>
      <div className="waveform__plot">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points} margin={{ top: 16, right: 12, left: 4, bottom: 20 }}>
            <CartesianGrid stroke={grid} strokeDasharray="3 3" />
            <XAxis
              dataKey="time_us"
              type="number"
              domain={[xMin, xMax]}
              allowDataOverflow
              tickFormatter={(value) => formatTick(Number(value), span)}
              tick={{ fill: axis, fontSize: 11 }}
              stroke={axis}
              label={{ value: "µs", position: "insideBottom", offset: -12, fill: axis, fontSize: 11 }}
            />
            <YAxis
              type="number"
              domain={[0, yMax]}
              tick={{ fill: axis, fontSize: 11 }}
              stroke={axis}
              width={56}
              label={{ value: "kV", angle: -90, position: "insideLeft", fill: axis, fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: "#141414", border: "1px solid #3a3a3a", borderRadius: 8, color: "#f4f6f4" }}
              labelStyle={{ color: "#f4f6f4" }}
              cursor={{ stroke: "#8a8a8a" }}
              formatter={(value, name) =>
                typeof value === "number"
                  ? [`${value.toFixed(2)} kV`, String(name)]
                  : [String(value ?? ""), String(name)]
              }
              labelFormatter={(label) => `${Number(label).toFixed(3)} µs`}
            />
            <ReferenceLine
              y={crestKv}
              stroke="#ffe08a"
              strokeDasharray="4 3"
              label={{ value: "crest", fill: "#ffe08a", fontSize: 11 }}
            />
            <ReferenceLine
              y={ninetyKv}
              stroke="#d5ddd8"
              strokeDasharray="2 4"
              label={{ value: "90%", fill: "#d5ddd8", fontSize: 10 }}
            />
            <ReferenceLine
              y={thirtyKv}
              stroke="#d5ddd8"
              strokeDasharray="2 4"
              label={{ value: "30%", fill: "#d5ddd8", fontSize: 10 }}
            />
            <ReferenceLine
              y={fiftyKv}
              stroke="#f0a050"
              strokeDasharray="2 4"
              label={{ value: "50%", fill: "#f0a050", fontSize: 10 }}
            />
            {showPeak ? (
              <ReferenceLine
                x={peakTimeUs}
                stroke="#ffe08a"
                strokeDasharray="2 6"
                label={{ value: "peak", fill: "#ffe08a", fontSize: 10 }}
              />
            ) : null}
            {points.some((point) => point.requested_kv != null) ? (
              <Line
                type="linear"
                dataKey="requested_kv"
                name="requested"
                stroke="#d5dde6"
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="6 4"
                isAnimationActive={false}
              />
            ) : null}
            <Line
              type="linear"
              dataKey="physics_kv"
              name="physics"
              stroke="#3ddc84"
              dot={false}
              strokeWidth={2}
              isAnimationActive={false}
            />
            {points.some((point) => point.corrected_kv != null) ? (
              <Line
                type="linear"
                dataKey="corrected_kv"
                name="ANN"
                stroke="#6cb6ff"
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function WaveformChart({
  waveform,
  requestedPeakKv,
  overlay,
  physicsFrontUs,
  physicsTailUs,
  physicsCrestKv,
  correctedMetrics,
  requestedFrontUs,
  requestedTailUs,
  impulseType = "lightning",
}: WaveformChartProps) {
  const points: PlotPoint[] = overlay
    ? overlay.time_us.map((timeUs, index) => ({
        time_us: timeUs,
        physics_kv: overlay.physics_kv[index],
        corrected_kv: overlay.corrected_kv ? overlay.corrected_kv[index] : undefined,
        requested_kv: overlay.requested_kv ? overlay.requested_kv[index] : undefined,
      }))
    : waveform.time_us.map((timeUs, index) => ({
        time_us: timeUs,
        physics_kv: waveform.voltage_kv[index],
        corrected_kv: undefined,
        requested_kv: undefined,
      }));
  const crestKv = waveform.metrics.crest_kv;
  const peakTimeUs = waveform.metrics.peak_time_us;
  const physicsFront = physicsFrontUs ?? waveform.metrics.front_time_us;
  const physicsTail = physicsTailUs ?? waveform.metrics.tail_time_us;
  const physicsCrest = physicsCrestKv ?? crestKv;
  const switching = impulseType === "switching";
  const { frontEndUs, recordEndUs } = timeWindows(points, impulseType);
  const yMax = Math.max(
    crestKv,
    requestedPeakKv ?? 0,
    physicsCrest,
    correctedMetrics?.crest_kv ?? 0,
    ...points.flatMap((point) =>
      [point.physics_kv, point.corrected_kv, point.requested_kv].filter(
        (value): value is number => typeof value === "number" && Number.isFinite(value),
      ),
    ),
    1,
  ) * 1.08;
  const frontPoints = points.filter((point) => point.time_us <= frontEndUs + 1e-9);
  const recordPoints = points.filter((point) => point.time_us <= recordEndUs + 1e-9);
  const frontCaption = `Front · 0–${frontEndUs.toFixed(switching ? 0 : 1)} µs`;
  const recordCaption = `Tail · 0–${recordEndUs.toFixed(0)} µs`;

  return (
    <div className="waveform">
      <h4 className="waveform__title">{switching ? "Switching" : "Lightning"}</h4>
      <WaveformPanel
        points={frontPoints.length > 1 ? frontPoints : points}
        xMax={frontEndUs}
        yMax={yMax}
        crestKv={crestKv}
        peakTimeUs={peakTimeUs}
        caption={frontCaption}
      />
      {recordEndUs > frontEndUs * 1.15 ? (
        <WaveformPanel
          points={recordPoints.length > 1 ? recordPoints : points}
          xMax={recordEndUs}
          yMax={yMax}
          crestKv={crestKv}
          peakTimeUs={peakTimeUs}
          caption={recordCaption}
        />
      ) : null}
      <table className="waveform__table">
        <caption className="visually-hidden">Crest, front, and tail</caption>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Requested</th>
            <th>Physics</th>
            <th>ANN</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Crest</td>
            <td>{formatKv(requestedPeakKv)}</td>
            <td>{formatKv(physicsCrest)}</td>
            <td>{correctedMetrics ? formatKv(correctedMetrics.crest_kv) : "not applied"}</td>
          </tr>
          <tr>
            <td>Front</td>
            <td>{formatUs(requestedFrontUs)}</td>
            <td>{formatUs(physicsFront)}</td>
            <td>{correctedMetrics ? formatUs(correctedMetrics.front_time_us) : "not applied"}</td>
          </tr>
          <tr>
            <td>Tail</td>
            <td>{formatUs(requestedTailUs)}</td>
            <td>{formatUs(physicsTail)}</td>
            <td>{correctedMetrics ? formatUs(correctedMetrics.tail_time_us) : "not applied"}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
