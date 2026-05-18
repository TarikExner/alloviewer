// src/components/plots.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Tweakables (top-level)
 */
export const PLOT_POINT_COLOR = "#64748b"; // out-of-gate fill
export const PLOT_GATE_COLOR = "#22c55e"; // in-gate fill

// dot outline
export const PLOT_POINT_STROKE_COLOR = "#0f172a";
export const PLOT_POINT_STROKE_WIDTH = 0.35;

export const PLOT_AXIS_COLOR = "#94a3b8";
export const PLOT_TEXT_COLOR = "#64748b";
export const PLOT_LINE_COLOR = "#64748b";
export const PLOT_LINE_WIDTH = 2;
export const PLOT_CUTOFF_COLOR = "#ef4444";

export type Point = { x: number; y: number; inGate?: boolean };

export type Series2D = {
  label: string;
  color: string;
  points: Point[];
};

export type Series1D = {
  key?: string;
  label: string;
  color: string;
  values: number[];
};

function finiteNumber(x: any): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function niceRange(min: number, max: number) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { min: 0, max: 1 };
  }
  const pad = (max - min) * 0.08;
  return { min: min - pad, max: max + pad };
}

function computeBounds2D(series?: Series2D[], points?: Point[]) {
  const xs: number[] = [];
  const ys: number[] = [];

  if (series?.length) {
    for (const s of series) {
      for (const p of s.points || []) {
        if (finiteNumber(p.x) && finiteNumber(p.y)) {
          xs.push(p.x);
          ys.push(p.y);
        }
      }
    }
  } else {
    for (const p of points || []) {
      if (finiteNumber(p.x) && finiteNumber(p.y)) {
        xs.push(p.x);
        ys.push(p.y);
      }
    }
  }

  if (!xs.length) return { xmin: 0, xmax: 1, ymin: 0, ymax: 1 };

  const xr = niceRange(Math.min(...xs), Math.max(...xs));
  const yr = niceRange(Math.min(...ys), Math.max(...ys));
  return { xmin: xr.min, xmax: xr.max, ymin: yr.min, ymax: yr.max };
}

function clean1D(values: number[]) {
  return (values || []).filter((v) => finiteNumber(v));
}

/**
 * Deterministic shuffle helpers (so the canvas doesn't "sparkle" between rerenders)
 */
function hashString32(str: string) {
  // FNV-1a style 32-bit hash
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledCopy<T>(arr: T[], seed: number) {
  const out = arr.slice();
  const rnd = mulberry32(seed || 1);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

type DrawDot = { px: number; py: number; fill: string; opacity: number; r: number };

/**
 * New: shuffle points (draw order) and draw them.
 * This reduces "blocky" overdraw where one series covers another.
 */
function drawShuffledDots(
  ctx: CanvasRenderingContext2D,
  dots: DrawDot[],
  seed: number,
  pointStrokeWidth: number,
  pointStrokeColor: string
) {
  const list = shuffledCopy(dots, seed);
  for (const d of list) {
    ctx.globalAlpha = d.opacity;

    ctx.beginPath();
    ctx.arc(d.px, d.py, d.r, 0, Math.PI * 2);
    ctx.fillStyle = d.fill;
    ctx.fill();

    ctx.lineWidth = pointStrokeWidth;
    ctx.strokeStyle = pointStrokeColor;
    ctx.stroke();

    ctx.globalAlpha = 1;
  }
}

function useResizeRect<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [rect, setRect] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setRect({ w: Math.max(0, r.width), h: Math.max(0, r.height) });
    });

    ro.observe(el);

    const r = el.getBoundingClientRect();
    setRect({ w: Math.max(0, r.width), h: Math.max(0, r.height) });

    return () => ro.disconnect();
  }, []);

  return { ref, rect };
}

export function PlotCard({
  title,
  subtitle,
  children,
  right,
  className = "",
  scrollable = false,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
  scrollable?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3",
        "min-h-0 flex flex-col",
        className,
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="min-w-0">
          <div className="font-medium">{title}</div>
          {subtitle ? (
            <div className="text-xs text-neutral-600 dark:text-neutral-400 truncate">
              {subtitle}
            </div>
          ) : null}
        </div>
        {right ? <div className="shrink-0">{right}</div> : null}
      </div>

      <div
        className={[
          "flex-1 min-h-0",
          scrollable ? "overflow-auto" : "overflow-hidden",
        ].join(" ")}
      >
        {children}
      </div>
    </div>
  );
}

export function ScatterPlot({
  points,
  series,
  xLabel,
  yLabel,
  xLine,
  yLine,
  height = 220,
  fixedWidth,
  fillParent = false,
  pointColor = PLOT_POINT_COLOR,
  gateColor = PLOT_GATE_COLOR,
  pointStrokeColor = PLOT_POINT_STROKE_COLOR,
  pointStrokeWidth = PLOT_POINT_STROKE_WIDTH,
  onDoubleClick,
  seriesPointRadius = 2.2,
  basePointRadius = 2.1,
  gatePointRadius = 2.3,
}: {
  points?: Point[];
  series?: Series2D[];
  xLabel: string;
  yLabel: string;
  xLine?: number;
  yLine?: number;
  height?: number;
  fixedWidth?: number;
  fillParent?: boolean;
  pointColor?: string;
  gateColor?: string;
  pointStrokeColor?: string;
  pointStrokeWidth?: number;
  onDoubleClick?: () => void;
  seriesPointRadius?: number;
  basePointRadius?: number;
  gatePointRadius?: number;
}) {
  const { ref: wrapRef, rect } = useResizeRect<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const bounds = useMemo(() => computeBounds2D(series, points), [series, points]);

  const boxStyle: React.CSSProperties = {
    height: fillParent ? "100%" : `${height}px`,
    width: fixedWidth ? `${fixedWidth}px` : "100%",
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const cssW = fixedWidth ?? rect.w;
    const cssH = rect.h;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 52;
    const padR = 12;
    const padT = 12;
    const padB = 38;

    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);

    const { xmin, xmax, ymin, ymax } = bounds;
    const dx = xmax - xmin || 1;
    const dy = ymax - ymin || 1;

    const sx = (x: number) => padL + ((x - xmin) / dx) * plotW;
    const sy = (y: number) => padT + (1 - (y - ymin) / dy) * plotH;

    // axes
    ctx.strokeStyle = PLOT_AXIS_COLOR;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.stroke();

    // labels
    ctx.fillStyle = PLOT_TEXT_COLOR;
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

    // x label
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(xLabel, padL + plotW / 2, cssH - 8);

    // y label (vertical)
    ctx.save();
    ctx.translate(14, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // reference lines
    if (finiteNumber(xLine)) {
      const xx = sx(xLine);
      ctx.strokeStyle = PLOT_CUTOFF_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xx, padT);
      ctx.lineTo(xx, padT + plotH);
      ctx.stroke();
    }

    if (finiteNumber(yLine)) {
      const yy = sy(yLine);
      ctx.strokeStyle = PLOT_CUTOFF_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(padL + plotW, yy);
      ctx.stroke();
    }

    // series mode: keep color per series; use inGate for visibility (opacity)
    // NEW: shuffle draw order so colors are better mixed
    if (series?.length) {
      const dots: DrawDot[] = [];

      // build draw list (project first, then shuffle)
      for (const s of series) {
        const fill = s.color || pointColor;
        for (const p of s.points || []) {
          if (!finiteNumber(p.x) || !finiteNumber(p.y)) continue;
          dots.push({
            px: sx(p.x),
            py: sy(p.y),
            fill,
            opacity: p.inGate ? 0.9 : 0.22,
            r: seriesPointRadius,
          });
        }
      }

      // stable seed per input (so it does not flicker between rerenders)
      const seedStr =
        (series.map((s) => `${s.label}|${s.color}|${(s.points || []).length}`).join("||") ||
          "series") + `|${Math.round(xmin * 1000)}|${Math.round(xmax * 1000)}|${Math.round(ymin * 1000)}|${Math.round(ymax * 1000)}`;
      const seed = hashString32(seedStr);

      drawShuffledDots(ctx, dots, seed, pointStrokeWidth, pointStrokeColor);
      return;
    }

    // points mode: use pointColor + gateColor
    const basePts: Point[] = [];
    const gatePts: Point[] = [];
    for (const p of points || []) {
      if (!finiteNumber(p.x) || !finiteNumber(p.y)) continue;
      (p.inGate ? gatePts : basePts).push(p);
    }

    // point helper (with outline) - keep as-is here (gate points still drawn last)
    const drawPoint = (
      px: number,
      py: number,
      fill: string,
      opacity: number,
      r: number
    ) => {
      ctx.globalAlpha = opacity;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();

      ctx.lineWidth = pointStrokeWidth;
      ctx.strokeStyle = pointStrokeColor;
      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    for (const p of basePts) drawPoint(sx(p.x), sy(p.y), pointColor, 0.22, basePointRadius);
    for (const p of gatePts) drawPoint(sx(p.x), sy(p.y), gateColor, 0.9, gatePointRadius);
  }, [
    rect.w,
    rect.h,
    fixedWidth,
    bounds,
    points,
    series,
    xLabel,
    yLabel,
    xLine,
    yLine,
    pointColor,
    gateColor,
    pointStrokeColor,
    pointStrokeWidth,
    seriesPointRadius,
    basePointRadius,
    gatePointRadius,
  ]);

  return (
    <div
      className="w-full h-full flex items-center justify-center"
      style={{ minHeight: fillParent ? "0px" : `${height}px` }}
    >
      <div
        ref={wrapRef}
        className="relative"
        style={boxStyle}
        onDoubleClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onDoubleClick?.();
        }}
      >
        <canvas
          ref={canvasRef}
          className="block w-full h-full select-none"
          onMouseDown={(e) => e.preventDefault()}
        />
      </div>
    </div>
  );
}

function histDensity(values: number[], bins: number, minV: number, maxV: number) {
  const xs = clean1D(values);
  const counts = new Array(bins).fill(0);
  const span = maxV - minV || 1;
  const binW = span / bins;

  for (const v of xs) {
    const t = Math.min(bins - 1, Math.max(0, Math.floor(((v - minV) / span) * bins)));
    counts[t] += 1;
  }

  const n = Math.max(1, xs.length);
  const dens = counts.map(c => c / (n * binW));
  const maxY = Math.max(1e-12, ...dens);
  return { dens, maxY };
}

export function LinePlot({
  values,
  lineColor = PLOT_LINE_COLOR,

  series,
  activeSeriesKey,

  xLabel,
  yLabel,
  cutoff,
  bins = 60,
  height = 220,
  fixedWidth,
  showLegend = true,
}: {
  values?: number[];
  lineColor?: string;
  series?: Series1D[];
  activeSeriesKey?: string | null;
  xLabel: string;
  yLabel?: string;
  cutoff?: number;
  bins?: number;
  height?: number;
  fixedWidth?: number;
  showLegend?: boolean;
}) {  
  const { ref: wrapRef, rect } = useResizeRect<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // normalize input to series list
  const seriesList: Series1D[] = useMemo(() => {
    if (series && series.length) return series;
    const v = values || [];
    return [{ label: "Series", color: lineColor, values: v }];
  }, [series, values, lineColor]);

  const global = useMemo(() => {
    const all = seriesList.flatMap((s) => clean1D(s.values));
    if (!all.length) return { minV: 0, maxV: 1, maxY: 1, per: [] as any[] };

    const minV0 = Math.min(...all);
    const maxV0 = Math.max(...all);
    const xr = niceRange(minV0, maxV0);
    const minV = xr.min;
    const maxV = xr.max;

    const per = seriesList.map((s) => {
      const h = histDensity(s.values, bins, minV, maxV);
      return {
        key: s.key || `${s.label}-${s.color}`,
        label: s.label,
        color: s.color,
        counts: h.dens,
        maxY: h.maxY,
      };
    });

    const maxY = Math.max(
      1e-12,
      ...per.flatMap((p) =>
        p.counts.filter((c: number) => Number.isFinite(c))
      )
    );
    return { minV, maxV, maxY, per };
  }, [seriesList, bins]);

  const boxStyle: React.CSSProperties = {
    height: `${height}px`,
    width: fixedWidth ? `${fixedWidth}px` : "100%",
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const cssW = fixedWidth ?? rect.w;
    const cssH = rect.h;
    if (cssW <= 0 || cssH <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const padL = 52;
    const padR = 12;
    const padT = 12;
    const padB = 38;

    const plotW = Math.max(1, cssW - padL - padR);
    const plotH = Math.max(1, cssH - padT - padB);

    // axes
    ctx.strokeStyle = PLOT_AXIS_COLOR;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.stroke();

    // labels
    ctx.fillStyle = PLOT_TEXT_COLOR;
    ctx.font = "12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";

    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(xLabel, padL + plotW / 2, cssH - 8);

    if (yLabel) {
      ctx.save();
      ctx.translate(14, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    // no data
    const anyData = global.per.some((p) => p.counts.some((c: number) => c > 0));
    if (!anyData) {
      ctx.fillStyle = PLOT_TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("No data", padL + 6, padT + 6);
      return;
    }

    const { minV, maxV, maxY } = global;
    const span = maxV - minV || 1;

    const sx = (x: number) => padL + ((x - minV) / span) * plotW;
    const sy = (y: number) => padT + (1 - y / (maxY || 1)) * plotH;

    // cutoff is an x-value in IgG units (backend sends that)
    if (finiteNumber(cutoff)) {
      const xx = sx(cutoff);
      ctx.strokeStyle = PLOT_CUTOFF_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xx, padT);
      ctx.lineTo(xx, padT + plotH);
      ctx.stroke();
    }

    // bin centers in x units
    const xCenters = new Array(bins).fill(0).map((_, i) => {
      return minV + (i + 0.5) * (span / bins);
    });

    // Draw inactive curves first so the highlighted curve stays visually on top.
    const drawOrder = activeSeriesKey
      ? [
          ...global.per.filter((s) => s.key !== activeSeriesKey),
          ...global.per.filter((s) => s.key === activeSeriesKey),
        ]
      : global.per;

    for (const s of drawOrder) {
      const isActive = !!activeSeriesKey && s.key === activeSeriesKey;
      const isDimmed = !!activeSeriesKey && s.key !== activeSeriesKey;

      ctx.save();
      ctx.globalAlpha = isDimmed ? 0.18 : 1;
      ctx.strokeStyle = s.color || PLOT_LINE_COLOR;
      ctx.lineWidth = isActive ? PLOT_LINE_WIDTH + 2 : PLOT_LINE_WIDTH;
      ctx.beginPath();

      let started = false;
      for (let i = 0; i < bins; i++) {
        const x = sx(xCenters[i]);
        const y = sy(s.counts[i]);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.restore();
    }
  }, [
    rect.w,
    rect.h,
    fixedWidth,
    global,
    xLabel,
    yLabel,
    cutoff,
    bins,
    height,
    activeSeriesKey,
  ]);

  return (
    <div
      className="w-full h-full flex items-center justify-center"
      style={{ minHeight: `${height}px` }}
    >
      <div ref={wrapRef} className="relative" style={boxStyle}>
        {showLegend && seriesList.length > 1 ? (
          <div className="absolute right-2 top-2 z-10 max-w-[52%] max-h-24 overflow-auto rounded-lg border bg-white/90 px-2 py-1 text-[11px] shadow-sm dark:bg-neutral-900/90 dark:border-neutral-700">
            <div className="space-y-1">
              {seriesList.map((s) => (
                <div
                  key={`${s.label}-${s.color}`}
                  className="flex items-center gap-1.5 min-w-0"
                >
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: s.color || PLOT_LINE_COLOR }}
                  />
                  <span className="truncate text-neutral-700 dark:text-neutral-300">
                    {s.label}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
  
        <canvas ref={canvasRef} className="block w-full h-full select-none" />
      </div>
    </div>
  );
}

