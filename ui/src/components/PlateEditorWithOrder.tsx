import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ROWS, COLS } from "../plateConfig";
import {
  ALL_WELLS,
  type Row,
  type WellID,
  type WellMap,
  type WellType,
} from "../types";
import { ROLE_STYLES, ROLE_SWATCH } from "../roleStyles";
import { buildOrder, type SnakeOrientation, type StartCorner } from "../imageOrder";

type CellMode = "T" | "B" | "T/B" | "empty";

type Props = {
  wells: WellMap;
  setWells: (next: WellMap) => void;
  onOrderChange: (order: WellID[]) => void;

  flipVertical?: boolean;
  onFlipChange?: (flip: boolean) => void;

  roleOptions?: WellType[];
  buildDefault?: () => WellMap;

  columnModes?: Record<number, CellMode>;
  onColumnModeChange?: (col: number, mode: CellMode) => void;
};

const DEFAULT_ROLE_OPTIONS: WellType[] = ["sample", "positive", "negative", "igm", "empty"];

export default function PlateEditorWithOrder({
  wells,
  setWells,
  onOrderChange,
  flipVertical: flipProp,
  onFlipChange,
  roleOptions = DEFAULT_ROLE_OPTIONS,
  buildDefault,
  columnModes,
  onColumnModeChange,
}: Props) {
  const { t } = useTranslation();

  const [orientation, setOrientation] = useState<SnakeOrientation>("horizontal");
  const [start, setStart] = useState<StartCorner>("tl");

  const isControlled = typeof flipProp === "boolean";
  const [flipUncontrolled, setFlipUncontrolled] = useState(false);
  const flipVertical = isControlled ? (flipProp as boolean) : flipUncontrolled;
  const setFlip = (next: boolean) => {
    if (!isControlled) setFlipUncontrolled(next);
    onFlipChange?.(next);
  };

  const baseOrder = useMemo(() => buildOrder(orientation, start), [orientation, start]);

  const [brush, setBrush] = useState<WellType | null>(null);
  const [isPainting, setIsPainting] = useState(false);
  const [anchor, setAnchor] = useState<WellID | null>(null);

  const [popOpen, setPopOpen] = useState(false);
  const [popWell, setPopWell] = useState<WellID | null>(null);
  const [popPos, setPopPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const popRef = useRef<HTMLDivElement | null>(null);

  const roleLabel = (role: WellType) => t(`PlateEditorWithOrder.roles.${role}`);

  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (!popRef.current) return;
      if (!popRef.current.contains(e.target as Node)) setPopOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPopOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, []);

  function setWell(id: WellID, role: WellType) {
    const next: WellMap = { ...(wells as any) };
    if (role === "empty") delete (next as any)[id];
    else (next as any)[id] = role;
    setWells(next);
  }

  function clearAll() {
    setWells({} as WellMap);
  }

  function resetAll() {
    if (buildDefault) setWells(buildDefault());
    else setWells(Object.fromEntries(ALL_WELLS.map((w) => [w, "sample"])) as WellMap);
  }

  const rowIndex = (id: WellID) => ROWS.indexOf(id[0] as Row);
  const colIndex = (id: WellID) => COLS.indexOf(Number(id.slice(1)) as any);

  function fillRange(a: WellID, b: WellID, role: WellType) {
    const r1 = rowIndex(a), r2 = rowIndex(b);
    const c1 = colIndex(a), c2 = colIndex(b);
    if (r1 < 0 || r2 < 0 || c1 < 0 || c2 < 0) return;
    const rmin = Math.min(r1, r2), rmax = Math.max(r1, r2);
    const cmin = Math.min(c1, c2), cmax = Math.max(c1, c2);
    const next: WellMap = { ...(wells as any) };
    for (let r = rmin; r <= rmax; r++) {
      for (let c = cmin; c <= cmax; c++) {
        const id = `${ROWS[r]}${COLS[c]}` as WellID;
        if (role === "empty") delete (next as any)[id];
        else (next as any)[id] = role;
      }
    }
    setWells(next);
  }

  function onMouseDown(id: WellID, e: React.MouseEvent<HTMLButtonElement>) {
    e.preventDefault();
    if (brush && e.shiftKey && anchor) {
      fillRange(anchor, id, brush);
      return;
    }
    if (brush) {
      setAnchor(id);
      setIsPainting(true);
      setWell(id, brush);
    } else {
      openPopover(id, e.clientX, e.clientY);
    }
  }

  function onMouseEnter(id: WellID) {
    if (!isPainting || !brush) return;
    setWell(id, brush);
  }

  function onMouseUp() {
    setIsPainting(false);
  }

  function openPopover(id: WellID, clientX: number, clientY: number) {
    setPopWell(id);
    setPopPos({ x: clientX, y: clientY });
    setPopOpen(true);
  }

  const gridRef = useRef<HTMLDivElement | null>(null);
  const [cell, setCell] = useState(32);

  useEffect(() => {
    const el = gridRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      const cols = COLS.length;
      const gap = 4;
      const labelCol = 18;
      const totalGaps = gap * (cols + 1);
      const free = Math.max(0, w - labelCol - totalGaps);
      const size = Math.floor(free / cols);
      setCell(Math.max(28, Math.min(size, 120)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const selectedSet = useMemo(() => {
    const s = new Set<WellID>();
    for (const [id, role] of Object.entries(wells as Record<WellID, WellType>)) {
      if (role && role !== "empty") s.add(id as WellID);
    }
    return s;
  }, [wells]);

  const rowMaxCol: number[] = useMemo(() => {
    const maxes = Array(ROWS.length).fill(-1);
    selectedSet.forEach((id) => {
      const r = rowIndex(id);
      const c = colIndex(id);
      if (r >= 0 && c >= 0) maxes[r] = Math.max(maxes[r], c);
    });
    return maxes;
  }, [selectedSet]);

  const filteredOrder = useMemo<WellID[]>(() => {
    const out: WellID[] = [];
    for (const w of baseOrder) {
      const r = rowIndex(w);
      const c = colIndex(w);
      if (r < 0 || c < 0) continue;
      if (rowMaxCol[r] >= 0 && c <= rowMaxCol[r]) out.push(w);
    }
    return out;
  }, [baseOrder, rowMaxCol]);

  useEffect(() => {
    onOrderChange(filteredOrder);
  }, [filteredOrder, onOrderChange]);

  const displayRows = useMemo(
    () => (flipVertical ? [...ROWS].reverse() : ROWS),
    [flipVertical]
  );

  const hasRole = (role: WellType) => roleOptions.includes(role);

  return (
    <div className="relative rounded-2xl border bg-white p-4 dark:bg-neutral-900 dark:border-neutral-800 flex flex-col min-h-0 select-none">
      <div className="mb-3 shrink-0">
        <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
          {t("PlateEditorWithOrder.title")}
        </h3>
        <div className="text-sm text-neutral-700 dark:text-neutral-300">
          {t("PlateEditorWithOrder.help.pickRole")}
        </div>
        <div className="text-sm text-neutral-700 dark:text-neutral-300">
          {t("PlateEditorWithOrder.help.paint")}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1rem_1fr] gap-4 min-h-0 flex-1">
        <div className="min-h-0 flex flex-col">
          <div className="mb-2 flex flex-wrap gap-2 text-sm">
            {hasRole("sample") && (
              <BrushOption
                label={roleLabel("sample")}
                active={brush === "sample"}
                swatch={ROLE_SWATCH.sample}
                onClick={() => {
                  setBrush((cur) => (cur === "sample" ? null : "sample"));
                  setPopOpen(false);
                }}
              />
            )}
            {hasRole("positive") && (
              <BrushOption
                label={roleLabel("positive")}
                active={brush === "positive"}
                swatch={ROLE_SWATCH.positive}
                onClick={() => {
                  setBrush((cur) => (cur === "positive" ? null : "positive"));
                  setPopOpen(false);
                }}
              />
            )}
            {hasRole("negative") && (
              <BrushOption
                label={roleLabel("negative")}
                active={brush === "negative"}
                swatch={ROLE_SWATCH.negative}
                onClick={() => {
                  setBrush((cur) => (cur === "negative" ? null : "negative"));
                  setPopOpen(false);
                }}
              />
            )}
            {hasRole("igm") && (
              <BrushOption
                label={roleLabel("igm")}
                active={brush === "igm"}
                swatch={ROLE_SWATCH.igm}
                onClick={() => {
                  setBrush((cur) => (cur === "igm" ? null : "igm"));
                  setPopOpen(false);
                }}
              />
            )}
            {hasRole("empty") && (
              <BrushOption
                label={roleLabel("empty")}
                active={brush === "empty"}
                swatch={ROLE_SWATCH.empty}
                onClick={() => {
                  setBrush((cur) => (cur === "empty" ? null : "empty"));
                  setPopOpen(false);
                }}
              />
            )}

            {brush && (
              <span className="px-2 py-1 text-xs rounded-md bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300">
                {t("PlateEditorWithOrder.brush.activeBrush", {
                  role: roleLabel(brush),
                })}
              </span>
            )}
          </div>

          <div
            className="flex-1 overflow-auto"
            ref={gridRef}
            onMouseLeave={() => setIsPainting(false)}
            onMouseUp={onMouseUp}
          >
            <div
              className="inline-grid gap-1"
              style={{ gridTemplateColumns: `auto repeat(${COLS.length}, ${cell}px)` }}
            >
              <div />
              {COLS.map((c) => (
                <div
                  key={`head-${c}`}
                  className="text-xs text-center text-neutral-600 dark:text-neutral-400"
                >
                  {String(c)}
                </div>
              ))}

              {displayRows.map((r) => (
                <React.Fragment key={`row-${r}`}>
                  <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">
                    {r}
                  </div>
                  {COLS.map((c) => {
                    const id = `${r}${c}` as WellID;
                    const role = (wells as any)[id] as WellType | undefined;
                    const cls = role ? ROLE_STYLES[role] : ROLE_STYLES.empty;
                    const label = role ? roleLabel(role) : "";

                    return (
                      <button
                        key={id}
                        title={
                          role
                            ? t("PlateEditorWithOrder.well.titleWithRole", {
                                well: id,
                                role: label,
                              })
                            : t("PlateEditorWithOrder.well.title", {
                                well: id,
                              })
                        }
                        aria-label={
                          role
                            ? t("PlateEditorWithOrder.well.titleWithRole", {
                                well: id,
                                role: label,
                              })
                            : t("PlateEditorWithOrder.well.title", {
                                well: id,
                              })
                        }
                        className={[
                          "rounded-md border text-[10px] flex items-center justify-center",
                          cls,
                          "hover:ring-2 hover:ring-offset-0 hover:ring-neutral-300 dark:hover:ring-neutral-600",
                        ].join(" ")}
                        style={{ width: cell, height: cell }}
                        onMouseDown={(e) => onMouseDown(id, e)}
                        onMouseEnter={() => onMouseEnter(id)}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          openPopover(id, e.clientX, e.clientY);
                        }}
                      >
                        {r}
                        {c}
                      </button>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>

            {columnModes && onColumnModeChange && (
              <div
                className="mt-2 inline-grid gap-1"
                style={{ gridTemplateColumns: `auto repeat(${COLS.length}, ${cell}px)` }}
              >
                <div className="text-[11px] text-right pr-1 text-neutral-600 dark:text-neutral-400">
                  {t("PlateEditorWithOrder.columnType.shortLabel")}
                </div>
                {COLS.map((c) => {
                  const val = (columnModes[c] ?? (c === 10 ? "empty" : "T/B")) as CellMode;

                  return (
                    <div key={`colsel-${c}`} className="flex items-center justify-center">
                      <select
                        value={val}
                        onChange={(e) => onColumnModeChange(c, e.target.value as CellMode)}
                        className="text-xs border rounded bg-white dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-200 w-full"
                        style={{ width: cell }}
                        aria-label={t("PlateEditorWithOrder.columnType.ariaLabel", {
                          column: c,
                        })}
                      >
                        <option value="T">
                          {t("PlateEditorWithOrder.columnType.options.t")}
                        </option>
                        <option value="B">
                          {t("PlateEditorWithOrder.columnType.options.b")}
                        </option>
                        <option value="T/B">
                          {t("PlateEditorWithOrder.columnType.options.tb")}
                        </option>
                        <option value="empty">
                          {t("PlateEditorWithOrder.columnType.options.empty")}
                        </option>
                      </select>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="mt-3 flex items-center gap-3 flex-wrap">
            <button
              onClick={clearAll}
              className="px-3 py-1.5 border rounded-lg bg-white hover:bg-neutral-50
                         dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
            >
              {t("PlateEditorWithOrder.actions.clear")}
            </button>
            <button
              onClick={resetAll}
              className="px-3 py-1.5 border rounded-lg bg-white hover:bg-neutral-50
                         dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
              title={t("PlateEditorWithOrder.actions.resetTitle")}
              aria-label={t("PlateEditorWithOrder.actions.resetTitle")}
            >
              {t("PlateEditorWithOrder.actions.reset")}
            </button>

            <label className="ml-auto inline-flex items-center gap-2 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                checked={flipVertical}
                onChange={(e) => setFlip(e.target.checked)}
              />
              {t("PlateEditorWithOrder.actions.flipVertical")}
            </label>
          </div>
        </div>

        <div className="hidden lg:block w-px bg-neutral-200 dark:bg-neutral-800" />

        <div className="min-h-0 flex flex-col">
          <div className="mb-2 shrink-0 text-sm font-medium text-neutral-900 dark:text-neutral-100">
            {t("PlateEditorWithOrder.imageOrder.title")}
          </div>

          <div className="flex-1 rounded-xl border dark:border-neutral-700 p-3 bg-white dark:bg-neutral-900 flex flex-col min-h-0">
            <div className="grid grid-cols-2 gap-3">
              <fieldset className="border rounded-lg p-3 dark:border-neutral-700">
                <legend className="text-sm px-1">
                  {t("PlateEditorWithOrder.imageOrder.orientation.title")}
                </legend>
                <div className="mt-1 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="orient"
                      checked={orientation === "horizontal"}
                      onChange={() => setOrientation("horizontal")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.orientation.horizontalSnake")}
                    </span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="orient"
                      checked={orientation === "vertical"}
                      onChange={() => setOrientation("vertical")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.orientation.verticalSnake")}
                    </span>
                  </label>
                </div>
              </fieldset>

              <fieldset className="border rounded-lg p-3 dark:border-neutral-700">
                <legend className="text-sm px-1">
                  {t("PlateEditorWithOrder.imageOrder.startCorner.title")}
                </legend>
                <div className="mt-1 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="start"
                      checked={start === "tl"}
                      onChange={() => setStart("tl")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.startCorner.topLeft")}
                    </span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="start"
                      checked={start === "tr"}
                      onChange={() => setStart("tr")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.startCorner.topRight")}
                    </span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="start"
                      checked={start === "bl"}
                      onChange={() => setStart("bl")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.startCorner.bottomLeft")}
                    </span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="start"
                      checked={start === "br"}
                      onChange={() => setStart("br")}
                    />
                    <span className="whitespace-nowrap">
                      {t("PlateEditorWithOrder.imageOrder.startCorner.bottomRight")}
                    </span>
                  </label>
                </div>
              </fieldset>
            </div>

            <div className="mt-3 flex-1 overflow-auto">
              <MiniPlate
                order={filteredOrder}
                selected={selectedSet}
                flipVertical={flipVertical}
              />
            </div>
          </div>
        </div>
      </div>

      {popOpen && popWell && (
        <div
          ref={popRef}
          style={{ left: popPos.x, top: popPos.y }}
          className="fixed z-50 bg-white border rounded-lg shadow-md p-1 text-sm
                     dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-200"
        >
          {roleOptions
            .filter((role) => role !== "empty")
            .map((role) => (
              <button
                key={role}
                onClick={() => {
                  setWell(popWell, role);
                  setPopOpen(false);
                }}
                className="px-3 py-1.5 w-full text-left hover:bg-neutral-100 rounded-md
                           dark:hover:bg-neutral-800"
              >
                {roleLabel(role)}
              </button>
            ))}

          {roleOptions.includes("empty") && (
            <>
              <div className="h-px bg-neutral-200 dark:bg-neutral-800 my-1" />
              <button
                onClick={() => {
                  setWell(popWell, "empty");
                  setPopOpen(false);
                }}
                className="px-3 py-1.5 w-full text-left hover:bg-neutral-100 rounded-md
                           dark:hover:bg-neutral-800"
              >
                {roleLabel("empty")}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function BrushOption({
  label,
  active,
  swatch,
  onClick,
}: {
  label: string;
  active: boolean;
  swatch: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        "inline-flex items-center gap-2 px-2.5 py-1 rounded-lg border transition-colors",
        active
          ? "bg-neutral-100 dark:bg-neutral-800 border-neutral-300 dark:border-neutral-700"
          : "bg-white dark:bg-neutral-900 border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800",
      ].join(" ")}
    >
      <span className={["w-3.5 h-3.5 rounded border", swatch].join(" ")} />
      <span className="text-xs">{label}</span>
    </button>
  );
}

function MiniPlate({
  order,
  selected,
  flipVertical = false,
}: {
  order: WellID[];
  selected: Set<WellID>;
  flipVertical?: boolean;
}) {
  const { t } = useTranslation();

  const cs = 10, pad = 6;
  const rows = flipVertical ? [...ROWS].reverse() : ROWS;
  const w = COLS.length * cs + pad * 2;
  const h = rows.length * cs + pad * 2;

  const pos = new Map<WellID, { x: number; y: number }>();
  rows.forEach((r, ri) =>
    COLS.forEach((c, ci) => {
      const id = `${r}${c}` as WellID;
      pos.set(id, { x: pad + ci * cs + cs / 2, y: pad + ri * cs + cs / 2 });
    })
  );

  type Seg = { d: string; dashed: boolean };
  const segs: Seg[] = [];
  let cur: string[] = [];
  let curDashed: boolean | null = null;

  const getP = (well: WellID) => pos.get(well)!;

  for (let i = 0; i < order.length - 1; i++) {
    const a = order[i], b = order[i + 1];
    const pa = getP(a), pb = getP(b);
    const dashed = !selected.has(b);

    const d = `M ${pa.x} ${pa.y} L ${pb.x} ${pb.y}`;
    if (curDashed === null) {
      curDashed = dashed;
      cur.push(d);
    } else if (curDashed === dashed) {
      cur.push(d);
    } else {
      segs.push({ d: cur.join(" "), dashed: curDashed });
      cur = [d];
      curDashed = dashed;
    }
  }

  if (cur.length) segs.push({ d: cur.join(" "), dashed: !!curDashed });

  const firstP = order.length ? pos.get(order[0]) : null;
  const lastP = order.length ? pos.get(order[order.length - 1]) : null;

  return (
    <div className="overflow-auto">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full max-w-xl border rounded-xl bg-white dark:bg-neutral-900 dark:border-neutral-800"
      >
        {rows.map((r, ri) =>
          COLS.map((c, ci) => {
            const x = pad + ci * cs, y = pad + ri * cs;
            const well = `${r}${c}` as WellID;

            return (
              <g key={well}>
                <rect
                  x={x}
                  y={y}
                  width={cs}
                  height={cs}
                  rx={1.5}
                  className="fill-white dark:fill-neutral-900 stroke-neutral-300 dark:stroke-neutral-700"
                  strokeWidth="0.6"
                />
              </g>
            );
          })
        )}

        {segs.map((s, i) => (
          <path
            key={i}
            d={s.d}
            className={
              s.dashed
                ? "stroke-blue-500/60 dark:stroke-sky-400/60"
                : "stroke-blue-500 dark:stroke-sky-400"
            }
            strokeWidth="1.2"
            fill="none"
            strokeDasharray={s.dashed ? "3 3" : undefined}
          />
        ))}

        {firstP && (
          <circle cx={firstP.x} cy={firstP.y} r={1.6} className="fill-emerald-500" />
        )}
        {lastP && (
          <circle cx={lastP.x} cy={lastP.y} r={1.6} className="fill-rose-500" />
        )}
      </svg>

      <div className="flex items-center gap-3 mt-2 text-xs">
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />
          {t("PlateEditorWithOrder.miniPlate.start")}
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-rose-500 inline-block" />
          {t("PlateEditorWithOrder.miniPlate.end")}
        </span>
        <span className="text-neutral-500 dark:text-neutral-400">
          {t("PlateEditorWithOrder.miniPlate.legend")}
        </span>
      </div>
    </div>
  );
}
