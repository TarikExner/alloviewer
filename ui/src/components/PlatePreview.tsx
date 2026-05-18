// src/components/PlatePreview.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "../App";
import { type ProcessResponse, type WellID } from "../types";
import { ROWS, COLS } from "../plateConfig";
import PlatePopupWindow from "./PlatePopupWindow";

export function PlatePreview({
  imagesByWell,
  result,
  summary,
  layout,
  flipVertical = false,
  wellStatus,
  progressPercent,
  jobStatus,
  imageScores,
}: {
  imagesByWell: Record<WellID, string | null>;
  result: ProcessResponse | null | any;
  summary?: Record<string, number> | null;
  layout?:
    | {
        wells: Record<
          string,
          {
            combo_id?: string | null;
            race?: string | null;
            loci?: { data: Record<string, string[]> };
          }
        >;
      }
    | null;
  flipVertical?: boolean;
  wellStatus?: Record<WellID, "idle" | "running" | "done">;
  progressPercent?: number;
  jobStatus?: "idle" | "queued" | "running" | "done" | "error";
  imageScores?: Record<string, number>;
}) {
  const gridWrapRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [cell, setCell] = useState(36);
  const clickTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    const el = gridWrapRef.current;
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

  useEffect(() => {
    return () => {
      if (clickTimeoutRef.current !== null) {
        window.clearTimeout(clickTimeoutRef.current);
      }
    };
  }, []);

  const rowsToRender = useMemo(
    () => (flipVertical ? [...ROWS].reverse() : ROWS),
    [flipVertical]
  );

  function fracOf(well: WellID): number | null {
    if (imageScores) {
      const url = imagesByWell[well];

      if (url) {
        const enc = url.substring(url.lastIndexOf("/") + 1);
        const fname = decodeURIComponent(enc);
        const s = imageScores[fname];

        if (typeof s === "number") return s;
      }
    }

    if (!result) return null;

    if (Array.isArray(result?.results)) {
      const r = result.results.find((x: any) => x.well === well);
      return typeof r?.score === "number" ? r.score : null;
    }

    if (result?.wells && result.wells[well]) {
      const w = result.wells[well];
      return typeof w.frac_pos === "number" ? w.frac_pos : null;
    }

    return null;
  }

  const [hoverWell, setHoverWell] = useState<WellID | null>(null);
  const [pinned, setPinned] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number }>({
    left: 0,
    top: 0,
  });

  const [detailWell, setDetailWell] = useState<WellID | null>(null);

  const data = useMemo(() => {
    if (!hoverWell) return null;

    const img = imagesByWell[hoverWell] || null;

    let role: string | null = null;
    const frac = fracOf(hoverWell);
    let status = "ok";

    if (result) {
      if (Array.isArray(result.results)) {
        const r = result.results.find((x: any) => x.well === hoverWell);

        if (r) {
          role = r.role ?? null;
          status = r.status ?? "ok";
        }
      } else if (result.wells && result.wells[hoverWell]) {
        const w = result.wells[hoverWell];
        role = w.role ?? null;
        status = "ok";
      }
    }

    const meta = layout?.wells?.[hoverWell];
    const comboId: string | null = (meta?.combo_id ?? null) || null;
    const race: string | null = (meta?.race ?? null) || null;
    const loci: Record<string, string[]> = meta?.loci?.data ?? {};

    const locusOrder = [
      "A",
      "B",
      "C",
      "Bw4",
      "Bw6",
      "DR",
      "DRB1",
      "DRB3",
      "DRB4",
      "DRB5",
      "DQ",
      "DQA1",
      "DQB1",
      "DP",
      "DPA1",
      "DPB1",
    ];

    const orderedLoci = Object.keys(loci).sort((a, b) => {
      const ia = locusOrder.indexOf(a);
      const ib = locusOrder.indexOf(b);

      if (ia === -1 && ib === -1) return a.localeCompare(b);
      if (ia === -1) return 1;
      if (ib === -1) return -1;

      return ia - ib;
    });

    return { img, role, frac, status, race, comboId, loci, orderedLoci };
  }, [hoverWell, imagesByWell, result, layout, imageScores]);

  const isHoverPositive = !!data && data.frac !== null && data.frac > 20;

  const CARD_W = 520;
  const CARD_H = 340;
  const MARGIN = 8;

  const detailSegmentedImageUrl = useMemo(() => {
    if (!detailWell) return null;
  
    const wellResult = result?.wells?.[detailWell];
    const url = wellResult?.segmented_image_url ?? null;
  
    if (!url) return null;
  
    if (url.startsWith("http://") || url.startsWith("https://")) {
      return url;
    }
  
    return `${API_BASE}${url}`;
  }, [detailWell, result]);

  function clamp(left: number, top: number, cr: DOMRect) {
    const L = Math.max(MARGIN, Math.min(left, cr.width - CARD_W - MARGIN));
    const T = Math.max(MARGIN, Math.min(top, cr.height - CARD_H - MARGIN));

    return { left: L, top: T };
  }

  function computePos(e: React.MouseEvent) {
    const cont = containerRef.current;

    if (!cont) return { left: 0, top: 0 };

    const cr = cont.getBoundingClientRect();

    const cursorX = e.clientX - cr.left;
    const cursorY = e.clientY - cr.top;

    let left = cursorX - CARD_W - 16;
    if (left < MARGIN) left = cursorX + 16;

    const top = cursorY + 16;

    return clamp(left, top, cr);
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!hoverWell || pinned) return;
    setPos(computePos(e));
  }

  function onEnterWell(id: WellID, e: React.MouseEvent<HTMLElement>) {
    if (pinned) return;

    setHoverWell(id);
    setPos(computePos(e));
  }

  function onClickWell(id: WellID, e: React.MouseEvent<HTMLElement>) {
    if (clickTimeoutRef.current !== null) {
      window.clearTimeout(clickTimeoutRef.current);
      clickTimeoutRef.current = null;
    }

    if (e.detail === 2) {
      setPinned(false);
      setHoverWell(null);
      setDetailWell(id);
      return;
    }

    clickTimeoutRef.current = window.setTimeout(() => {
      setPos(computePos(e));
      setHoverWell(id);
      setPinned((p) => !p);
      clickTimeoutRef.current = null;
    }, 200);
  }

  function onLeavePlate(e: React.MouseEvent) {
    if (pinned) return;

    const next = e.relatedTarget as Node | null;

    if (containerRef.current && next && containerRef.current.contains(next)) {
      return;
    }

    setHoverWell(null);
  }

  function onContainerMouseLeave() {
    if (pinned) return;

    setHoverWell(null);
  }

  return (
    <div
      ref={containerRef}
      className="rounded-2xl border bg-white p-4 dark:bg-neutral-900 dark:border-neutral-800 min-h-0 flex flex-col relative"
      onMouseLeave={onContainerMouseLeave}
      style={{ overflow: "visible" }}
    >
      <h3 className="font-medium mb-2 text-neutral-900 dark:text-neutral-100">
        Plate (Preview)
      </h3>

      {typeof progressPercent === "number" &&
        jobStatus &&
        jobStatus !== "idle" && (
          <div
            className="mb-3"
            onMouseEnter={() => {
              if (!pinned) setHoverWell(null);
            }}
          >
            <div className="flex justify-between text-xs mb-1">
              <span className="text-neutral-500 dark:text-neutral-400">
                {jobStatus === "queued" && "Job queued…"}
                {jobStatus === "running" && "Processing wells…"}
                {jobStatus === "done" && "Finished"}
                {jobStatus === "error" && "Error"}
              </span>

              <span className="font-medium text-neutral-700 dark:text-neutral-200">
                {Math.round(progressPercent)}%
              </span>
            </div>

            <div className="h-2 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-500 via-indigo-500 to-emerald-500 transition-[width] duration-300 ease-out"
                style={{
                  width: `${Math.max(0, Math.min(progressPercent, 100))}%`,
                }}
              />
            </div>
          </div>
        )}

      <div
        className="overflow-auto"
        ref={gridWrapRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={onLeavePlate}
      >
        <div
          className="inline-grid gap-1"
          style={{
            gridTemplateColumns: `auto repeat(${COLS.length}, ${cell}px)`,
          }}
        >
          <div />

          {COLS.map((c) => (
            <div
              key={`col-${c}`}
              className="text-xs text-center text-neutral-600 dark:text-neutral-400"
            >
              {String(c)}
            </div>
          ))}

          {rowsToRender.map((r) => (
            <React.Fragment key={`row-${r}`}>
              <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">
                {String(r)}
              </div>

              {COLS.map((c) => {
                const id = `${r}${c}` as WellID;
                const url = imagesByWell[id] || null;
                const fp = fracOf(id);
                const isPositive = fp !== null && fp > 20;

                const status =
                  wellStatus && wellStatus[id] ? wellStatus[id] : "idle";

                const isDoneVisually =
                  status === "done" || (jobStatus === "done" && !!url);

                const opacity = !url ? 1 : isDoneVisually ? 1 : 0.5;

                return (
                  <div
                    key={id}
                    data-well={id}
                    onMouseEnter={(e) => onEnterWell(id, e)}
                    onClick={(e) => onClickWell(id, e)}
                    className={[
                      "relative",
                      "rounded-md overflow-hidden transition-shadow",
                      isPositive
                        ? "border-8 border-rose-500 dark:border-rose-400"
                        : "border border-neutral-300 dark:border-neutral-700",
                      "bg-white dark:bg-neutral-900",
                      "hover:ring-2 hover:ring-blue-300 dark:hover:ring-sky-500",
                    ].join(" ")}
                    style={{
                      width: cell,
                      height: cell,
                      opacity,
                      transition: "opacity 200ms ease-out",
                      ...(url
                        ? {
                            backgroundImage: `url("${url}")`,
                            backgroundSize: "cover",
                            backgroundPosition: "center",
                          }
                        : {
                            background:
                              "linear-gradient(45deg,#f3f4f6 25%,transparent 25%),linear-gradient(-45deg,#f3f4f6 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#f3f4f6 75%),linear-gradient(-45deg,transparent 75%,#f3f4f6 75%)",
                            backgroundSize: "6px 6px",
                            backgroundPosition:
                              "0 0,0 3px,3px -3px,-3px 0",
                          }),
                    }}
                    title={id}
                    aria-label={`Well ${id}${url ? " (has image)" : ""}${
                      isPositive ? " (POSITIVE)" : ""
                    }`}
                  />
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      {hoverWell && data && (
        <div
          className={[
            "absolute z-30 rounded-xl border shadow-lg transition-colors",
            pinned ? "pointer-events-auto" : "pointer-events-none",
            isHoverPositive
              ? "bg-rose-50 border-rose-200 dark:bg-rose-950 dark:border-rose-800"
              : "bg-neutral-50 dark:bg-neutral-800 dark:border-neutral-700",
          ].join(" ")}
          style={{
            left: pos.left,
            top: pos.top,
            width: 520,
            maxWidth: "min(90vw, 520px)",
            maxHeight: "min(75vh, 360px)",
            padding: 12,
            overflow: "auto",
          }}
        >
          <div className="flex items-start gap-3">
            {data.img ? (
              <img
                src={data.img}
                alt={`Well ${hoverWell}`}
                className="w-64 h-64 object-contain border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700"
                draggable={false}
              />
            ) : (
              <div className="w-64 h-64 grid place-items-center text-xs text-neutral-500 border rounded-lg bg-white dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700">
                No image
              </div>
            )}

            <div className="text-sm text-neutral-900 dark:text-neutral-100">
              <div className="font-medium mb-1">
                Well: {hoverWell}{" "}
                {data.frac !== null && data.frac > 20 && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
                    POSITIVE
                  </span>
                )}
              </div>

              {data.role && (
                <div>
                  <span className="text-neutral-600 dark:text-neutral-400">
                    Role:
                  </span>{" "}
                  {data.role}
                </div>
              )}

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  Frac pos:
                </span>{" "}
                {data.frac === null ? "—" : `${Math.round(data.frac)}%`}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  Status:
                </span>{" "}
                {data.status}
              </div>

              <div className="mt-2">
                <span className="text-neutral-600 dark:text-neutral-400">
                  Combo ID:
                </span>{" "}
                {data.comboId ?? "—"}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  Race:
                </span>{" "}
                {data.race ?? "—"}
              </div>

              <div className="mt-2">
                <div className="text-neutral-600 dark:text-neutral-400 mb-1">
                  HLA loci
                </div>

                {data.orderedLoci.length === 0 ? (
                  <div className="text-sm text-neutral-500 dark:text-neutral-400">
                    No loci in layout for this well.
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1.5 max-w-[360px]">
                    {data.orderedLoci.map((locus) => {
                      const alleles = data.loci[locus] || [];
                      const label = `${locus}: ${alleles.join(", ")}`;

                      return (
                        <span
                          key={locus + label}
                          className="inline-flex items-center px-2 py-0.5 rounded-md border text-[11px]
                                     bg-white dark:bg-neutral-900 dark:border-neutral-700"
                          title={label}
                        >
                          {label}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={() => setPinned((p) => !p)}
                  className="px-3 py-1.5 rounded-md border bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
                >
                  {pinned ? "Unpin" : "Pin"}
                </button>

                {!pinned && (
                  <button
                    type="button"
                    onClick={() => setHoverWell(null)}
                    className="px-3 py-1.5 rounded-md border bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
                  >
                    Close
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {detailWell && (
        <PlatePopupWindow
          well={detailWell}
          imageUrl={imagesByWell[detailWell] || null}
          segmentedImageUrl={detailSegmentedImageUrl}
          onClose={() => setDetailWell(null)}
        />
      )}

      <div
        className="mt-4"
        onMouseEnter={() => {
          if (!pinned) setHoverWell(null);
        }}
      >
      </div>
    </div>
  );
}

export default PlatePreview;
