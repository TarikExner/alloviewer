// src/components/PlatePreview.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { API_BASE } from "../App";
import { COLS, ROWS } from "../plateConfig";
import { type ProcessResponse, type WellID } from "../types";
import PlatePopupWindow from "./PlatePopupWindow";

type WellRunStatus = "idle" | "running" | "done";
type JobRunStatus = "idle" | "queued" | "running" | "done" | "error";

type PlateLayoutMetadata = {
  wells: Record<
    string,
    {
      combo_id?: string | null;
      race?: string | null;
      loci?: { data: Record<string, string[]> };
    }
  >;
};

type PlatePreviewProps = {
  imagesByWell: Record<WellID, string | null>;
  result: ProcessResponse | null | any;
  layout?: PlateLayoutMetadata | null;
  flipVertical?: boolean;
  wellStatus?: Record<WellID, WellRunStatus>;
  progressPercent?: number;
  jobStatus?: JobRunStatus;
  imageScores?: Record<string, number>;
  columnLabels?: Partial<Record<number, string>>;
};

export function PlatePreview({
  imagesByWell,
  result,
  layout,
  flipVertical = false,
  wellStatus,
  progressPercent,
  jobStatus = "idle",
  imageScores,
  columnLabels,
}: PlatePreviewProps) {
  const { t } = useTranslation();
  const gridWrapRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clickTimeoutRef = useRef<number | null>(null);
  const [cell, setCell] = useState(36);
  const [hoverWell, setHoverWell] = useState<WellID | null>(null);
  const [pinned, setPinned] = useState(false);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  const [detailWell, setDetailWell] = useState<WellID | null>(null);

  useEffect(() => {
    const element = gridWrapRef.current;
    if (!element) return;

    const observer = new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      const gap = 4;
      const labelColumn = 18;
      const totalGaps = gap * (COLS.length + 1);
      const freeWidth = Math.max(0, width - labelColumn - totalGaps);
      const size = Math.floor(freeWidth / COLS.length);
      setCell(Math.max(28, Math.min(size, 120)));
    });

    observer.observe(element);
    return () => observer.disconnect();
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
    [flipVertical],
  );

  const preloadUrls = useMemo(
    () =>
      Array.from(
        new Set(
          Object.values(imagesByWell).filter(
            (url): url is string => typeof url === "string" && url.length > 0,
          ),
        ),
      ),
    [imagesByWell],
  );

  function fracOf(well: WellID): number | null {
    if (imageScores) {
      const url = imagesByWell[well];

      if (url) {
        const encodedFilename = url.substring(url.lastIndexOf("/") + 1);
        const filename = decodeURIComponent(encodedFilename);
        const score = imageScores[filename];
        if (typeof score === "number") return score;
      }
    }

    if (!result) return null;

    if (Array.isArray(result?.results)) {
      const wellResult = result.results.find((item: any) => item.well === well);
      return typeof wellResult?.score === "number" ? wellResult.score : null;
    }

    const wellResult = result?.wells?.[well];
    return typeof wellResult?.frac_pos === "number" ? wellResult.frac_pos : null;
  }

  const data = useMemo(() => {
    if (!hoverWell) return null;

    const img = imagesByWell[hoverWell] || null;
    const frac = fracOf(hoverWell);
    let role: string | null = null;
    let status = "ok";

    if (result) {
      if (Array.isArray(result.results)) {
        const wellResult = result.results.find(
          (item: any) => item.well === hoverWell,
        );

        if (wellResult) {
          role = wellResult.role ?? null;
          status = wellResult.status ?? "ok";
        }
      } else if (result.wells?.[hoverWell]) {
        role = result.wells[hoverWell].role ?? null;
      }
    }

    const metadata = layout?.wells?.[hoverWell];
    const comboId = metadata?.combo_id ?? null;
    const race = metadata?.race ?? null;
    const loci = metadata?.loci?.data ?? {};
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
      const indexA = locusOrder.indexOf(a);
      const indexB = locusOrder.indexOf(b);

      if (indexA === -1 && indexB === -1) return a.localeCompare(b);
      if (indexA === -1) return 1;
      if (indexB === -1) return -1;
      return indexA - indexB;
    });

    return { img, role, frac, status, race, comboId, loci, orderedLoci };
  }, [hoverWell, imagesByWell, result, layout, imageScores]);

  const isHoverPositive = !!data && data.frac !== null && data.frac > 20;
  const CARD_W = 520;
  const CARD_H = 340;
  const MARGIN = 8;

  const detailSegmentedImageUrl = useMemo(() => {
    if (!detailWell) return null;

    const url = result?.wells?.[detailWell]?.segmented_image_url ?? null;
    if (!url) return null;

    if (url.startsWith("http://") || url.startsWith("https://")) {
      return url;
    }

    return `${API_BASE}${url}`;
  }, [detailWell, result]);

  function clamp(left: number, top: number, containerRect: DOMRect) {
    return {
      left: Math.max(MARGIN, Math.min(left, containerRect.width - CARD_W - MARGIN)),
      top: Math.max(MARGIN, Math.min(top, containerRect.height - CARD_H - MARGIN)),
    };
  }

  function computePos(event: React.MouseEvent) {
    const container = containerRef.current;
    if (!container) return { left: 0, top: 0 };

    const containerRect = container.getBoundingClientRect();
    const cursorX = event.clientX - containerRect.left;
    const cursorY = event.clientY - containerRect.top;
    let left = cursorX - CARD_W - 16;

    if (left < MARGIN) left = cursorX + 16;

    return clamp(left, cursorY + 16, containerRect);
  }

  function handleMouseMove(event: React.MouseEvent) {
    if (!hoverWell || pinned) return;
    setPos(computePos(event));
  }

  function onEnterWell(id: WellID, event: React.MouseEvent<HTMLElement>) {
    if (pinned) return;
    setHoverWell(id);
    setPos(computePos(event));
  }

  function onClickWell(id: WellID, event: React.MouseEvent<HTMLElement>) {
    if (clickTimeoutRef.current !== null) {
      window.clearTimeout(clickTimeoutRef.current);
      clickTimeoutRef.current = null;
    }

    if (event.detail === 2) {
      setPinned(false);
      setHoverWell(null);
      setDetailWell(id);
      return;
    }

    clickTimeoutRef.current = window.setTimeout(() => {
      setPos(computePos(event));
      setHoverWell(id);
      setPinned((previous) => !previous);
      clickTimeoutRef.current = null;
    }, 200);
  }

  function onLeavePlate(event: React.MouseEvent) {
    if (pinned) return;

    const next = event.relatedTarget as Node | null;

    if (containerRef.current && next && containerRef.current.contains(next)) {
      return;
    }

    setHoverWell(null);
  }

  function onContainerMouseLeave() {
    if (!pinned) setHoverWell(null);
  }

  function progressLabel() {
    if (jobStatus === "queued") return t("PlatePreview.progress.queued");
    if (jobStatus === "running") return t("PlatePreview.progress.running");
    if (jobStatus === "done") return t("PlatePreview.progress.done");
    if (jobStatus === "error") return t("PlatePreview.progress.error");
    return "";
  }

  function wellAriaLabel(id: WellID, hasImage: boolean, isPositive: boolean) {
    return t("PlatePreview.well_aria", {
      well: id,
      imageStatus: hasImage
        ? t("PlatePreview.image_status.has_image")
        : t("PlatePreview.image_status.no_image"),
      positiveStatus: isPositive ? t("PlatePreview.positive") : "",
    });
  }

  const analysisActive = jobStatus === "queued" || jobStatus === "running";

  return (
    <div
      ref={containerRef}
      className="rounded-2xl border bg-white p-4 dark:bg-neutral-900 dark:border-neutral-800 min-h-0 flex flex-col relative"
      onMouseLeave={onContainerMouseLeave}
      style={{ overflow: "visible" }}
    >
      <div
        aria-hidden="true"
        className="absolute h-px w-px overflow-hidden opacity-0 pointer-events-none"
      >
        {preloadUrls.map((url) => (
          <img
            key={url}
            src={url}
            alt=""
            loading="eager"
            decoding="async"
            draggable={false}
          />
        ))}
      </div>

      <h3 className="font-medium mb-2 text-neutral-900 dark:text-neutral-100">
        {t("PlatePreview.title")}
      </h3>

      {typeof progressPercent === "number" && jobStatus !== "idle" && (
        <div
          className="mb-3"
          onMouseEnter={() => {
            if (!pinned) setHoverWell(null);
          }}
        >
          <div className="flex justify-between text-xs mb-1">
            <span className="text-neutral-500 dark:text-neutral-400">
              {progressLabel()}
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
        ref={gridWrapRef}
        className="overflow-auto"
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

          {COLS.map((column) => {
            const rawLabel = columnLabels?.[column];
            const label = rawLabel && rawLabel !== "empty" ? rawLabel : "";

            return (
              <div
                key={`col-${column}`}
                className="min-h-[34px] text-center text-neutral-600 dark:text-neutral-400"
              >
                <div className="text-xs">{column}</div>
                <div
                  className={[
                    "mt-0.5 text-[10px] font-semibold leading-3",
                    label ? "visible" : "invisible",
                  ].join(" ")}
                >
                  {label || "T/B"}
                </div>
              </div>
            );
          })}

          {rowsToRender.map((row) => (
            <React.Fragment key={`row-${row}`}>
              <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">
                {row}
              </div>

              {COLS.map((column) => {
                const id = `${row}${column}` as WellID;
                const url = imagesByWell[id] || null;
                const fraction = fracOf(id);
                const isPositive = fraction !== null && fraction > 20;
                const status = wellStatus?.[id] ?? "idle";
                const isRunning = status === "running";
                const isDone = status === "done" || (jobStatus === "done" && !!url);
                const opacity = !url
                  ? 1
                  : isDone
                    ? 1
                    : isRunning
                      ? 0.72
                      : analysisActive
                        ? 0.35
                        : 1;
                const filter = !url
                  ? undefined
                  : isDone
                    ? "brightness(1.18)"
                    : isRunning
                      ? "brightness(1.05)"
                      : analysisActive
                        ? "brightness(0.75)"
                        : undefined;

                return (
                  <div
                    key={id}
                    data-well={id}
                    onMouseEnter={(event) => onEnterWell(id, event)}
                    onClick={(event) => onClickWell(id, event)}
                    className={[
                      "relative rounded-md overflow-hidden transition-[opacity,filter] duration-300",
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
                      filter,
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
                            backgroundPosition: "0 0,0 3px,3px -3px,-3px 0",
                          }),
                    }}
                    title={id}
                    aria-label={wellAriaLabel(id, !!url, isPositive)}
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
                alt={t("PlatePreview.well_image_alt", { well: hoverWell })}
                className="w-64 h-64 object-contain border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700"
                draggable={false}
              />
            ) : (
              <div className="w-64 h-64 grid place-items-center text-xs text-neutral-500 border rounded-lg bg-white dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700">
                {t("PlatePreview.no_image")}
              </div>
            )}

            <div className="text-sm text-neutral-900 dark:text-neutral-100">
              <div className="font-medium mb-1">
                {t("PlatePreview.fields.well")}: {hoverWell}
                {data.frac !== null && data.frac > 20 && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">
                    {t("PlatePreview.positive")}
                  </span>
                )}
              </div>

              {data.role && (
                <div>
                  <span className="text-neutral-600 dark:text-neutral-400">
                    {t("PlatePreview.fields.role")}:
                  </span>{" "}
                  {data.role}
                </div>
              )}

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.frac_pos")}:
                </span>{" "}
                {data.frac === null
                  ? t("PlatePreview.empty_value")
                  : `${Math.round(data.frac)}%`}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.status")}:
                </span>{" "}
                {data.status}
              </div>

              <div className="mt-2">
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.combo_id")}:
                </span>{" "}
                {data.comboId ?? t("PlatePreview.empty_value")}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.race")}:
                </span>{" "}
                {data.race ?? t("PlatePreview.empty_value")}
              </div>

              <div className="mt-2">
                <div className="text-neutral-600 dark:text-neutral-400 mb-1">
                  {t("PlatePreview.fields.hla_loci")}
                </div>

                {data.orderedLoci.length === 0 ? (
                  <div className="text-sm text-neutral-500 dark:text-neutral-400">
                    {t("PlatePreview.no_loci")}
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1.5 max-w-[360px]">
                    {data.orderedLoci.map((locus) => {
                      const alleles = data.loci[locus] || [];
                      const label = `${locus}: ${alleles.join(", ")}`;

                      return (
                        <span
                          key={`${locus}-${label}`}
                          className="inline-flex items-center px-2 py-0.5 rounded-md border text-[11px] bg-white dark:bg-neutral-900 dark:border-neutral-700"
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
                  onClick={() => setPinned((previous) => !previous)}
                  className="px-3 py-1.5 rounded-md border bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
                >
                  {pinned
                    ? t("PlatePreview.actions.unpin")
                    : t("PlatePreview.actions.pin")}
                </button>

                {!pinned && (
                  <button
                    type="button"
                    onClick={() => setHoverWell(null)}
                    className="px-3 py-1.5 rounded-md border bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
                  >
                    {t("PlatePreview.actions.close")}
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
      />
    </div>
  );
}

export default PlatePreview;
