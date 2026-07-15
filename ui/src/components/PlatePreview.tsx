// src/components/PlatePreview.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { API_BASE } from "../App";
import { COLS, ROWS } from "../plateConfig";
import type {
  ManualWellCall,
  ProcessResponse,
  WellCall,
  WellClassificationOverride,
  WellID,
} from "../types";
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
  onWellOverride?: (
    well: WellID,
    call: ManualWellCall | null,
  ) => Promise<void>;
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
  onWellOverride,
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
  const [overrideBusy, setOverrideBusy] = useState(false);
  const [overrideError, setOverrideError] = useState<string | null>(null);

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

  useEffect(() => {
    setOverrideError(null);
  }, [hoverWell]);

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

  function finiteNumber(value: unknown): number | null {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function correctedFracOf(well: WellID): number | null {
    if (imageScores) {
      const url = imagesByWell[well];

      if (url) {
        const encodedFilename = url.substring(url.lastIndexOf("/") + 1);
        const filename = decodeURIComponent(encodedFilename);
        const score = finiteNumber(imageScores[filename]);
        if (score !== null) return score;
      }
    }

    const wellResult = result?.wells?.[well];
    if (wellResult) {
      return (
        finiteNumber(wellResult.frac_pos_corrected) ??
        finiteNumber(wellResult.corrected_frac_pos)
      );
    }

    return null;
  }

  function rawFracOf(well: WellID): number | null {
    return finiteNumber(result?.wells?.[well]?.frac_pos);
  }


  function wellResultOf(well: WellID): any | null {
    return result?.wells?.[well] ?? null;
  }

  function automatedCallOf(well: WellID): WellCall | null {
    const wellResult = wellResultOf(well);
    const explicit = wellResult?.automated_call;

    if (
      explicit === "positive" ||
      explicit === "negative" ||
      explicit === "borderline" ||
      explicit === "not_available"
    ) {
      return explicit;
    }

    const role = String(wellResult?.role ?? "").toLowerCase();

    if (role === "positive") return "positive";
    if (role === "negative") return "negative";
    if (role !== "sample") return null;

    const corrected = correctedFracOf(well);
    if (corrected === null) return "not_available";

    const threshold = finiteNumber(result?.pra_analysis?.positivity_threshold) ?? 20;
    return corrected >= threshold ? "positive" : "negative";
  }

  function manualOverrideOf(
    well: WellID,
  ): WellClassificationOverride | null {
    const override = wellResultOf(well)?.manual_override;

    if (
      override &&
      (override.call === "positive" || override.call === "negative")
    ) {
      return override as WellClassificationOverride;
    }

    return null;
  }

  function effectiveCallOf(well: WellID): WellCall | null {
    const explicit = wellResultOf(well)?.effective_call;

    if (
      explicit === "positive" ||
      explicit === "negative" ||
      explicit === "borderline" ||
      explicit === "not_available"
    ) {
      return explicit;
    }

    return manualOverrideOf(well)?.call ?? automatedCallOf(well);
  }

  const data = useMemo(() => {
    if (!hoverWell) return null;

    const img = imagesByWell[hoverWell] || null;
    const correctedFrac = correctedFracOf(hoverWell);
    const rawFrac = rawFracOf(hoverWell);
    const automatedCall = automatedCallOf(hoverWell);
    const effectiveCall = effectiveCallOf(hoverWell);
    const manualOverride = manualOverrideOf(hoverWell);
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

    return {
      img,
      role,
      correctedFrac,
      rawFrac,
      automatedCall,
      effectiveCall,
      manualOverride,
      status,
      race,
      comboId,
      loci,
      orderedLoci,
    };
  }, [hoverWell, imagesByWell, result, layout, imageScores]);

  const isHoverPositive = !!data && data.effectiveCall === "positive";
  const canOverrideHoveredWell =
    !!hoverWell &&
    data?.role === "sample" &&
    data.correctedFrac !== null &&
    !!onWellOverride;
  const CARD_W = 520;
  const CARD_H = 460;
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

  async function applyHoverOverride(call: ManualWellCall | null) {
    if (!hoverWell || !onWellOverride || overrideBusy) return;

    setOverrideBusy(true);
    setOverrideError(null);

    try {
      await onWellOverride(hoverWell, call);
    } catch (error: any) {
      setOverrideError(
        error?.message || t("PlatePreview.override.update_error"),
      );
    } finally {
      setOverrideBusy(false);
    }
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
                const isPositive = effectiveCallOf(id) === "positive";
                const manualOverride = manualOverrideOf(id);
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
                  >
                    {manualOverride ? (
                      <span className="absolute right-0.5 top-0.5 grid h-5 min-w-5 place-items-center rounded-full border border-violet-700 bg-violet-600 px-1 text-[10px] font-bold text-white shadow">
                        U
                      </span>
                    ) : null}
                  </div>
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
            maxHeight: "min(82vh, 480px)",
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
                {data.effectiveCall === "positive" && (
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
                  {t("PlatePreview.fields.automated_call")}:
                </span>{" "}
                {data.automatedCall
                  ? t(`PlatePopupWindow.calls.${data.automatedCall}`)
                  : t("PlatePreview.empty_value")}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.effective_call")}:
                </span>{" "}
                {data.effectiveCall
                  ? t(`PlatePopupWindow.calls.${data.effectiveCall}`)
                  : t("PlatePreview.empty_value")}
                {data.manualOverride ? (
                  <span className="ml-2 rounded-full border border-violet-300 bg-violet-50 px-2 py-0.5 text-[10px] font-semibold text-violet-800 dark:border-violet-800 dark:bg-violet-950 dark:text-violet-200">
                    {t("PlatePreview.user_override")}
                  </span>
                ) : null}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  Frac. pos. corrected:
                </span>{" "}
                {data.correctedFrac === null
                  ? t("PlatePreview.empty_value")
                  : `${data.correctedFrac.toFixed(1)}%`}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  Frac. pos. raw:
                </span>{" "}
                {data.rawFrac === null
                  ? t("PlatePreview.empty_value")
                  : `${data.rawFrac.toFixed(1)}%`}
              </div>

              <div>
                <span className="text-neutral-600 dark:text-neutral-400">
                  {t("PlatePreview.fields.status")}:
                </span>{" "}
                {data.status}
              </div>

              {data.role === "sample" && onWellOverride ? (
                <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 p-2.5 dark:border-blue-900 dark:bg-blue-950/40">
                  <div className="text-xs text-blue-800 dark:text-blue-200">
                    {pinned
                      ? t("PlatePreview.override.pinned_hint")
                      : t("PlatePreview.override.hint")}
                  </div>

                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => void applyHoverOverride("negative")}
                      disabled={
                        !pinned ||
                        !canOverrideHoveredWell ||
                        overrideBusy
                      }
                      className={[
                        "w-full rounded-md border px-2 py-1.5 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-60",
                        data.manualOverride?.call === "negative"
                          ? "border-emerald-700 bg-emerald-700 text-white"
                          : "border-emerald-300 bg-white text-emerald-800 hover:bg-emerald-50 dark:border-emerald-800 dark:bg-neutral-900 dark:text-emerald-200 dark:hover:bg-emerald-950",
                      ].join(" ")}
                    >
                      {t("PlatePopupWindow.actions.declare_negative")}
                    </button>

                    <button
                      type="button"
                      onClick={() => void applyHoverOverride("positive")}
                      disabled={
                        !pinned ||
                        !canOverrideHoveredWell ||
                        overrideBusy
                      }
                      className={[
                        "w-full rounded-md border px-2 py-1.5 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-60",
                        data.manualOverride?.call === "positive"
                          ? "border-rose-700 bg-rose-700 text-white"
                          : "border-rose-300 bg-white text-rose-800 hover:bg-rose-50 dark:border-rose-800 dark:bg-neutral-900 dark:text-rose-200 dark:hover:bg-rose-950",
                      ].join(" ")}
                    >
                      {t("PlatePopupWindow.actions.declare_positive")}
                    </button>
                  </div>

                  {data.manualOverride ? (
                    <button
                      type="button"
                      onClick={() => void applyHoverOverride(null)}
                      disabled={!pinned || overrideBusy}
                      className="mt-2 w-full rounded-md border bg-white px-2 py-1 text-xs font-medium hover:bg-neutral-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-neutral-700 dark:bg-neutral-900 dark:hover:bg-neutral-800"
                    >
                      {t("PlatePopupWindow.actions.restore_automated")}
                    </button>
                  ) : null}

                  {pinned && !canOverrideHoveredWell ? (
                    <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                      {t("PlatePreview.override.unavailable")}
                    </div>
                  ) : null}

                  {overrideBusy ? (
                    <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
                      {t("PlatePreview.override.updating")}
                    </div>
                  ) : null}

                  {overrideError ? (
                    <div className="mt-2 text-xs text-red-700 dark:text-red-400">
                      {overrideError}
                    </div>
                  ) : null}
                </div>
              ) : null}

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
          correctedFraction={correctedFracOf(detailWell)}
          rawFraction={rawFracOf(detailWell)}
          automatedCall={automatedCallOf(detailWell)}
          effectiveCall={effectiveCallOf(detailWell)}
          manualOverride={manualOverrideOf(detailWell)}
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