// src/components/PlatePopupWindow.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { type WellID } from "../types";

type HoverPoint = {
  xPct: number;
  yPct: number;
};

const MAGNIFICATION = 5;
const MAG_BOX_SIZE = 260;

function formatFraction(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)}%`;
}

export function PlatePopupWindow({
  well,
  imageUrl,
  segmentedImageUrl,
  correctedFraction,
  rawFraction,
  onClose,
}: {
  well: WellID;
  imageUrl: string | null;
  segmentedImageUrl: string | null;
  correctedFraction: number | null;
  rawFraction: number | null;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const originalWrapRef = useRef<HTMLDivElement | null>(null);
  const [hoverPoint, setHoverPoint] = useState<HoverPoint | null>(null);
  const [hoverWindowPos, setHoverWindowPos] = useState({ left: 0, top: 0 });

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  function handleImageMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const el = originalWrapRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const xPct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    const yPct = Math.max(0, Math.min(100, (y / rect.height) * 100));

    setHoverPoint({ xPct, yPct });

    const margin = 16;
    const windowWidth = MAG_BOX_SIZE * 2 + 12;
    const windowHeight = MAG_BOX_SIZE + 38;
    let left = e.clientX + margin;
    let top = e.clientY + margin;

    if (left + windowWidth > window.innerWidth - margin) {
      left = e.clientX - windowWidth - margin;
    }

    if (top + windowHeight > window.innerHeight - margin) {
      top = e.clientY - windowHeight - margin;
    }

    setHoverWindowPos({
      left: Math.max(margin, left),
      top: Math.max(margin, top),
    });
  }

  function handleImageMouseLeave() {
    setHoverPoint(null);
  }

  const magnifiedBackgroundStyle = useMemo<React.CSSProperties | null>(() => {
    if (!hoverPoint) return null;

    return {
      backgroundSize: `${MAGNIFICATION * 100}% ${MAGNIFICATION * 100}%`,
      backgroundPosition: `${hoverPoint.xPct}% ${hoverPoint.yPct}%`,
      backgroundRepeat: "no-repeat",
    };
  }, [hoverPoint]);

  return (
    <div className="fixed inset-0 z-40 bg-black/70">
      <div className="flex h-full w-full flex-col bg-white dark:bg-neutral-900">
        <div className="flex items-start justify-between gap-4 border-b p-4 dark:border-neutral-800">
          <div>
            <h2 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {t("PlatePopupWindow.title", { well })}
            </h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              {t("PlatePopupWindow.description")}
            </p>
            <div className="mt-2 text-sm text-neutral-800 dark:text-neutral-200">
              <div>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Frac. pos. corrected:
                </span>{" "}
                <span className="font-medium">
                  {formatFraction(correctedFraction)}
                </span>
              </div>
              <div>
                <span className="text-neutral-500 dark:text-neutral-400">
                  Frac. pos. raw:
                </span>{" "}
                <span className="font-medium">{formatFraction(rawFraction)}</span>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-md border text-sm bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
            aria-label={t("PlatePopupWindow.actions.close")}
          >
            {t("PlatePopupWindow.actions.close")}
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 sm:p-6">
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <div className="text-xs font-medium mb-1 text-neutral-600 dark:text-neutral-300">
                {t("PlatePopupWindow.images.original.title")}
              </div>

              {imageUrl ? (
                <div
                  ref={originalWrapRef}
                  onMouseMove={handleImageMouseMove}
                  onMouseLeave={handleImageMouseLeave}
                  className="relative h-[72vh] w-full overflow-hidden border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700"
                >
                  <img
                    src={imageUrl}
                    alt={t("PlatePopupWindow.images.original.alt", { well })}
                    className="h-full w-full object-contain"
                    draggable={false}
                  />

                  {hoverPoint && (
                    <div
                      className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-black bg-white/40 dark:border-white"
                      style={{
                        left: `${hoverPoint.xPct}%`,
                        top: `${hoverPoint.yPct}%`,
                      }}
                    />
                  )}
                </div>
              ) : (
                <div className="h-[72vh] w-full grid place-items-center text-xs text-neutral-500 border rounded-lg bg-neutral-50 dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700">
                  {t("PlatePopupWindow.images.original.empty")}
                </div>
              )}
            </div>

            <div>
              <div className="text-xs font-medium mb-1 text-neutral-600 dark:text-neutral-300">
                {t("PlatePopupWindow.images.segmented.title")}
              </div>

              {segmentedImageUrl ? (
                <div className="relative h-[72vh] w-full overflow-hidden border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700">
                  <img
                    src={segmentedImageUrl}
                    alt={t("PlatePopupWindow.images.segmented.alt", { well })}
                    className="h-full w-full object-contain"
                    draggable={false}
                  />

                  {hoverPoint && (
                    <div
                      className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-black bg-white/40 dark:border-white"
                      style={{
                        left: `${hoverPoint.xPct}%`,
                        top: `${hoverPoint.yPct}%`,
                      }}
                    />
                  )}
                </div>
              ) : (
                <div className="h-[72vh] w-full grid place-items-center text-xs text-neutral-500 border rounded-lg bg-neutral-50 dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700">
                  {t("PlatePopupWindow.images.segmented.empty")}
                </div>
              )}

              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-neutral-600 dark:text-neutral-300">
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-orange-400 border border-black/10" />
                  <span>{t("PlatePopupWindow.legend.positive")}</span>
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-green-600 border border-black/10" />
                  <span>{t("PlatePopupWindow.legend.negative")}</span>
                </div>
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-blue-500 border border-black/10" />
                  <span>{t("PlatePopupWindow.legend.uncertain")}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {hoverPoint && magnifiedBackgroundStyle && imageUrl && (
        <div
          className="fixed z-50 rounded-xl border shadow-2xl bg-white p-2 dark:bg-neutral-900 dark:border-neutral-700"
          style={{ left: hoverWindowPos.left, top: hoverWindowPos.top }}
        >
          <div className="mb-2 grid grid-cols-2 gap-3 text-xs font-medium text-neutral-600 dark:text-neutral-300">
            <div>{t("PlatePopupWindow.magnifier.original")}</div>
            <div>{t("PlatePopupWindow.magnifier.segmented")}</div>
          </div>

          <div className="flex gap-3">
            <div
              className="border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700"
              style={{
                width: MAG_BOX_SIZE,
                height: MAG_BOX_SIZE,
                ...magnifiedBackgroundStyle,
                backgroundImage: `url("${imageUrl}")`,
              }}
            />

            {segmentedImageUrl ? (
              <div
                className="border rounded-lg bg-white dark:bg-neutral-900 dark:border-neutral-700"
                style={{
                  width: MAG_BOX_SIZE,
                  height: MAG_BOX_SIZE,
                  ...magnifiedBackgroundStyle,
                  backgroundImage: `url("${segmentedImageUrl}")`,
                }}
              />
            ) : (
              <div
                className="grid place-items-center border rounded-lg bg-neutral-50 text-xs text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700"
                style={{ width: MAG_BOX_SIZE, height: MAG_BOX_SIZE }}
              >
                {t("PlatePopupWindow.magnifier.noSegmentation")}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default PlatePopupWindow;
