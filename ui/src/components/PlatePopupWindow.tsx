// src/components/PlatePopupWindow.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { type WellID } from "../types";

type HoverPoint = {
  xPct: number;
  yPct: number;
};

const MAGNIFICATION = 5;
const MAG_BOX_SIZE = 260;

export function PlatePopupWindow({
  well,
  imageUrl,
  segmentedImageUrl,
  onClose,
}: {
  well: WellID;
  imageUrl: string | null;
  segmentedImageUrl: string | null;
  onClose: () => void;
}) {
  const originalWrapRef = useRef<HTMLDivElement | null>(null);
  const [hoverPoint, setHoverPoint] = useState<HoverPoint | null>(null);
  const [hoverWindowPos, setHoverWindowPos] = useState<{
    left: number;
    top: number;
  }>({
    left: 0,
    top: 0,
  });

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
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
              Well {well} details
            </h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400">
              Original image and segmentation result. Hover over the original
              image to inspect both views.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-md border text-sm bg-white hover:bg-neutral-50
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
          >
            Close
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 sm:p-6">
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <div className="text-xs font-medium mb-1 text-neutral-600 dark:text-neutral-300">
                Original image
              </div>

              {imageUrl ? (
                <div
                  ref={originalWrapRef}
                  onMouseMove={handleImageMouseMove}
                  onMouseLeave={handleImageMouseLeave}
                  className="relative h-[72vh] w-full overflow-hidden border rounded-lg bg-white
                             dark:bg-neutral-900 dark:border-neutral-700"
                >
                  <img
                    src={imageUrl}
                    alt={`Well ${well} original`}
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
                <div
                  className="h-[72vh] w-full grid place-items-center text-xs text-neutral-500 border rounded-lg bg-neutral-50
                             dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700"
                >
                  No image for this well
                </div>
              )}
            </div>

            <div>
              <div className="text-xs font-medium mb-1 text-neutral-600 dark:text-neutral-300">
                Segmented image
              </div>
            
              {segmentedImageUrl ? (
                <div
                  className="relative h-[72vh] w-full overflow-hidden border rounded-lg bg-white
                             dark:bg-neutral-900 dark:border-neutral-700"
                >
                  <img
                    src={segmentedImageUrl}
                    alt={`Well ${well} segmented`}
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
                <div
                  className="h-[72vh] w-full grid place-items-center text-xs text-neutral-500 border rounded-lg bg-neutral-50
                             dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700"
                >
                  Segmented image not available yet
                </div>
              )}
            
              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-neutral-600 dark:text-neutral-300">
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-orange-400 border border-black/10" />
                  <span>Positive</span>
                </div>
            
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-green-600 border border-black/10" />
                  <span>Negative</span>
                </div>
            
                <div className="inline-flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-sm bg-blue-500 border border-black/10" />
                  <span>Uncertain</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {hoverPoint && magnifiedBackgroundStyle && imageUrl && (
        <div
          className="fixed z-50 rounded-xl border shadow-2xl bg-white p-2 dark:bg-neutral-900 dark:border-neutral-700"
          style={{
            left: hoverWindowPos.left,
            top: hoverWindowPos.top,
          }}
        >
          <div className="mb-2 grid grid-cols-2 gap-3 text-xs font-medium text-neutral-600 dark:text-neutral-300">
            <div>Original</div>
            <div>Segmented</div>
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
                className="grid place-items-center border rounded-lg bg-neutral-50 text-xs text-neutral-500
                           dark:bg-neutral-900 dark:text-neutral-400 dark:border-neutral-700"
                style={{
                  width: MAG_BOX_SIZE,
                  height: MAG_BOX_SIZE,
                }}
              >
                No segmentation
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default PlatePopupWindow;
