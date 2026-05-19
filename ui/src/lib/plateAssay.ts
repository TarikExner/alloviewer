import type { WellID, WellMap } from "../types";

export type WellRunStatus = "idle" | "running" | "done";

export function computeSummary(proc: any): any | null {
  if (!proc) return null;
  return proc.summary ?? null;
}

export function activeWellIds(allWells: WellID[], wells: WellMap): WellID[] {
  return allWells.filter(
    (w) => (wells as any)[w] && (wells as any)[w] !== "empty"
  );
}

export function countWellsByType(
  allWells: WellID[],
  wells: WellMap,
  type: string
): number {
  return activeWellIds(allWells, wells).filter((w) => (wells as any)[w] === type)
    .length;
}

export function buildImageOrder(scanOrder: WellID[], wells: WellMap): WellID[] {
  return scanOrder.filter(
    (w) => (wells as any)[w] && (wells as any)[w] !== "empty"
  );
}

export function buildThumbnailUrls(
  imageSavedNames: string[],
  apiBase: string
): string[] {
  return imageSavedNames.map(
    (name) => `${apiBase}/api/thumbnails/${encodeURIComponent(name)}`
  );
}

export function buildImagesByWell(
  allWells: WellID[],
  imageOrder: WellID[],
  imageUrls: string[]
): Record<WellID, string | null> {
  const map: Record<WellID, string | null> = Object.create(null);

  allWells.forEach((w) => {
    map[w] = null;
  });

  imageOrder.forEach((well, idx) => {
    map[well] = imageUrls[idx] || null;
  });

  return map;
}

export function buildInitialWellStatus(
  imageOrder: WellID[]
): Record<WellID, WellRunStatus> {
  const initialStatus: Record<WellID, WellRunStatus> = {} as any;

  imageOrder.forEach((w) => {
    initialStatus[w] = "idle";
  });

  return initialStatus;
}

export function buildWellToFileMap(
  imageOrder: WellID[],
  imageSavedNames: string[]
): Record<WellID, string | null> {
  const wellToFile: Record<WellID, string | null> = {} as any;

  imageOrder.forEach((well, idx) => {
    wellToFile[well] = imageSavedNames[idx] ?? null;
  });

  return wellToFile;
}

export function extractImageScores(
  result: any,
  wellToFileAtRun: Record<WellID, string | null>
): Record<string, number> {
  const scores: Record<string, number> = {};

  if (!result?.wells) return scores;

  Object.entries(result.wells).forEach(([wellId, w]: [string, any]) => {
    const fname = wellToFileAtRun[wellId as WellID];

    if (fname && typeof w.frac_pos === "number") {
      scores[fname] = w.frac_pos;
    }
  });

  return scores;
}

export function plateStageMessage(stage: string | null): string | null {
  return stage === "segmenting"
    ? "Segmenting wells."
    : stage === "calibrating"
    ? "Calibrating controls."
    : stage === "classifying"
    ? "Classifying ROIs."
    : stage === "saving_previews"
    ? "Saving segmented previews."
    : stage === "finalizing"
    ? "Finalizing summary."
    : null;
}

export function clampPercent(done: number, total: number): number | null {
  if (total <= 0) return null;
  return Math.max(0, Math.min(100, (done / total) * 100));
}
