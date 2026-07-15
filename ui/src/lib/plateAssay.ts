import type { WellID, WellMap } from "../types";

export type WellRunStatus = "idle" | "running" | "done";

type PreloadImageOptions = {
  concurrency?: number;
  signal?: AbortSignal;
};

function encodeJobRelativePath(path: string): string {
  return path
    .replace(/\\/g, "/")
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function finiteNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function computeSummary(proc: any): any | null {
  return proc?.summary ?? null;
}

export function activeWellIds(allWells: WellID[], wells: WellMap): WellID[] {
  return allWells.filter(
    (well) => (wells as any)[well] && (wells as any)[well] !== "empty",
  );
}

export function countWellsByType(
  allWells: WellID[],
  wells: WellMap,
  type: string,
): number {
  return activeWellIds(allWells, wells).filter(
    (well) => (wells as any)[well] === type,
  ).length;
}

export function buildImageOrder(
  scanOrder: WellID[],
  wells: WellMap,
): WellID[] {
  return scanOrder.filter(
    (well) => (wells as any)[well] && (wells as any)[well] !== "empty",
  );
}

export function buildJobThumbnailUrls(
  jobId: string | null,
  filenames: string[],
  apiBase: string,
): string[] {
  if (!jobId) return [];

  return filenames.map(
    (filename) =>
      `${apiBase}/api/jobs/${encodeURIComponent(
        jobId,
      )}/thumbnails/${encodeJobRelativePath(filename)}`,
  );
}

export async function preloadImageUrls(
  urls: string[],
  options: PreloadImageOptions = {},
): Promise<void> {
  const uniqueUrls = Array.from(new Set(urls.filter(Boolean)));
  if (!uniqueUrls.length) return;

  const concurrency = Math.max(
    1,
    Math.min(options.concurrency ?? 4, uniqueUrls.length),
  );
  let index = 0;

  async function worker(): Promise<void> {
    while (index < uniqueUrls.length) {
      if (options.signal?.aborted) return;
      const url = uniqueUrls[index++];

      await new Promise<void>((resolve) => {
        const image = new Image();
        const finish = () => {
          image.onload = null;
          image.onerror = null;
          resolve();
        };

        image.onload = finish;
        image.onerror = finish;
        image.src = url;

        if (image.complete) finish();
      });
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
}

export function buildImagesByWell(
  allWells: WellID[],
  imageOrder: WellID[],
  imageUrls: string[],
): Record<WellID, string | null> {
  const map: Record<WellID, string | null> = Object.create(null);

  allWells.forEach((well) => {
    map[well] = null;
  });

  imageOrder.forEach((well, index) => {
    map[well] = imageUrls[index] || null;
  });

  return map;
}

export function buildInitialWellStatus(
  imageOrder: WellID[],
): Record<WellID, WellRunStatus> {
  const initialStatus: Record<WellID, WellRunStatus> = {} as any;

  imageOrder.forEach((well) => {
    initialStatus[well] = "idle";
  });

  return initialStatus;
}

export function buildWellToFileMap(
  imageOrder: WellID[],
  imageSavedNames: string[],
): Record<WellID, string | null> {
  const wellToFile: Record<WellID, string | null> = {} as any;

  imageOrder.forEach((well, index) => {
    wellToFile[well] = imageSavedNames[index] ?? null;
  });

  return wellToFile;
}

export function extractImageScores(
  result: any,
  wellToFileAtRun: Record<WellID, string | null>,
): Record<string, number> {
  const scores: Record<string, number> = {};

  if (!result?.wells) return scores;

  Object.entries(result.wells).forEach(([wellId, value]) => {
    const well = value as any;
    const filename = wellToFileAtRun[wellId as WellID];
    const corrected =
      finiteNumber(well.frac_pos_corrected) ??
      finiteNumber(well.corrected_frac_pos);

    if (filename && corrected !== null) {
      scores[filename] = corrected;
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
