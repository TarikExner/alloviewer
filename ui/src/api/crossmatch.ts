import type { WellID, WellMap, ProcessStartResponse } from "../types";
import { API_BASE } from "../App";

type BackendProgress = {
  status: "queued" | "running" | "done";
  done: number;
  total: number;
  current_well: WellID | null;
  done_wells: WellID[];
  result?: any;
};

export async function runProcess(
  wells: WellMap,
  order: WellID[],
  files: { templateFilename: string | null; imageFilenames: string[] }
): Promise<ProcessStartResponse> {
  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      layout: { wells },
      image_order: order,
      template_filename: files.templateFilename,
      image_filenames: files.imageFilenames,
      assay_type: files.assayType ?? "crossmatch",
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Server error (${res.status})`);
  }
  return (await res.json()) as ProcessStartResponse;
}

export async function fetchProgress(jobId: string): Promise<BackendProgress> {
  const res = await fetch(`${API_BASE}/api/process/${jobId}`);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Server error (${res.status})`);
  }
  return (await res.json()) as BackendProgress;
}

