import type {
  WellID,
  WellMap,
  ProcessStartResponse,
} from "../types";
import { API_BASE } from "../App";

export type BackendProgress = {
  status: "queued" | "running" | "done" | "error";
  stage?: string;
  done: number;
  total: number;
  current_well: WellID | null;
  done_wells: WellID[];
  error?: string;
  result?: any;
};

export async function runProcess(
  wells: WellMap,
  order: WellID[],
  files: {
    templateFilename: string | null;
    imageFilenames: string[];
    assayType?: "pra" | "crossmatch";
  }
): Promise<ProcessStartResponse> {
  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      layout: { wells },
      image_order: order,
      template_filename: files.templateFilename,
      image_filenames: files.imageFilenames,
      assay_type: "pra",
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
