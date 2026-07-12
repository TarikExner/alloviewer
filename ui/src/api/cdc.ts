import type {
  WellID,
  WellMap,
  ProcessStartResponse,
} from "../types";
import { API_BASE } from "../App";

export type BackendProgress = {
  status: "queued" | "running" | "done" | "error";
  stage?: string | null;

  done: number;
  total: number;
  current_well?: string | null;
  done_wells?: string[];

  result?: unknown;

  error?: string | null;
  error_type?: string | null;
  failed_stage?: string | null;
  failed_well?: string | null;
  support_id?: string | null;
};

export type RunProcessOptions = {
  templateFilename?: string | null;
  imageFilenames: string[];
  assayType?: "pra" | "crossmatch";

  hlaLayoutUploadId?: string | null;
  praPositivityThreshold?: number;
};

export async function runProcess(
  wells: WellMap,
  order: WellID[],
  files: RunProcessOptions
): Promise<ProcessStartResponse> {
  const assayType = files.assayType ?? "pra";

  const body = {
    layout: { wells },
    image_order: order,
    template_filename: files.templateFilename ?? null,
    image_filenames: files.imageFilenames,
    assay_type: assayType,

    hla_layout_upload_id:
      assayType === "pra" ? files.hlaLayoutUploadId ?? null : null,

    pra_positivity_threshold: files.praPositivityThreshold ?? 20.0,
  };

  console.log("CDC process request", body);

  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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

export async function downloadCDCSummaryPdf(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/process/${jobId}/summary.pdf`);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Server error (${res.status})`);
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `cdc_summary_${jobId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();

  window.URL.revokeObjectURL(url);
}
