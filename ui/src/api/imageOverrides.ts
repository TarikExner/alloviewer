import { API_BASE } from "../App";
import type { ManualWellCall, ProcessResponse, WellID } from "../types";

async function readApiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");

  if (!text.trim()) {
    return `Could not update well classification (${response.status}).`;
  }

  try {
    const parsed = JSON.parse(text) as { detail?: unknown };

    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail.trim();
    }

    if (parsed.detail !== undefined) {
      return JSON.stringify(parsed.detail);
    }
  } catch {
    // Plain-text error response.
  }

  return text.trim();
}

export async function setWellClassificationOverride(
  jobId: string,
  wellId: WellID,
  call: ManualWellCall | null,
): Promise<ProcessResponse> {
  if (!jobId?.trim()) {
    throw new Error("The image-analysis request is missing job_id.");
  }

  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/image/wells/${encodeURIComponent(wellId)}/classification`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ call }),
    },
  );

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as ProcessResponse;
}
