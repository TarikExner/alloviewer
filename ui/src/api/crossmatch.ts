import type {
  CrossmatchColumnModes,
  ProcessStartResponse,
  WellID,
  WellMap,
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
  imageFilenames: string[];
  columnModes: CrossmatchColumnModes;
  flipVertical?: boolean;
};

type FastApiValidationError = {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
};

type FastApiErrorBody = {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
};

function formatValidationError(error: FastApiValidationError): string {
  const location = Array.isArray(error.loc)
    ? error.loc.filter((part) => part !== "body").map(String).join(".")
    : "";
  const message = error.msg?.trim() || "Invalid request value";
  return location ? `${location}: ${message}` : message;
}

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail.trim();

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object") {
          return formatValidationError(item as FastApiValidationError);
        }

        return String(item);
      })
      .filter(Boolean)
      .join("\n");
  }

  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }

  return "";
}

function extractApiErrorMessage(body: unknown): string {
  if (!body || typeof body !== "object") {
    return typeof body === "string" ? body.trim() : "";
  }

  const data = body as FastApiErrorBody;
  const detail = formatErrorDetail(data.detail);
  if (detail) return detail;

  const message = formatErrorDetail(data.message);
  if (message) return message;

  return formatErrorDetail(data.error);
}

async function readApiError(
  response: Response,
  fallbackMessage: string,
): Promise<string> {
  const text = await response.text().catch(() => "");

  if (!text.trim()) {
    return `${fallbackMessage} (${response.status})`;
  }

  try {
    const parsed = JSON.parse(text);
    const message = extractApiErrorMessage(parsed);
    if (message) return message;
  } catch {
    // Plain-text response.
  }

  return text.trim() || `${fallbackMessage} (${response.status})`;
}

function requireJobId(jobId: string): void {
  if (!jobId?.trim()) {
    throw new Error("The crossmatch request is missing job_id.");
  }
}

function extractDownloadFilename(response: Response, fallback: string): string {
  const disposition = response.headers.get("Content-Disposition");
  if (!disposition) return fallback;

  const utf8Match = disposition.match(/filename\*=UTF-8''([^;\n]+)/i);

  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1].replace(/["']/g, ""));
    } catch {
      return utf8Match[1].replace(/["']/g, "");
    }
  }

  const normalMatch = disposition.match(/filename\s*=\s*"?([^";\n]+)"?/i);
  return normalMatch?.[1]?.trim() || fallback;
}

export async function runProcess(
  jobId: string,
  wells: WellMap,
  order: WellID[],
  options: RunProcessOptions,
): Promise<ProcessStartResponse> {
  requireJobId(jobId);

  if (!options.imageFilenames.length) {
    throw new Error("At least one uploaded image is required.");
  }

  const body = {
    layout: { wells },
    image_order: order,
    image_filenames: options.imageFilenames,
    pra_positivity_threshold: 20.0,
    column_modes: options.columnModes,
    flip_vertical: options.flipVertical ?? false,
  };

  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/image/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response, "Could not start crossmatch analysis"),
    );
  }

  return (await response.json()) as ProcessStartResponse;
}

export async function fetchProgress(jobId: string): Promise<BackendProgress> {
  requireJobId(jobId);

  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/image/progress`,
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response, "Could not load crossmatch progress"),
    );
  }

  return (await response.json()) as BackendProgress;
}

export async function downloadCDCSummaryPdf(
  jobId: string,
  flipVertical: boolean,
): Promise<void> {
  requireJobId(jobId);

  const response = await fetch(
    `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/image/summary.pdf?flip_vertical=${flipVertical ? "true" : "false"}`,
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(response, "Could not download crossmatch summary"),
    );
  }

  const blob = await response.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const filename = extractDownloadFilename(
    response,
    `crossmatch_summary_${jobId}.pdf`,
  );

  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;

  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.setTimeout(() => {
    window.URL.revokeObjectURL(objectUrl);
  }, 0);
}

