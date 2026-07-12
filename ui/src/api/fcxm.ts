import { API_BASE } from "../App";

export type ChannelRole = "Scatter" | "Population Marker" | "IgG";
export type SampleRole = "NC" | "PC" | "SAMPLE";

export type SimPoint = {
  x: number;
  y: number;
  inGate?: boolean;
};

export type PlotSeries = {
  label: string;
  color: string;
  points: SimPoint[];
  n_total: number;
  n_pos: number;
  pos_pct: number;
};

export type LineSeries = {
  label: string;
  color: string;

  values: number[];
  values_raw?: number[];

  n_total: number;
  n_pos: number;
  pos_pct: number;

  filename?: string | null;
  sample_name?: string | null;
  role?: string | null;

  raw_median?: number | null;
  x_label?: string | null;
};

export type GatingPlot = {
  title: string;
  x_label: string;
  y_label: string;
  points: SimPoint[];
};

export type FCXMGateMetrics = {
  label: string;
  n_events: number;

  igg_pos_fraction: number;
  igg_median_raw: number;
  igg_median_t: number;
  igg_median_shift: number;
  igg_median_ratio: number;
  igg_fluorescence_index: number;
  igg_cutoff_t: number;
  igg_nc_median_raw: number;
  igg_pc_median_raw: number | null;
};

export type FCXMResultsResponse = {
  gate_options: string[];
  selected_gate?: string;

  gating_plots: GatingPlot[];
  final_scatter_series: PlotSeries[];
  line_series: LineSeries[];

  cutoff: number;

  selected_file_metrics?: FCXMGateMetrics | null;
  selected_sample_metrics?: FCXMGateMetrics | null;
};

export type FcsPanelResponse = {
  panel_name?: string;
  markers?: string[];
  channels?: string[];
  files_seen?: number;
  example_file?: string | null;
  rows?: PanelRow[];
  [key: string]: unknown;
};

export type PanelRow = {
  channel: string;
  role: string;
  antibody: string;
  population: string;
};

export type FCXMSample = {
  id: string;
  name: string;
  role: SampleRole;
  file_paths: string[];
};

export type FCXMRunRequest = {
  panel_rows: PanelRow[];
  samples: FCXMSample[];
};

export type FCXMRunStartResponse = {
  job_id: string;
};

export type FCXMRunProgressResponse = {
  status: "queued" | "running" | "done" | "error";

  message?: string | null;
  stage?: string | null;
  result?: unknown;

  total_files?: number | null;
  done_files?: number | null;
  current_file?: string | null;
  done_filenames?: string[];

  error?: string | null;
  error_type?: string | null;
  failed_stage?: string | null;
  failed_file?: string | null;
  support_id?: string | null;
};

/*
 * Kept as an alias so existing imports do not break.
 * New code may use FCXMRunProgressResponse directly.
 */
export type FCXMRunProgress = FCXMRunProgressResponse;

export type FCXMResultsRequest = {
  job_id: string;
  fcs_filename: string;
  gate?: string;
  timeoutMs?: number;
};

export type FcsDisplayNameMode = "filename" | "tube_name";

export type FcsDisplayNamesResponse = {
  names: Record<string, string>;
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

type JsonRequestOptions = {
  url: string;
  init?: RequestInit;
  timeoutMs: number;
  timeoutMessage: string;
  fallbackError: string;
};

function formatValidationError(error: FastApiValidationError): string {
  const location = Array.isArray(error.loc)
    ? error.loc
        .filter((part) => part !== "body")
        .map(String)
        .join(".")
    : "";

  const message = error.msg?.trim() || "Invalid request value";

  return location ? `${location}: ${message}` : message;
}

function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail.trim();
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object") {
          return formatValidationError(item as FastApiValidationError);
        }

        return String(item);
      })
      .filter(Boolean);

    return messages.join("\n");
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
  fallbackMessage: string
): Promise<string> {
  const text = await response.text().catch(() => "");

  if (!text.trim()) {
    return `${fallbackMessage} (${response.status})`;
  }

  try {
    const parsed = JSON.parse(text);
    const message = extractApiErrorMessage(parsed);

    if (message) {
      return message;
    }
  } catch {
    // The response was plain text rather than JSON.
  }

  return text.trim() || `${fallbackMessage} (${response.status})`;
}

async function requestJson<T>({
  url,
  init,
  timeoutMs,
  timeoutMessage,
  fallbackError,
}: JsonRequestOptions): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(await readApiError(response, fallbackError));
    }

    return (await response.json()) as T;
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(timeoutMessage);
    }

    if (
      typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError"
    ) {
      throw new Error(timeoutMessage);
    }

    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function extractDownloadFilename(
  response: Response,
  fallback: string
): string {
  const disposition = response.headers.get("Content-Disposition");

  if (!disposition) {
    return fallback;
  }

  const utf8Match = disposition.match(
    /filename\*=UTF-8''([^;\n]+)/i
  );

  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1].replace(/["']/g, ""));
    } catch {
      return utf8Match[1].replace(/["']/g, "");
    }
  }

  const normalMatch = disposition.match(
    /filename\s*=\s*"?([^";\n]+)"?/i
  );

  return normalMatch?.[1]?.trim() || fallback;
}

export async function fetchFcsDisplayNames(params: {
  filenames: string[];
  mode: FcsDisplayNameMode;
  timeoutMs?: number;
}): Promise<FcsDisplayNamesResponse> {
  return requestJson<FcsDisplayNamesResponse>({
    url: `${API_BASE}/api/fcxm/fcs-display-names`,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filenames: params.filenames,
        mode: params.mode,
      }),
    },
    timeoutMs: params.timeoutMs ?? 10_000,
    timeoutMessage: "FCS display-name request timed out.",
    fallbackError: "Failed to resolve FCS display names",
  });
}

export async function extractPanelFromFcs(
  fcsFilenames: string[],
  timeoutMs = 30_000
): Promise<FcsPanelResponse> {
  return requestJson<FcsPanelResponse>({
    url: `${API_BASE}/api/fcs/panel`,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        fcs_filenames: fcsFilenames,
      }),
    },
    timeoutMs,
    timeoutMessage: "FCS panel extraction timed out.",
    fallbackError: "Failed to extract the FCS panel",
  });
}

export async function fetchFCXMResults(
  params: FCXMResultsRequest
): Promise<FCXMResultsResponse> {
  if (!params.job_id) {
    throw new Error("Results request is missing job_id.");
  }

  if (!params.fcs_filename) {
    throw new Error("Results request is missing fcs_filename.");
  }

  return requestJson<FCXMResultsResponse>({
    url: `${API_BASE}/api/fcxm/results`,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        job_id: params.job_id,
        fcs_filename: params.fcs_filename,
        gate: params.gate ?? "",
      }),
    },
    timeoutMs: params.timeoutMs ?? 10_000,
    timeoutMessage: "Results request timed out.",
    fallbackError: "Failed to load FCXM results",
  });
}

export async function runFCXMAnalysis(
  payload: FCXMRunRequest,
  timeoutMs = 30_000
): Promise<FCXMRunStartResponse> {
  return requestJson<FCXMRunStartResponse>({
    url: `${API_BASE}/api/fcxm/run`,
    init: {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    timeoutMs,
    timeoutMessage: "FCXM run request timed out.",
    fallbackError: "Failed to start FCXM analysis",
  });
}

export async function fetchFCXMRunProgress(
  jobId: string,
  timeoutMs = 10_000
): Promise<FCXMRunProgressResponse> {
  if (!jobId) {
    throw new Error("Progress request is missing job_id.");
  }

  return requestJson<FCXMRunProgressResponse>({
    url: `${API_BASE}/api/fcxm/run/${encodeURIComponent(jobId)}`,
    timeoutMs,
    timeoutMessage: "Progress request timed out.",
    fallbackError: "Failed to load FCXM progress",
  });
}

export async function downloadFCXMSummaryPdf(
  jobId: string,
  positivityMetric:
    | "Median Ratio"
    | "Median Shift"
    | "Fluorescence Index"
    | "% pos",
  positivityThreshold: string,
  timeoutMs = 30_000
): Promise<void> {
  if (!jobId) {
    throw new Error("Summary download is missing job_id.");
  }

  const query = new URLSearchParams();

  query.set("positivity_metric", positivityMetric);

  const threshold = positivityThreshold.trim();

  if (threshold) {
    query.set("positivity_threshold", threshold);
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(
      `${API_BASE}/api/fcxm/summary/${encodeURIComponent(jobId)}?${query.toString()}`,
      {
        method: "GET",
        signal: controller.signal,
      }
    );

    if (!response.ok) {
      throw new Error(
        await readApiError(response, "Failed to download FCXM summary")
      );
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    const filename = extractDownloadFilename(
      response,
      `fcxm_summary_${jobId}.pdf`
    );

    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;

    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();

    URL.revokeObjectURL(url);
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Summary download timed out.");
    }

    if (
      typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError"
    ) {
      throw new Error("Summary download timed out.");
    }

    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
