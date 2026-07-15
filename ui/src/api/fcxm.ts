import { API_BASE } from "../App";


export type ChannelRole =
  | "Scatter"
  | "Population Marker"
  | "IgG Marker";

export type SampleRole =
  | "NC"
  | "PC"
  | "SAMPLE";


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


export type FcsPanelWarning = {
  type: "PANEL_MISMATCH";
  message: string;
  files_seen: number;
  common_channels_count: number;
  dropped_channels: string[];
  files: Array<{
    file: string;
    missing_channels: string[];
    extra_channels: string[];
    n_channels: number;
  }>;
  example_file: string | null;
};


export type FcsPanelResponse = {
  panel_name?: string | null;
  rows: PanelRow[];
  files_seen: number;
  example_file?: string | null;
  warning?: FcsPanelWarning | null;
};


export type PanelRow = {
  channel: string;
  role: ChannelRole;
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
  status:
    | "queued"
    | "running"
    | "done"
    | "error";

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
 * Alias retained so existing imports do not need to change.
 */
export type FCXMRunProgress =
  FCXMRunProgressResponse;


export type FCXMResultsRequest = {
  fcs_filename: string;
  gate?: string;
  timeoutMs?: number;
};


export type FcsDisplayNameMode =
  | "filename"
  | "tube_name";


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


function requireJobId(
  jobId: string
): void {
  if (!jobId?.trim()) {
    throw new Error(
      "The FCXM request is missing job_id."
    );
  }
}


function jobFcxmUrl(
  jobId: string,
  path: string
): string {
  requireJobId(jobId);

  const normalizedPath =
    path.replace(/^\/+/, "");

  return (
    `${API_BASE}/api/jobs/` +
    `${encodeURIComponent(jobId)}/fcxm/` +
    normalizedPath
  );
}


function formatValidationError(
  error: FastApiValidationError
): string {
  const location =
    Array.isArray(error.loc)
      ? error.loc
          .filter(
            (part) =>
              part !== "body"
          )
          .map(String)
          .join(".")
      : "";

  const message =
    error.msg?.trim() ||
    "Invalid request value";

  return location
    ? `${location}: ${message}`
    : message;
}


function formatErrorDetail(
  detail: unknown
): string {
  if (
    typeof detail === "string"
  ) {
    return detail.trim();
  }

  if (
    Array.isArray(detail)
  ) {
    return detail
      .map((item) => {
        if (
          item &&
          typeof item ===
            "object"
        ) {
          return formatValidationError(
            item as FastApiValidationError
          );
        }

        return String(item);
      })
      .filter(Boolean)
      .join("\n");
  }

  if (
    detail &&
    typeof detail === "object"
  ) {
    try {
      return JSON.stringify(
        detail
      );
    } catch {
      return String(detail);
    }
  }

  return "";
}


function extractApiErrorMessage(
  body: unknown
): string {
  if (
    !body ||
    typeof body !== "object"
  ) {
    return typeof body ===
      "string"
      ? body.trim()
      : "";
  }

  const data =
    body as FastApiErrorBody;

  const detail =
    formatErrorDetail(
      data.detail
    );

  if (detail) {
    return detail;
  }

  const message =
    formatErrorDetail(
      data.message
    );

  if (message) {
    return message;
  }

  return formatErrorDetail(
    data.error
  );
}


async function readApiError(
  response: Response,
  fallbackMessage: string
): Promise<string> {
  const text =
    await response
      .text()
      .catch(() => "");

  if (!text.trim()) {
    return (
      `${fallbackMessage} ` +
      `(${response.status})`
    );
  }

  try {
    const parsed =
      JSON.parse(text);

    const message =
      extractApiErrorMessage(
        parsed
      );

    if (message) {
      return message;
    }
  } catch {
    // The response was plain text.
  }

  return (
    text.trim() ||
    `${fallbackMessage} (${response.status})`
  );
}


function isAbortError(
  error: unknown
): boolean {
  if (
    error instanceof DOMException
  ) {
    return (
      error.name ===
      "AbortError"
    );
  }

  return Boolean(
    typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError"
  );
}


async function requestJson<T>({
  url,
  init,
  timeoutMs,
  timeoutMessage,
  fallbackError,
}: JsonRequestOptions): Promise<T> {
  const controller =
    new AbortController();

  const timeout =
    window.setTimeout(
      () =>
        controller.abort(),
      timeoutMs
    );

  try {
    const response =
      await fetch(url, {
        ...init,
        signal:
          controller.signal,
      });

    if (!response.ok) {
      throw new Error(
        await readApiError(
          response,
          fallbackError
        )
      );
    }

    return (
      await response.json()
    ) as T;

  } catch (error: unknown) {
    if (
      isAbortError(error)
    ) {
      throw new Error(
        timeoutMessage
      );
    }

    throw error;

  } finally {
    window.clearTimeout(
      timeout
    );
  }
}


function extractDownloadFilename(
  response: Response,
  fallback: string
): string {
  const disposition =
    response.headers.get(
      "Content-Disposition"
    );

  if (!disposition) {
    return fallback;
  }

  const utf8Match =
    disposition.match(
      /filename\*=UTF-8''([^;\n]+)/i
    );

  if (utf8Match?.[1]) {
    const rawName =
      utf8Match[1].replace(
        /["']/g,
        ""
      );

    try {
      return decodeURIComponent(
        rawName
      );
    } catch {
      return rawName;
    }
  }

  const normalMatch =
    disposition.match(
      /filename\s*=\s*"?([^";\n]+)"?/i
    );

  return (
    normalMatch?.[1]?.trim() ||
    fallback
  );
}


export async function fetchFcsDisplayNames(
  jobId: string,
  params: {
    filenames: string[];
    mode: FcsDisplayNameMode;
    timeoutMs?: number;
  }
): Promise<FcsDisplayNamesResponse> {
  if (
    params.filenames.length === 0
  ) {
    return {
      names: {},
    };
  }

  return requestJson<FcsDisplayNamesResponse>({
    url: jobFcxmUrl(
      jobId,
      "fcs-display-names"
    ),
    init: {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify({
        filenames:
          params.filenames,
        mode: params.mode,
      }),
    },
    timeoutMs:
      params.timeoutMs ??
      10_000,
    timeoutMessage:
      "FCS display-name request timed out.",
    fallbackError:
      "Failed to resolve FCS display names",
  });
}


export async function extractPanelFromFcs(
  jobId: string,
  fcsFilenames: string[],
  timeoutMs = 30_000
): Promise<FcsPanelResponse> {
  requireJobId(jobId);

  if (
    fcsFilenames.length === 0
  ) {
    throw new Error(
      "Panel extraction requires at least one FCS file."
    );
  }

  return requestJson<FcsPanelResponse>({
    url: jobFcxmUrl(
      jobId,
      "panel"
    ),
    init: {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify({
        fcs_filenames:
          fcsFilenames,
      }),
    },
    timeoutMs,
    timeoutMessage:
      "FCS panel extraction timed out.",
    fallbackError:
      "Failed to extract the FCS panel",
  });
}


export async function fetchFCXMResults(
  jobId: string,
  params: FCXMResultsRequest
): Promise<FCXMResultsResponse> {
  requireJobId(jobId);

  if (
    !params.fcs_filename
  ) {
    throw new Error(
      "Results request is missing fcs_filename."
    );
  }

  return requestJson<FCXMResultsResponse>({
    url: jobFcxmUrl(
      jobId,
      "results"
    ),
    init: {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify({
        fcs_filename:
          params.fcs_filename,
        gate:
          params.gate ?? "",
      }),
    },
    timeoutMs:
      params.timeoutMs ??
      10_000,
    timeoutMessage:
      "Results request timed out.",
    fallbackError:
      "Failed to load FCXM results",
  });
}


export async function runFCXMAnalysis(
  jobId: string,
  payload: FCXMRunRequest,
  timeoutMs = 30_000
): Promise<FCXMRunStartResponse> {
  requireJobId(jobId);

  if (
    payload.samples.length === 0
  ) {
    throw new Error(
      "FCXM analysis requires at least one sample."
    );
  }

  return requestJson<FCXMRunStartResponse>({
    url: jobFcxmUrl(
      jobId,
      "run"
    ),
    init: {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify(
        payload
      ),
    },
    timeoutMs,
    timeoutMessage:
      "FCXM run request timed out.",
    fallbackError:
      "Failed to start FCXM analysis",
  });
}


export async function fetchFCXMRunProgress(
  jobId: string,
  timeoutMs = 10_000
): Promise<FCXMRunProgressResponse> {
  return requestJson<FCXMRunProgressResponse>({
    url: jobFcxmUrl(
      jobId,
      "progress"
    ),
    timeoutMs,
    timeoutMessage:
      "Progress request timed out.",
    fallbackError:
      "Failed to load FCXM progress",
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
  requireJobId(jobId);

  const query =
    new URLSearchParams();

  query.set(
    "positivity_metric",
    positivityMetric
  );

  const threshold =
    positivityThreshold.trim();

  if (threshold) {
    query.set(
      "positivity_threshold",
      threshold
    );
  }

  const controller =
    new AbortController();

  const timeout =
    window.setTimeout(
      () =>
        controller.abort(),
      timeoutMs
    );

  try {
    const response =
      await fetch(
        `${jobFcxmUrl(
          jobId,
          "summary.pdf"
        )}?${query.toString()}`,
        {
          method: "GET",
          signal:
            controller.signal,
        }
      );

    if (!response.ok) {
      throw new Error(
        await readApiError(
          response,
          "Failed to download FCXM summary"
        )
      );
    }

    const blob =
      await response.blob();

    const objectUrl =
      URL.createObjectURL(
        blob
      );

    const filename =
      extractDownloadFilename(
        response,
        `fcxm_summary_${jobId}.pdf`
      );

    const anchor =
      document.createElement(
        "a"
      );

    anchor.href =
      objectUrl;

    anchor.download =
      filename;

    document.body.appendChild(
      anchor
    );

    anchor.click();
    anchor.remove();

    window.setTimeout(
      () =>
        URL.revokeObjectURL(
          objectUrl
        ),
      0
    );

  } catch (error: unknown) {
    if (
      isAbortError(error)
    ) {
      throw new Error(
        "Summary download timed out."
      );
    }

    throw error;

  } finally {
    window.clearTimeout(
      timeout
    );
  }
}
