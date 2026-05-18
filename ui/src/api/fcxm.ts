import { API_BASE } from "../App";

export type ChannelRole = "Scatter" | "Population Marker" | "IgG";
export type SimPoint = { x: number; y: number; inGate?: boolean };
export type SampleRole = "NC" | "PC" | "SAMPLE";

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
  n_total: number;
  n_pos: number;
  pos_pct: number;
  filename?: string | null;
  sample_name?: string | null;
  role?: string | null;
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
  [k: string]: any;
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

export type FCXMRunProgress = {
  status: "queued" | "running" | "done" | "error";
  message?: string | null;
  stage?: string | null;
  result?: any;
  total_files?: number | null;
  done_files?: number | null;
  current_file?: string | null;
  done_filenames?: string[];
};

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

export async function fetchFcsDisplayNames(params: {
  filenames: string[];
  mode: FcsDisplayNameMode;
  timeoutMs?: number;
}): Promise<FcsDisplayNamesResponse> {
  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    params.timeoutMs ?? 10_000
  );

  try {
    const res = await fetch(`${API_BASE}/api/fcxm/fcs-display-names`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filenames: params.filenames,
        mode: params.mode,
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      let detail = "";
      try {
        const data = await res.json();
        detail = data?.detail || data?.message || "";
      } catch {
        // ignore
      }

      throw new Error(
        detail || `Failed to resolve FCS display names (${res.status})`
      );
    }

    return await res.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function extractPanelFromFcs(
  fcsFilenames: string[]
): Promise<FcsPanelResponse> {
  const res = await fetch(`${API_BASE}/api/fcs/panel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fcs_filenames: fcsFilenames }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Server error (${res.status})`);
  }

  return (await res.json()) as FcsPanelResponse;
}

export async function fetchFCXMResults(
  params: FCXMResultsRequest
): Promise<FCXMResultsResponse> {
  const timeoutMs = params.timeoutMs ?? 10_000;

  if (!params.job_id) {
    throw new Error("Results request is missing job_id");
  }
  if (!params.fcs_filename) {
    throw new Error("Results request is missing fcs_filename");
  }

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}/api/fcxm/results`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_id: params.job_id,
        fcs_filename: params.fcs_filename,
        gate: params.gate ?? "",
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `Server error (${res.status})`);
    }
    return (await res.json()) as FCXMResultsResponse;
  } catch (err: any) {
    if (err?.name === "AbortError") throw new Error("Results request timed out.");
    throw err;
  } finally {
    clearTimeout(t);
  }
}

export async function runFCXMAnalysis(
  payload: FCXMRunRequest,
  timeoutMs = 30_000
): Promise<FCXMRunStartResponse> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}/api/fcxm/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `Server error (${res.status})`);
    }
    return (await res.json()) as FCXMRunStartResponse;
  } catch (err: any) {
    if (err?.name === "AbortError") throw new Error("Run request timed out.");
    throw err;
  } finally {
    clearTimeout(t);
  }
}

export async function fetchFCXMRunProgress(
  jobId: string,
  timeoutMs = 10_000
): Promise<FCXMRunProgress> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(
      `${API_BASE}/api/fcxm/run/${encodeURIComponent(jobId)}`,
      { signal: controller.signal }
    );

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `Server error (${res.status})`);
    }
    return (await res.json()) as FCXMRunProgress;
  } catch (err: any) {
    if (err?.name === "AbortError") throw new Error("Progress request timed out.");
    throw err;
  } finally {
    clearTimeout(t);
  }
}

export async function downloadFCXMSummaryPdf(
  jobId: string,
  positivityMetric: "Median Ratio" | "Median Shift" | "Fluorescence Index" | "% pos",
  positivityThreshold: string,
  timeoutMs = 30_000
): Promise<void> {
  if (!jobId) throw new Error("Missing jobId");

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const params = new URLSearchParams({
      positivity_metric: positivityMetric,
      positivity_threshold: positivityThreshold,
    });
    
    const res = await fetch(
      `${API_BASE}/api/fcxm/summary/${encodeURIComponent(jobId)}?${params.toString()}`,
      { method: "GET", signal: controller.signal }
    );

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `Server error (${res.status})`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "summary.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  } catch (err: any) {
    if (err?.name === "AbortError") throw new Error("Download timed out.");
    throw err;
  } finally {
    clearTimeout(t);
  }
}
