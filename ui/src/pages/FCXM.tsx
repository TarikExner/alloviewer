import { useEffect, useMemo, useState } from "react";
import type { DragEvent } from "react";
import { Toolbar } from "../components/Toolbar";
import { UploadCard } from "../components/UploadCard";
import { SampleCard, type SampleCardModel } from "../components/SampleCard";
import { type PanelRow } from "../components/PanelTable";
import { PlotCard, ScatterPlot, LinePlot } from "../components/Plots";
import { FullPanelTable } from "../components/FullPanelTable";
import { StepButton, type StepState } from "../components/StepButton";
import { StatusPill, type RunStatus } from "../components/StatusPill";
import { ZoomModal } from "../components/ZoomModal";
import { fmtNum, fmtPct01 } from "../lib/format";
import { createId } from "../lib/id";
import {
  makeInitialCards,
  mapSampleRole,
  nextSampleTitle,
  readDraggedFcs,
} from "../lib/sampleCards";
import {
  extractPanelFromFcs,
  fetchFCXMResults,
  runFCXMAnalysis,
  fetchFCXMRunProgress,
  downloadFCXMSummaryPdf,
  fetchFcsDisplayNames,
  type FCXMSample,
  type FCXMResultsResponse,
  type FcsPanelResponse,
  type FcsDisplayNameMode,
} from "../api/fcxm";

type ActiveStep = 1 | 2 | 3 | 4;

type PositivityMetric =
  | "Median Ratio"
  | "Median Shift"
  | "Fluorescence Index"
  | "% pos";

type ZoomPlot = {
  title: string;
  subtitle?: string;
  xLabel: string;
  yLabel: string;
  points: { x: number; y: number; inGate?: boolean }[];
};

export default function FCXMApp() {
  const [, setFcsFilesLocal] = useState<File[]>([]);
  const [fcsSavedNames, setFcsSavedNames] = useState<string[]>([]);
  const [uploadResetKey, setUploadResetKey] = useState(0);

  const [fcsNameMode, setFcsNameMode] =
    useState<FcsDisplayNameMode>("tube_name");
  const [fcsDisplayNames, setFcsDisplayNames] = useState<Record<string, string>>(
    {}
  );
  const [fcsDisplayNamesBusy, setFcsDisplayNamesBusy] = useState(false);
  const [fcsDisplayNamesError, setFcsDisplayNamesError] = useState<
    string | null
  >(null);

  const [selectedFile, setSelectedFile] = useState("");
  const [selectedGate, setSelectedGate] = useState("");
  const [gateOptions, setGateOptions] = useState<string[]>([]);
  const [showGating, setShowGating] = useState(true);

  const [resultsBusy, setResultsBusy] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const [results, setResults] = useState<FCXMResultsResponse | null>(null);

  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [positivityMetric, setPositivityMetric] =
    useState<PositivityMetric>("Median Ratio");
  const [positivityThreshold, setPositivityThreshold] = useState("");

  const [cards, setCards] = useState<SampleCardModel[]>(makeInitialCards);
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null);

  const [runBusy, setRunBusy] = useState(false);
  const [runMsg, setRunMsg] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus>("idle");
  const [runJobId, setRunJobId] = useState<string | null>(null);
  const [runStage, setRunStage] = useState<string | null>(null);
  const [runDoneFiles, setRunDoneFiles] = useState<number | null>(null);
  const [runTotalFiles, setRunTotalFiles] = useState<number | null>(null);
  const [runCurrentFile, setRunCurrentFile] = useState<string | null>(null);

  const [activeStep, setActiveStep] = useState<ActiveStep>(1);
  const [panelVisited, setPanelVisited] = useState(false);
  const [autoJumpedToPanel, setAutoJumpedToPanel] = useState(false);

  const [zoomPlot, setZoomPlot] = useState<ZoomPlot | null>(null);

  const [panelBusy, setPanelBusy] = useState(false);
  const [panelError, setPanelError] = useState<string | null>(null);
  const [, setPanelData] = useState<FcsPanelResponse | null>(null);
  const [panelRows, setPanelRows] = useState<PanelRow[]>([]);

  const [activeCurveKey, setActiveCurveKey] = useState<string | null>(null);

  const selectedCard = useMemo(
    () => cards.find((c) => c.id === selectedCardId) ?? null,
    [cards, selectedCardId]
  );

  const fileOptions = useMemo(() => selectedCard?.fcsFiles ?? [], [selectedCard]);

  const assignedFilenames = useMemo(() => {
    const s = new Set<string>();
    cards.forEach((c) => c.fcsFiles.forEach((f) => s.add(f)));
    return Array.from(s);
  }, [cards]);

  const uploadedCount = fcsSavedNames.length;
  const assignedCount = assignedFilenames.length;
  const unassignedCount = Math.max(0, uploadedCount - assignedCount);

  const hasUploadedFiles = uploadedCount > 0;
  const hasAnyNC = cards.some(
    (c) => c.sampleType === "negative" && c.fcsFiles.length > 0
  );
  const allUploadedAssigned =
    hasUploadedFiles && assignedFilenames.length >= fcsSavedNames.length;

  const hasPanelRows = panelRows.length > 0;
  const hasIgGMarker = panelRows.some((r) =>
    String(r.role || "").toLowerCase().includes("igg")
  );

  const setupStepStatus: StepState = !hasUploadedFiles
    ? "not_started"
    : !hasAnyNC
    ? "needs_attention"
    : allUploadedAssigned
    ? "done"
    : "ready";

  const panelStepStatus: StepState = panelError
    ? "error"
    : panelBusy
    ? "running"
    : !hasUploadedFiles
    ? "not_started"
    : hasPanelRows && !panelVisited
    ? "needs_review"
    : hasPanelRows
    ? "done"
    : "needs_attention";

  const runStepStatus: StepState =
    runStatus === "error"
      ? "error"
      : runStatus === "running" || runStatus === "queued" || resultsBusy
      ? "running"
      : runStatus === "done"
      ? "done"
      : hasAnyNC && hasPanelRows
      ? "ready"
      : "not_started";

  const reportStepStatus: StepState =
    runStatus === "done" ? "ready" : "not_started";

  const canRun = !(runBusy || panelBusy || !!panelError || panelRows.length === 0);
  const canDownload = !(summaryBusy || runStatus !== "done" || !runJobId);

  const statusForRun: RunStatus = resultsBusy
    ? "running"
    : resultsError
    ? "error"
    : runStatus;

  const runProgressPercent = useMemo(() => {
    if (
      typeof runDoneFiles !== "number" ||
      typeof runTotalFiles !== "number" ||
      runTotalFiles <= 0
    ) {
      return null;
    }

    return Math.max(0, Math.min(100, (runDoneFiles / runTotalFiles) * 100));
  }, [runDoneFiles, runTotalFiles]);

  const runStageMessage =
    runStage === "fit_qc"
      ? "Running file QC."
      : runStage === "fit_marker_calibration"
      ? "Calibrating markers."
      : runStage === "fit_lymphocytes"
      ? "Gating lymphocytes."
      : runStage === "fit_clustering"
      ? "Fitting clustering model."
      : runStage === "fit_cluster_labels"
      ? "Labeling clusters."
      : runStage === "fit_control_stats"
      ? "Building control statistics."
      : runStage === "apply_file"
      ? "Applying gates to files."
      : runStage === "payload"
      ? "Building result payload."
      : runStage === "plot_cache"
      ? "Building plot cache."
      : null;

  const messageForRun = resultsBusy
    ? "Loading results…"
    : runStatus === "running"
    ? runStageMessage || runMsg || "Running analysis."
    : runJobId && runStatus !== "done"
    ? runMsg || "Results will load when the run is finished."
    : runMsg;

  useEffect(() => {
    if (!fileOptions.length) {
      setSelectedFile("");
      return;
    }

    setSelectedFile((prev) =>
      prev && fileOptions.includes(prev) ? prev : fileOptions[0]
    );
  }, [fileOptions]);

  useEffect(() => {
    if (selectedCardId && !cards.some((c) => c.id === selectedCardId)) {
      setSelectedCardId(null);
    }
  }, [cards, selectedCardId]);

  useEffect(() => {
    setActiveCurveKey(null);
  }, [results]);

  useEffect(() => {
    if (runStatus !== "done") return;
    if (selectedCardId) return;

    const firstCardWithFiles =
      cards.find((c) => c.sampleType === "sample" && c.fcsFiles.length > 0) ??
      cards.find((c) => c.fcsFiles.length > 0) ??
      cards[0];

    if (firstCardWithFiles) {
      setSelectedCardId(firstCardWithFiles.id);
    }
  }, [runStatus, selectedCardId, cards]);

  useEffect(() => {
    if (!selectedCard || !selectedFile) {
      setResults(null);
      setResultsError(null);
      return;
    }
  
    if (!runJobId || runStatus !== "done") {
      setResults(null);
      setResultsError(null);
      return;
    }
  
    const jobId = runJobId;
    let cancelled = false;
  
    async function loadResults() {
      setResultsBusy(true);
      setResultsError(null);
  
      try {
        const res = await fetchFCXMResults({
          job_id: jobId,
          fcs_filename: selectedFile,
          gate: selectedGate,
          timeoutMs: 10_000,
        });
  
        if (cancelled) return;
  
        setResults(res);
  
        const opts = res.gate_options || [];
        setGateOptions(opts);
        setSelectedGate((prev) =>
          prev && opts.includes(prev) ? prev : opts[0] || ""
        );
      } catch (err: any) {
        if (cancelled) return;
        setResultsError(err?.message || "Failed to load results");
        setResults(null);
      } finally {
        if (!cancelled) setResultsBusy(false);
      }
    }
  
    loadResults();
  
    return () => {
      cancelled = true;
    };
  }, [selectedCard?.id, selectedFile, selectedGate, runJobId, runStatus]);

  useEffect(() => {
    if (!fcsSavedNames.length) {
      setFcsDisplayNames({});
      setFcsDisplayNamesError(null);
      setFcsDisplayNamesBusy(false);
      return;
    }

    let cancelled = false;

    async function loadDisplayNames() {
      setFcsDisplayNamesBusy(true);
      setFcsDisplayNamesError(null);

      try {
        const res = await fetchFcsDisplayNames({
          filenames: fcsSavedNames,
          mode: fcsNameMode,
          timeoutMs: 10_000,
        });

        if (cancelled) return;

        const fallback = Object.fromEntries(fcsSavedNames.map((f) => [f, f]));
        setFcsDisplayNames({
          ...fallback,
          ...(res.names || {}),
        });
      } catch (err: any) {
        if (cancelled) return;

        const fallback = Object.fromEntries(fcsSavedNames.map((f) => [f, f]));
        setFcsDisplayNames(fallback);
        setFcsDisplayNamesError(
          err?.message || "Could not read FCS display names."
        );
      } finally {
        if (!cancelled) setFcsDisplayNamesBusy(false);
      }
    }

    loadDisplayNames();

    return () => {
      cancelled = true;
    };
  }, [fcsSavedNames, fcsNameMode]);

  useEffect(() => {
    if (autoJumpedToPanel) return;
    if (!allUploadedAssigned) return;
    if (!hasPanelRows) return;
    if (activeStep !== 1) return;

    setActiveStep(2);
    setPanelVisited(true);
    setAutoJumpedToPanel(true);
  }, [autoJumpedToPanel, allUploadedAssigned, hasPanelRows, activeStep]);

  function goToStep(step: ActiveStep) {
    setActiveStep(step);
    if (step === 2) setPanelVisited(true);
  }

  function updateCard(id: string, patch: Partial<SampleCardModel>) {
    setCards((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }

  function addSampleCard() {
    setCards((prev) => {
      const title = nextSampleTitle(prev);
      return [
        ...prev,
        { id: createId(), sampleType: "sample", title, name: title, fcsFiles: [] },
      ];
    });
  }

  function removeSampleCard(id: string) {
    setCards((prev) => prev.filter((c) => c.id !== id));
  }

  function displayNameForFcs(fname: string) {
    return fcsDisplayNames[fname] || fname;
  }
  
  function basename(pathOrName: string | null | undefined) {
    const text = String(pathOrName || "");
    return text.split(/[\\/]/).pop() || text;
  }
  
  function displayNameForFcsLoose(fname: string | null | undefined) {
    if (!fname) return "";
  
    const direct = fcsDisplayNames[fname];
    if (direct) return direct;
  
    const base = basename(fname);
    const byBase = fcsDisplayNames[base];
    if (byBase) return byBase;
  
    const matchedKey = Object.keys(fcsDisplayNames).find(
      (key) => basename(key) === base
    );
  
    if (matchedKey) return fcsDisplayNames[matchedKey];
  
    return base;
  }

  type OverviewCurveCard = {
    key: string;
    label: string;
    roleLabel: string;
    filename: string | null;
    color: string;
    median: number | null;
  };

  type OverviewLineSeries = {
    key: string;
    label: string;
    color: string;
    values: number[];
    filename: string | null;
    role: string | null;
    roleLabel: string | null;
    rawMedian: number | null;
    xLabel: string | null;
  };  

  function median(values: number[] | undefined | null): number | null {
    const clean = (values || [])
      .filter((v) => typeof v === "number" && Number.isFinite(v))
      .sort((a, b) => a - b);
  
    if (!clean.length) return null;
  
    const mid = Math.floor(clean.length / 2);
  
    if (clean.length % 2 === 1) {
      return clean[mid];
    }
  
    return (clean[mid - 1] + clean[mid]) / 2;
  }
  
  function roleLabelForLineSeries(s: { label: string; role?: string | null }) {
    const role = String(s.role || "").toUpperCase();
  
    if (role === "NC") return "Negative control";
    if (role === "PC") return "Positive control";
  
    if (String(s.label || "").startsWith("Negative control")) {
      return "Negative control";
    }
  
    if (String(s.label || "").startsWith("Positive control")) {
      return "Positive control";
    }
  
    return null;
  }

  const overviewLineSeries: OverviewLineSeries[] = useMemo(() => {
    return (results?.line_series || []).map((line: any, idx: number) => {
      const roleLabel = roleLabelForLineSeries(line);

      const filename =
        line.filename ||
        String(line.label || "")
          .split(" · ")
          .at(-1) ||
        null;

      return {
        key: `${line.label || "series"}-${filename || ""}-${idx}`,
        label: line.label || `Series ${idx + 1}`,
        color: line.color || "#64748b",
      
        // transformed values for the plot
        values: Array.isArray(line.values) ? line.values : [],
      
        filename,
        role: typeof line.role === "string" ? line.role : null,
        roleLabel,
      
        // raw values for the cards; never fall back to line.values
        rawMedian:
          typeof line.raw_median === "number"
            ? line.raw_median
            : Array.isArray(line.values_raw)
            ? median(line.values_raw)
            : null,
      
        xLabel: line.x_label || null,
      };
    });
  }, [results?.line_series]);


  
  const controlCurveCards: OverviewCurveCard[] = useMemo(() => {
    return overviewLineSeries
      .filter((s) => !!s.roleLabel)
      .map((s) => ({
        key: s.key,
        label: displayNameForFcsLoose(s.filename),
        roleLabel: s.roleLabel as string,
        filename: s.filename,
        color: s.color,
        median: s.rawMedian,
      }));
  }, [overviewLineSeries, fcsDisplayNames]);

  const selectedFileCurveKey = useMemo(() => {
    const selectedSeries = overviewLineSeries.find((s) => {
      const role = String((s as any).role || "").toUpperCase();
  
      if (role === "__SEL__") return true;
  
      if (String(s.label || "") === "Selected file") return true;
  
      const selectedBase = basename(selectedFile);
      const seriesBase = basename(s.filename);
  
      return !!selectedBase && !!seriesBase && selectedBase === seriesBase;
    });
  
    return selectedSeries?.key || null;
  }, [overviewLineSeries, selectedFile]);

  function handleDropFcs(targetId: string, e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();

    const item = readDraggedFcs(e);
    if (!item) return;

    const { fname } = item;

    setCards((prev) => {
      const cleared = prev.map((c) => {
        if (!c.fcsFiles.includes(fname)) return c;
        return { ...c, fcsFiles: c.fcsFiles.filter((x) => x !== fname) };
      });

      return cleared.map((c) => {
        if (c.id !== targetId) return c;
        if (c.fcsFiles.includes(fname)) return c;
        return { ...c, fcsFiles: [...c.fcsFiles, fname] };
      });
    });
  }

  function allowDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
  }

  async function extractPanel(names: string[]) {
    if (!names.length) return;

    setPanelBusy(true);
    setPanelError(null);
    setPanelData(null);
    setPanelRows([]);

    try {
      const res = await extractPanelFromFcs(names);
      setPanelData(res);
      setPanelRows((res as any).rows ?? []);
    } catch (err: any) {
      setPanelError(err?.message || "Panel extraction failed");
    } finally {
      setPanelBusy(false);
    }
  }

  async function onRunAnalysis() {
    setActiveStep(3);
    setRunBusy(true);
    setRunMsg(null);
    setRunStatus("queued");
    setRunJobId(null);

    setRunStage(null);
    setRunDoneFiles(0);
    setRunTotalFiles(null);
    setRunCurrentFile(null);

    setResults(null);
    setResultsError(null);

    const samples: FCXMSample[] = cards.map((c) => ({
      id: c.id,
      name: c.name || c.title,
      role: mapSampleRole(c.sampleType),
      file_paths: (c.fcsFiles || []).slice(),
    }));

    if (!panelRows || panelRows.length === 0) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg("Panel is empty. Upload FCS files first.");
      return;
    }

    const missingRequired = samples.filter(
      (s) => s.role === "NC" && s.file_paths.length === 0
    );

    if (missingRequired.length > 0) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg(
        `Missing files for: ${missingRequired
          .map((s) => `${s.name} (${s.role})`)
          .join(", ")}`
      );
      return;
    }

    const hasAnyNCFile = samples.some(
      (s) => s.role === "NC" && s.file_paths.length > 0
    );

    if (!hasAnyNCFile) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg("At least one Negative Control (NC) file is required.");
      return;
    }

    const samplesToSend = samples.filter(
      (s) => s.role !== "PC" || s.file_paths.length > 0
    );

    try {
      const { job_id } = await runFCXMAnalysis({
        panel_rows: panelRows,
        samples: samplesToSend,
      });

      setRunJobId(job_id);
      setRunStatus("running");
      setRunMsg("Run started.");

      const poll = async () => {
        try {
          const prog = await fetchFCXMRunProgress(job_id);

          setRunStatus(prog.status);
          setRunStage(prog.stage ?? null);
          setRunDoneFiles(
            typeof prog.done_files === "number" ? prog.done_files : null
          );
          setRunTotalFiles(
            typeof prog.total_files === "number" ? prog.total_files : null
          );
          setRunCurrentFile(prog.current_file ?? null);

          if (prog.message) setRunMsg(prog.message);

          if (prog.status === "done") {
            setRunBusy(false);
            setRunMsg(prog.message || "Done.");
            setRunStage(null);
            setRunCurrentFile(null);

            if (typeof prog.total_files === "number") {
              setRunDoneFiles(prog.total_files);
              setRunTotalFiles(prog.total_files);
            }

            return;
          }

          if (prog.status === "error") {
            setRunBusy(false);
            setRunMsg(prog.message || "Run failed.");
            setRunStage(null);
            setRunCurrentFile(null);
            return;
          }

          setTimeout(poll, 800);
        } catch (err: any) {
          setRunBusy(false);
          setRunStatus("error");
          setRunMsg(err?.message || "Progress check failed.");
          setRunStage(null);
          setRunCurrentFile(null);
        }
      };

      poll();
    } catch (err: any) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg(err?.message || "Run failed.");
      setRunStage(null);
      setRunCurrentFile(null);
    }
  }

  async function onDownloadSummary() {
    if (!runJobId) return;

    setSummaryBusy(true);
    setSummaryError(null);

    try {
      await downloadFCXMSummaryPdf(
        runJobId,
        positivityMetric,
        positivityThreshold
      );
    } catch (err: any) {
      setSummaryError(err?.message || "Download failed.");
    } finally {
      setSummaryBusy(false);
    }
  }

  function resetExperiment() {
    setUploadResetKey((x) => x + 1);

    setFcsFilesLocal([]);
    setFcsSavedNames([]);

    setFcsNameMode("tube_name");
    setFcsDisplayNames({});
    setFcsDisplayNamesBusy(false);
    setFcsDisplayNamesError(null);

    setSelectedFile("");
    setSelectedGate("");
    setGateOptions([]);

    setShowGating(true);

    setResultsBusy(false);
    setResultsError(null);
    setResults(null);
    setActiveCurveKey(null);

    setSummaryBusy(false);
    setSummaryError(null);

    setPositivityMetric("Median Ratio");
    setPositivityThreshold("");

    setCards(makeInitialCards());
    setSelectedCardId(null);

    setRunBusy(false);
    setRunMsg(null);
    setRunStatus("idle");
    setRunJobId(null);

    setRunStage(null);
    setRunDoneFiles(null);
    setRunTotalFiles(null);
    setRunCurrentFile(null);

    setActiveStep(1);
    setPanelVisited(false);
    setAutoJumpedToPanel(false);

    setZoomPlot(null);

    setPanelBusy(false);
    setPanelError(null);
    setPanelData(null);
    setPanelRows([]);
  }

  return (
    <div className="h-screen overflow-hidden bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar title="AlloViewer - Flow Cytometry Crossmatch" />

      <div className="flex-1 min-h-0 overflow-hidden p-4 grid grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)] gap-4">
        <aside className="min-h-0 overflow-y-auto pr-1">
          <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
            <div className="font-medium">Workflow</div>
            <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              Select a step. The main work area stays on the right.
            </div>
          </div>

          <div className="space-y-3">
            <StepButton
              number={1}
              title="Upload and assign"
              status={setupStepStatus}
              active={activeStep === 1}
              onClick={() => goToStep(1)}
              summary={
                !hasUploadedFiles
                  ? "Upload .fcs files"
                  : !hasAnyNC
                  ? `${uploadedCount} file(s), NC missing`
                  : unassignedCount > 0
                  ? `${uploadedCount} file(s), ${unassignedCount} unassigned`
                  : `${uploadedCount} file(s), all assigned`
              }
            />

            <StepButton
              number={2}
              title="Review panel"
              status={panelStepStatus}
              active={activeStep === 2}
              onClick={() => goToStep(2)}
              summary={
                panelBusy
                  ? "Reading panel"
                  : panelError
                  ? "Panel error"
                  : hasPanelRows && !panelVisited
                  ? `${panelRows.length} channel(s), needs review`
                  : hasPanelRows
                  ? `${panelRows.length} channel(s)`
                  : "No panel yet"
              }
            />

            <StepButton
              number={3}
              title="Run and results"
              status={runStepStatus}
              active={activeStep === 3}
              onClick={() => goToStep(3)}
              summary={
                runStatus === "done"
                  ? "Analysis completed"
                  : runStatus === "running" || runStatus === "queued"
                  ? "Analysis running"
                  : hasAnyNC && hasPanelRows
                  ? "Ready to run"
                  : "Setup incomplete"
              }
            />

            <StepButton
              number={4}
              title="Report"
              status={reportStepStatus}
              active={activeStep === 4}
              onClick={() => goToStep(4)}
              summary={
                runStatus === "done"
                  ? "Set rule and download"
                  : "Available after run"
              }
            />

            <div className="pt-3">
              <button
                type="button"
                onClick={resetExperiment}
                className="w-full rounded-2xl border px-3 py-2.5 text-sm
                           bg-white hover:bg-red-50 text-red-700 border-red-200
                           dark:bg-neutral-900 dark:hover:bg-red-950
                           dark:border-red-900 dark:text-red-300"
              >
                Reset Experiment
              </button>
            </div>
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto pr-1">
          <section className={activeStep === 1 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">Step 1 · Upload and assign files</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Upload a folder of .fcs files, then drag files into the correct cards.
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 items-stretch">
              <div className="min-h-0 flex flex-col">
                <UploadCard
                  key={uploadResetKey}
                  title="FCS Files (Folder)"
                  accept=".fcs"
                  allowDirectory
                  autoUpload
                  fileFilter={(file) => file.name.toLowerCase().endsWith(".fcs")}
                  onPicked={setFcsFilesLocal}
                  onUploaded={(saved) => {
                    const names = (saved || [])
                      .map((x: any) =>
                        typeof x === "string" ? x : x?.filename
                      )
                      .filter(Boolean);

                    setFcsSavedNames(names);
                    setPanelVisited(false);
                    setAutoJumpedToPanel(false);
                    extractPanel(names);
                  }}
                  showUploadedList
                  uploadedListLabel="Uploaded (.fcs)"
                  assignedFilenames={assignedFilenames}
                  hideAssigned={true}
                  hideSelectedList={true}
                  renderUploadedItem={(fname) => {
                    const displayName = displayNameForFcs(fname);
                    const hasDisplayName = displayName !== fname;

                    return (
                      <div className="min-w-0">
                        <div className="truncate font-medium">{displayName}</div>

                        {hasDisplayName ? (
                          <div className="truncate text-[11px] text-neutral-500 dark:text-neutral-500">
                            {fname}
                          </div>
                        ) : null}
                      </div>
                    );
                  }}
                  className="h-[520px]"
                />

                <div className="mt-3 rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                  <div className="text-sm font-medium mb-2">display names</div>

                  <select
                    value={fcsNameMode}
                    onChange={(e) =>
                      setFcsNameMode(e.target.value as FcsDisplayNameMode)
                    }
                    className="w-full h-10 rounded-xl border px-3 bg-white
                               dark:bg-neutral-900 dark:border-neutral-700"
                  >
                    <option value="filename">Use file names</option>
                    <option value="tube_name">Use tube names</option>
                  </select>

                  {fcsDisplayNamesBusy ? (
                    <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                      Reading FCS labels…
                    </div>
                  ) : null}

                  {fcsDisplayNamesError ? (
                    <div className="mt-2 text-xs text-red-600 dark:text-red-400">
                      {fcsDisplayNamesError}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 h-[620px] min-h-0 flex flex-col">
                <div className="flex items-center justify-between gap-2 mb-3">
                  <div>
                    <h3 className="font-medium">Sample Cards</h3>
                    <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
                      Negative Control is required. Positive Control is optional.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={addSampleCard}
                    className="text-sm px-3 py-1.5 rounded-lg border bg-white hover:bg-neutral-50
                               dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
                  >
                    Add Sample
                  </button>
                </div>

                <div className="flex-1 min-h-0 overflow-auto pr-1 space-y-3">
                  {cards.map((c) => (
                    <SampleCard
                      key={c.id}
                      card={c}
                      selected={c.id === selectedCardId}
                      onSelect={() =>
                        setSelectedCardId((prev) =>
                          prev === c.id ? null : c.id
                        )
                      }
                      onRemoveCard={
                        c.sampleType === "sample"
                          ? () => removeSampleCard(c.id)
                          : undefined
                      }
                      onNameChange={(next) => updateCard(c.id, { name: next })}
                      onDragOverFiles={allowDrop}
                      onDropFile={(e) => handleDropFcs(c.id, e)}
                      onRemoveFile={(fname) =>
                        updateCard(c.id, {
                          fcsFiles: c.fcsFiles.filter((x) => x !== fname),
                        })
                      }
                      fileDisplayNames={fcsDisplayNames}
                    />
                  ))}
                </div>

                <p className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
                  Drag uploaded files from the left list into the correct card.
                </p>
              </div>
            </div>
          </section>

          <section className={activeStep === 2 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">Step 2 · Review panel</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Check channel roles and remove channels that should not be used.
              </div>
            </div>

            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <h3 className="font-medium">Panel</h3>
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  {uploadedCount
                    ? `${uploadedCount} file(s) uploaded`
                    : "No files yet"}
                </div>
              </div>

              <div className="space-y-3">
                {panelBusy ? (
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Reading panel from .fcs files…
                  </div>
                ) : panelError ? (
                  <div className="text-sm text-red-600 dark:text-red-400">
                    {panelError}
                  </div>
                ) : (
                  <FullPanelTable rows={panelRows} onChange={setPanelRows} />
                )}

                {hasPanelRows && !hasIgGMarker ? (
                  <div className="text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                    No IgG marker is selected. Check the panel before running.
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className={activeStep === 3 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">Step 3 · Run and inspect results</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Start analysis, then inspect the selected sample and file.
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_220px] gap-3 items-center">
                  <StatusPill
                    status={statusForRun}
                    message={messageForRun}
                    error={resultsError}
                    jobId={runJobId}
                    busy={runBusy || resultsBusy}
                    progressPercent={runProgressPercent}
                    currentFile={runCurrentFile}
                  />

                  <button
                    type="button"
                    onClick={onRunAnalysis}
                    disabled={!canRun}
                    className="w-full py-3 rounded-xl border bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:hover:bg-blue-600
                               dark:border-blue-500"
                    title={
                      panelBusy
                        ? "Panel is still loading"
                        : panelError
                        ? "Fix the panel error first"
                        : panelRows.length === 0
                        ? "Load a panel first"
                        : "Run analysis"
                    }
                  >
                    {runBusy ? "Running…" : "Run analysis"}
                  </button>
                </div>
              </div>

              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="font-medium mb-3">Samples</div>

                <div className="overflow-x-auto">
                  <div className="flex gap-2 pb-1">
                    {cards.map((c) => {
                      const active = c.id === selectedCardId;
                      const fileCount = c.fcsFiles.length;

                      return (
                        <button
                          key={c.id}
                          type="button"
                          onClick={() => setSelectedCardId(c.id)}
                          disabled={fileCount === 0}
                          className={[
                            "shrink-0 w-[150px] min-h-[78px] rounded-xl border p-3 text-left transition-colors disabled:opacity-50",
                            active
                              ? "ring-2 ring-blue-500 border-blue-300 dark:border-blue-500 bg-blue-50 dark:bg-blue-950"
                              : "bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700",
                          ].join(" ")}
                          title={c.name || c.title}
                        >
                          <div className="text-sm font-medium truncate">
                            {c.name || c.title}
                          </div>

                          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
                            {c.sampleType === "negative"
                              ? "Negative Control"
                              : c.sampleType === "positive"
                              ? "Positive Control"
                              : "Sample"}
                          </div>

                          <div className="mt-2 text-xs">
                            {fileCount} file{fileCount === 1 ? "" : "s"}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[420px_minmax(0,1fr)] gap-4 items-stretch">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 h-full flex flex-col">
                  <div className="font-medium">Result selection</div>

                  <div className="mt-3 space-y-3">
                    <div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                        Selected sample
                      </div>
                      <div className="h-11 rounded-xl border px-3 flex items-center bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-800">
                        <span className="truncate text-sm font-medium">
                          {selectedCard
                            ? selectedCard.name || selectedCard.title
                            : "No sample selected"}
                        </span>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                        File
                      </div>
                      <select
                        value={selectedFile}
                        onChange={(e) => setSelectedFile(e.target.value)}
                        disabled={!selectedCard || fileOptions.length === 0}
                        className="w-full h-11 rounded-xl border px-3 bg-white
                                   dark:bg-neutral-900 dark:border-neutral-700 disabled:opacity-50 truncate"
                      >
                        {fileOptions.length === 0 ? (
                          <option value="">No files</option>
                        ) : (
                          fileOptions.map((f) => (
                            <option key={f} value={f}>
                              {displayNameForFcs(f)}
                            </option>
                          ))
                        )}
                      </select>
                    </div>

                    <div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                        Gate
                      </div>
                      <select
                        value={selectedGate}
                        onChange={(e) => setSelectedGate(e.target.value)}
                        disabled={!selectedCard || !runJobId || runStatus !== "done"}
                        className="w-full h-11 rounded-xl border px-3 bg-white
                                   dark:bg-neutral-900 dark:border-neutral-700 disabled:opacity-50"
                      >
                        {gateOptions.map((g) => (
                          <option key={g} value={g}>
                            {g}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 overflow-hidden flex flex-col">
                  <div className="font-medium">Overview</div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    Asinh-transformed IgG density curves for controls and selected file
                  </div>

                  <div className="mt-3 grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4">
                    <div className="min-h-0 flex flex-col gap-3">
                      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                        {controlCurveCards.map((card) => {
                          const hasTubeName =
                            card.filename && card.label && card.label !== basename(card.filename);                      
                          return (
                            <button
                              type="button"
                              key={card.key}
                              onClick={() =>
                                setActiveCurveKey((prev) =>
                                  prev === card.key ? null : card.key
                                )
                              }
                              className={[
                                "rounded-xl border p-3 dark:border-neutral-800 min-w-0 text-left transition",
                                activeCurveKey === card.key
                                  ? "ring-2 ring-blue-500 border-blue-400"
                                  : "hover:bg-neutral-50 dark:hover:bg-neutral-800",
                              ].join(" ")}
                            >
                              <div className="flex items-center gap-2 text-sm font-medium min-w-0">
                                <span
                                  className="inline-block w-3 h-3 rounded shrink-0"
                                  style={{ background: card.color }}
                                />
                                <span className="truncate">{card.label}</span>
                              </div>
                      
                              <div className="mt-1 text-[11px] text-neutral-500 dark:text-neutral-500 truncate">
                                {card.roleLabel}
                                {hasTubeName ? ` · ${basename(card.filename)}` : ""}
                              </div>
                      
                              <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                                IgG median
                              </div>
                      
                              <div className="text-base font-semibold">
                                {fmtNum(card.median)}
                              </div>
                            </button>
                          );
                        })}
                      
                        {results?.selected_file_metrics ? (
                          <button
                            type="button"
                            onClick={() => {
                              if (!selectedFileCurveKey) return;
                              setActiveCurveKey((prev) =>
                                prev === selectedFileCurveKey ? null : selectedFileCurveKey
                              );
                            }}
                            disabled={!selectedFileCurveKey}
                            className={[
                              "rounded-xl border p-3 dark:border-neutral-800 min-w-0 text-left transition disabled:cursor-default",
                              activeCurveKey === selectedFileCurveKey
                                ? "ring-2 ring-blue-500 border-blue-400"
                                : selectedFileCurveKey
                                ? "hover:bg-neutral-50 dark:hover:bg-neutral-800"
                                : "",
                            ].join(" ")}
                          >
                            <div className="flex items-center gap-2 text-sm font-medium min-w-0">
                              <span className="inline-block w-3 h-3 rounded shrink-0 bg-blue-500" />
                              <span className="truncate">Selected file</span>
                            </div>
                      
                            <div className="mt-1 text-[11px] text-neutral-500 dark:text-neutral-500 truncate">
                              {selectedFile ? displayNameForFcsLoose(selectedFile) : "—"}
                              {selectedFile && displayNameForFcsLoose(selectedFile) !== basename(selectedFile)
                                ? ` · ${basename(selectedFile)}`
                                : ""}
                            </div>                      
                            <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                              IgG median
                            </div>
                      
                            <div className="text-base font-semibold">
                              {fmtNum(results.selected_file_metrics.igg_median_raw)}
                            </div>
                          </button>
                        ) : null}
                      
                        {selectedCard && !results ? (
                          <div className="text-neutral-600 dark:text-neutral-400 text-sm">
                            No results yet.
                          </div>
                        ) : null}
                      </div>

                      <div className="overflow-hidden rounded-xl border dark:border-neutral-800 p-2 h-[260px]">
                        <LinePlot
                          series={overviewLineSeries.map((s) => ({
                            key: s.key,
                            label: s.label,
                            color: s.color,
                            values: s.values,
                          }))}
                          activeSeriesKey={activeCurveKey}
                          xLabel={"IgG"}
                          showLegend={false}
                        />
                      </div>
                    </div>

                    <div className="min-h-0 flex flex-col gap-3">
                      {results?.selected_file_metrics ? (
                        <div className="rounded-xl border p-3 dark:border-neutral-800">
                          <div className="text-sm font-medium mb-2">
                            Selected file - {results.selected_file_metrics.label}
                          </div>

                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            <div className="text-neutral-600 dark:text-neutral-400">
                              Events
                            </div>
                            <div>{results.selected_file_metrics.n_events}</div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Median shift
                            </div>
                            <div>
                              {fmtNum(results.selected_file_metrics.igg_median_shift)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Median ratio
                            </div>
                            <div>
                              {fmtNum(results.selected_file_metrics.igg_median_ratio)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Fluorescence index
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_file_metrics.igg_fluorescence_index
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              IgG+ fraction
                            </div>
                            <div>
                              {fmtPct01(results.selected_file_metrics.igg_pos_fraction)}
                            </div>
                          </div>
                        </div>
                      ) : null}

                      {results?.selected_sample_metrics ? (
                        <div className="rounded-xl border p-3 dark:border-neutral-800">
                          <div className="text-sm font-medium mb-2">
                            Combined sample - {results.selected_sample_metrics.label}
                          </div>

                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            <div className="text-neutral-600 dark:text-neutral-400">
                              Events
                            </div>
                            <div>{results.selected_sample_metrics.n_events}</div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Median shift
                            </div>
                            <div>
                              {fmtNum(results.selected_sample_metrics.igg_median_shift)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Median ratio
                            </div>
                            <div>
                              {fmtNum(results.selected_sample_metrics.igg_median_ratio)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              Fluorescence index
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_sample_metrics.igg_fluorescence_index
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              IgG+ fraction
                            </div>
                            <div>
                              {fmtPct01(results.selected_sample_metrics.igg_pos_fraction)}
                            </div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>

              <div
                className={`rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 overflow-hidden ${
                  showGating ? "h-auto" : "h-[64px]"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="font-medium">Gating strategy</div>

                  <label className="flex items-center gap-2 select-none">
                    <input
                      type="checkbox"
                      checked={showGating}
                      onChange={(e) => setShowGating(e.target.checked)}
                      className="rounded border"
                      disabled={!selectedCard}
                    />
                    <span className="text-sm">Show gating strategy</span>
                  </label>
                </div>

                {showGating ? (
                  <div className="mt-3 overflow-x-auto">
                    <div className="grid grid-flow-col auto-cols-[260px] gap-3">
                      {(results?.gating_plots ?? []).map((gp, idx) => (
                        <PlotCard
                          key={gp.title + idx}
                          title={gp.title}
                          subtitle={`${gp.x_label} vs ${gp.y_label}`}
                          className="min-h-[200px]"
                        >
                          <ScatterPlot
                            points={gp.points || []}
                            xLabel={gp.x_label}
                            yLabel={gp.y_label}
                            basePointRadius={1.3}
                            gatePointRadius={1.3}
                            onDoubleClick={() =>
                              setZoomPlot({
                                title: gp.title,
                                subtitle: `${gp.x_label} vs ${gp.y_label}`,
                                xLabel: gp.x_label,
                                yLabel: gp.y_label,
                                points: gp.points || [],
                              })
                            }
                          />
                        </PlotCard>
                      ))}

                      {selectedCard && (results?.gating_plots ?? []).length === 0 ? (
                        <div className="w-[260px] h-[200px] rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3 text-sm text-neutral-600 dark:text-neutral-400 flex items-center justify-center">
                          No gating plots yet.
                        </div>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className={activeStep === 4 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">Step 4 · Report settings</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Set the positivity rule used for the report, then download the summary.
              </div>
            </div>

            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
              <div className="font-medium">Report settings</div>

              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 items-end">
                <div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                    Metric
                  </div>

                  <select
                    value={positivityMetric}
                    onChange={(e) =>
                      setPositivityMetric(e.target.value as PositivityMetric)
                    }
                    className="w-full h-11 rounded-xl border px-3 bg-white
                               dark:bg-neutral-900 dark:border-neutral-700"
                  >
                    <option value="Median Ratio">Median Ratio</option>
                    <option value="Median Shift">Median Shift</option>
                    <option value="Fluorescence Index">Fluorescence Index</option>
                    <option value="% pos">% pos</option>
                  </select>
                </div>

                <div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                    Positive above
                  </div>

                  <input
                    type="number"
                    step="1"
                    value={positivityThreshold}
                    onChange={(e) => setPositivityThreshold(e.target.value)}
                    className="w-full h-11 rounded-xl border px-3 bg-white
                               dark:bg-neutral-900 dark:border-neutral-700"
                    placeholder="0.0"
                  />
                </div>
              </div>

              <div className="mt-3 rounded-xl border p-3 dark:border-neutral-800 text-sm">
                <div className="text-neutral-600 dark:text-neutral-400">
                  Current rule
                </div>

                <div className="mt-1 font-medium">
                  Positive if {positivityMetric} &gt;{" "}
                  {positivityThreshold === "" ? "—" : positivityThreshold}
                </div>
              </div>

              <button
                type="button"
                onClick={onDownloadSummary}
                disabled={!canDownload}
                className="mt-3 w-full py-2.5 rounded-xl border bg-white hover:bg-neutral-50 disabled:opacity-50
                           dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
                title={
                  runStatus !== "done"
                    ? "Run must be done before download"
                    : "Download summary"
                }
              >
                {summaryBusy ? "Preparing summary…" : "Download summary"}
              </button>

              {summaryError ? (
                <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                  {summaryError}
                </div>
              ) : null}
            </div>
          </section>
        </main>
      </div>

      <ZoomModal
        open={!!zoomPlot}
        onClose={() => setZoomPlot(null)}
        title={zoomPlot?.title || ""}
        subtitle={zoomPlot?.subtitle}
      >
        <ScatterPlot
          points={zoomPlot?.points || []}
          xLabel={zoomPlot?.xLabel || ""}
          yLabel={zoomPlot?.yLabel || ""}
          fillParent
        />
      </ZoomModal>
    </div>
  );
}
