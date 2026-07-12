import { useEffect, useMemo, useState } from "react";
import type { DragEvent } from "react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();

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

  const runStageMessage = runStage
    ? t(`FCXM.run_stages.${runStage}`, {
        defaultValue: t("FCXM.run_stages.default"),
      })
    : null;

  const messageForRun = resultsBusy
    ? t("FCXM.messages.loading_results")
    : runStatus === "running"
    ? runStageMessage || runMsg || t("FCXM.messages.running_analysis")
    : runJobId && runStatus !== "done"
    ? runMsg || t("FCXM.messages.results_after_run")
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
        setResultsError(err?.message || t("FCXM.messages.failed_to_load_results"));
        setResults(null);
      } finally {
        if (!cancelled) setResultsBusy(false);
      }
    }

    loadResults();

    return () => {
      cancelled = true;
    };
  }, [selectedCard?.id, selectedFile, selectedGate, runJobId, runStatus, t]);

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
          err?.message || t("FCXM.messages.could_not_read_fcs_labels")
        );
      } finally {
        if (!cancelled) setFcsDisplayNamesBusy(false);
      }
    }

    loadDisplayNames();

    return () => {
      cancelled = true;
    };
  }, [fcsSavedNames, fcsNameMode, t]);

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

    if (role === "NC") return t("FCXM.sample_types.negative_control");
    if (role === "PC") return t("FCXM.sample_types.positive_control");

    if (String(s.label || "").startsWith("Negative control")) {
      return t("FCXM.sample_types.negative_control");
    }

    if (String(s.label || "").startsWith("Positive control")) {
      return t("FCXM.sample_types.positive_control");
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
        label: line.label || t("FCXM.overview.series_fallback", { number: idx + 1 }),
        color: line.color || "#64748b",
        values: Array.isArray(line.values) ? line.values : [],
        filename,
        role: typeof line.role === "string" ? line.role : null,
        roleLabel,
        rawMedian:
          typeof line.raw_median === "number"
            ? line.raw_median
            : Array.isArray(line.values_raw)
            ? median(line.values_raw)
            : null,
        xLabel: line.x_label || null,
      };
    });
  }, [results?.line_series, t]);

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
      setPanelError(err?.message || t("FCXM.messages.panel_extraction_failed"));
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
      setRunMsg(t("FCXM.messages.panel_empty"));
      return;
    }

    const missingRequired = samples.filter(
      (s) => s.role === "NC" && s.file_paths.length === 0
    );

    if (missingRequired.length > 0) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg(
        t("FCXM.messages.missing_files_for", {
          samples: missingRequired
            .map((s) => `${s.name} (${s.role})`)
            .join(", "),
        })
      );
      return;
    }

    const hasAnyNCFile = samples.some(
      (s) => s.role === "NC" && s.file_paths.length > 0
    );

    if (!hasAnyNCFile) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg(t("FCXM.messages.nc_required"));
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
      setRunMsg(t("FCXM.messages.run_started"));

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
            setRunMsg(prog.message || t("FCXM.messages.done"));
            setRunStage(null);
            setRunCurrentFile(null);

            if (typeof prog.total_files === "number") {
              setRunDoneFiles(prog.total_files);
              setRunTotalFiles(prog.total_files);
            }

            return;
          }

          if (prog.status === "error") {
            const stage = prog.failed_stage ?? prog.stage;
            const supportId = prog.support_id ?? job_id;
          
            const details = [
              prog.error || prog.message || t("FCXM.messages.run_failed"),
              prog.error_type
                ? `${t("common.errors.type")}: ${prog.error_type}`
                : null,
              stage
                ? `${t("common.errors.stage")}: ${stage}`
                : null,
              prog.failed_file
                ? `${t("common.errors.file")}: ${prog.failed_file}`
                : null,
              `${t("common.errors.job_id")}: ${supportId}`,
            ]
              .filter(Boolean)
              .join("\n");
          
            setRunBusy(false);
            setRunMsg(details);
            setRunStage(null);
            setRunCurrentFile(null);
            return;
          }

          setTimeout(poll, 800);
        } catch (err: any) {
          setRunBusy(false);
          setRunStatus("error");
          setRunMsg(err?.message || t("FCXM.messages.progress_check_failed"));
          setRunStage(null);
          setRunCurrentFile(null);
        }
      };

      poll();
    } catch (err: any) {
      setRunBusy(false);
      setRunStatus("error");
      setRunMsg(err?.message || t("FCXM.messages.run_failed"));
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
      setSummaryError(err?.message || t("FCXM.messages.download_failed"));
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

  function sampleTypeLabel(sampleType: SampleCardModel["sampleType"]) {
    if (sampleType === "negative") return t("FCXM.sample_types.negative_control");
    if (sampleType === "positive") return t("FCXM.sample_types.positive_control");
    return t("FCXM.sample_types.sample");
  }

  function metricLabel(metric: PositivityMetric) {
    if (metric === "Median Ratio") return t("FCXM.metrics.median_ratio");
    if (metric === "Median Shift") return t("FCXM.metrics.median_shift");
    if (metric === "Fluorescence Index") {
      return t("FCXM.metrics.fluorescence_index");
    }
    return t("FCXM.metrics.percent_positive");
  }

  return (
    <div className="h-screen overflow-hidden bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar title={t("FCXM.toolbar_title")} />

      <div className="flex-1 min-h-0 overflow-hidden p-4 grid grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)] gap-4">
        <aside className="min-h-0 overflow-y-auto pr-1">
          <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
            <div className="font-medium">{t("FCXM.workflow.title")}</div>
            <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              {t("FCXM.workflow.description")}
            </div>
          </div>

          <div className="space-y-3">
            <StepButton
              number={1}
              title={t("FCXM.steps.upload.title")}
              status={setupStepStatus}
              active={activeStep === 1}
              onClick={() => goToStep(1)}
              summary={
                !hasUploadedFiles
                  ? t("FCXM.steps.upload.summary_missing")
                  : !hasAnyNC
                  ? t("FCXM.steps.upload.summary_nc_missing", {
                      count: uploadedCount,
                    })
                  : unassignedCount > 0
                  ? t("FCXM.steps.upload.summary_unassigned", {
                      uploaded: uploadedCount,
                      unassigned: unassignedCount,
                    })
                  : t("FCXM.steps.upload.summary_done", {
                      count: uploadedCount,
                    })
              }
            />

            <StepButton
              number={2}
              title={t("FCXM.steps.panel.title")}
              status={panelStepStatus}
              active={activeStep === 2}
              onClick={() => goToStep(2)}
              summary={
                panelBusy
                  ? t("FCXM.steps.panel.summary_reading")
                  : panelError
                  ? t("FCXM.steps.panel.summary_error")
                  : hasPanelRows && !panelVisited
                  ? t("FCXM.steps.panel.summary_needs_review", {
                      count: panelRows.length,
                    })
                  : hasPanelRows
                  ? t("FCXM.steps.panel.summary_done", {
                      count: panelRows.length,
                    })
                  : t("FCXM.steps.panel.summary_empty")
              }
            />

            <StepButton
              number={3}
              title={t("FCXM.steps.run.title")}
              status={runStepStatus}
              active={activeStep === 3}
              onClick={() => goToStep(3)}
              summary={
                runStatus === "done"
                  ? t("FCXM.steps.run.summary_done")
                  : runStatus === "running" || runStatus === "queued"
                  ? t("FCXM.steps.run.summary_running")
                  : hasAnyNC && hasPanelRows
                  ? t("FCXM.steps.run.summary_ready")
                  : t("FCXM.steps.run.summary_incomplete")
              }
            />

            <StepButton
              number={4}
              title={t("FCXM.steps.report.title")}
              status={reportStepStatus}
              active={activeStep === 4}
              onClick={() => goToStep(4)}
              summary={
                runStatus === "done"
                  ? t("FCXM.steps.report.summary_ready")
                  : t("FCXM.steps.report.summary_not_ready")
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
                {t("FCXM.actions.reset_experiment")}
              </button>
            </div>
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto pr-1">
          <section className={activeStep === 1 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">{t("FCXM.panels.upload.heading")}</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("FCXM.panels.upload.description")}
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 items-stretch">
              <div className="min-h-0 flex flex-col">
                <UploadCard
                  key={uploadResetKey}
                  title={t("FCXM.upload_card.title")}
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
                  uploadedListLabel={t("FCXM.upload_card.uploaded_list_label")}
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
                  <div className="text-sm font-medium mb-2">
                    {t("FCXM.display_names.title")}
                  </div>

                  <select
                    value={fcsNameMode}
                    onChange={(e) =>
                      setFcsNameMode(e.target.value as FcsDisplayNameMode)
                    }
                    className="w-full h-10 rounded-xl border px-3 bg-white
                               dark:bg-neutral-900 dark:border-neutral-700"
                  >
                    <option value="filename">
                      {t("FCXM.display_names.use_file_names")}
                    </option>
                    <option value="tube_name">
                      {t("FCXM.display_names.use_tube_names")}
                    </option>
                  </select>

                  {fcsDisplayNamesBusy ? (
                    <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                      {t("FCXM.display_names.reading")}
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
                    <h3 className="font-medium">{t("FCXM.sample_cards.title")}</h3>
                    <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
                      {t("FCXM.sample_cards.description")}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={addSampleCard}
                    className="text-sm px-3 py-1.5 rounded-lg border bg-white hover:bg-neutral-50
                               dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
                  >
                    {t("FCXM.actions.add_sample")}
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
                  {t("FCXM.sample_cards.drag_hint")}
                </p>
              </div>
            </div>
          </section>

          <section className={activeStep === 2 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">{t("FCXM.panels.panel.heading")}</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("FCXM.panels.panel.description")}
              </div>
            </div>

            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <h3 className="font-medium">{t("FCXM.panel.title")}</h3>
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  {uploadedCount
                    ? t("FCXM.panel.files_uploaded", { count: uploadedCount })
                    : t("FCXM.panel.no_files_yet")}
                </div>
              </div>

              <div className="space-y-3">
                {panelBusy ? (
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    {t("FCXM.panel.reading")}
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
                    {t("FCXM.panel.no_igg_marker")}
                  </div>
                ) : null}
              </div>
            </div>
          </section>

          <section className={activeStep === 3 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">{t("FCXM.panels.run.heading")}</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("FCXM.panels.run.description")}
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
                        ? t("FCXM.run_button_titles.panel_loading")
                        : panelError
                        ? t("FCXM.run_button_titles.fix_panel_error")
                        : panelRows.length === 0
                        ? t("FCXM.run_button_titles.load_panel_first")
                        : t("FCXM.run_button_titles.run_analysis")
                    }
                  >
                    {runBusy ? t("FCXM.actions.running") : t("FCXM.actions.run_analysis")}
                  </button>
                </div>
              </div>

              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="font-medium mb-3">{t("FCXM.samples.title")}</div>

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
                            {sampleTypeLabel(c.sampleType)}
                          </div>

                          <div className="mt-2 text-xs">
                            {t("FCXM.samples.file_count", { count: fileCount })}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[420px_minmax(0,1fr)] gap-4 items-stretch">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 h-full flex flex-col">
                  <div className="font-medium">
                    {t("FCXM.result_selection.title")}
                  </div>

                  <div className="mt-3 space-y-3">
                    <div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                        {t("FCXM.result_selection.selected_sample")}
                      </div>
                      <div className="h-11 rounded-xl border px-3 flex items-center bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-800">
                        <span className="truncate text-sm font-medium">
                          {selectedCard
                            ? selectedCard.name || selectedCard.title
                            : t("FCXM.result_selection.no_sample_selected")}
                        </span>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                        {t("FCXM.result_selection.file")}
                      </div>
                      <select
                        value={selectedFile}
                        onChange={(e) => setSelectedFile(e.target.value)}
                        disabled={!selectedCard || fileOptions.length === 0}
                        className="w-full h-11 rounded-xl border px-3 bg-white
                                   dark:bg-neutral-900 dark:border-neutral-700 disabled:opacity-50 truncate"
                      >
                        {fileOptions.length === 0 ? (
                          <option value="">
                            {t("FCXM.result_selection.no_files")}
                          </option>
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
                        {t("FCXM.result_selection.gate")}
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
                  <div className="font-medium">{t("FCXM.overview.title")}</div>
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">
                    {t("FCXM.overview.description")}
                  </div>

                  <div className="mt-3 grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_320px] gap-4">
                    <div className="min-h-0 flex flex-col gap-3">
                      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
                        {controlCurveCards.map((card) => {
                          const hasTubeName =
                            card.filename &&
                            card.label &&
                            card.label !== basename(card.filename);

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
                                {hasTubeName
                                  ? ` · ${basename(card.filename)}`
                                  : ""}
                              </div>

                              <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                                {t("FCXM.metrics.igg_median")}
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
                                prev === selectedFileCurveKey
                                  ? null
                                  : selectedFileCurveKey
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
                              <span className="truncate">
                                {t("FCXM.overview.selected_file")}
                              </span>
                            </div>

                            <div className="mt-1 text-[11px] text-neutral-500 dark:text-neutral-500 truncate">
                              {selectedFile
                                ? displayNameForFcsLoose(selectedFile)
                                : t("FCXM.empty_value")}
                              {selectedFile &&
                              displayNameForFcsLoose(selectedFile) !==
                                basename(selectedFile)
                                ? ` · ${basename(selectedFile)}`
                                : ""}
                            </div>

                            <div className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.igg_median")}
                            </div>

                            <div className="text-base font-semibold">
                              {fmtNum(results.selected_file_metrics.igg_median_raw)}
                            </div>
                          </button>
                        ) : null}

                        {selectedCard && !results ? (
                          <div className="text-neutral-600 dark:text-neutral-400 text-sm">
                            {t("FCXM.overview.no_results_yet")}
                          </div>
                        ) : null}
                      </div>

                      <div className="overflow-hidden rounded-xl border dark:border-neutral-800 p-2 h-[260px]">
                        <LinePlot
                          series={overviewLineSeries.map((s) => {
                            const isSelectedFileCurve = s.key === selectedFileCurveKey;
                        
                            return {
                              key: s.key,
                              label: s.label,
                              color: isSelectedFileCurve ? "#3b82f6" : s.color,
                              values: s.values,
                              dashed: isSelectedFileCurve,
                              foreground: isSelectedFileCurve,
                            };
                          })}
                          activeSeriesKey={activeCurveKey}
                          xLabel={t("FCXM.axes.igg")}
                          showLegend={false}
                        />
                      </div>
                    </div>

                    <div className="min-h-0 flex flex-col gap-3">
                      {results?.selected_file_metrics ? (
                        <div className="rounded-xl border p-3 dark:border-neutral-800">
                          <div className="text-sm font-medium mb-2">
                            {t("FCXM.metrics_cards.selected_file", {
                              label: results.selected_file_metrics.label,
                            })}
                          </div>

                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.events")}
                            </div>
                            <div>{results.selected_file_metrics.n_events}</div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.median_shift")}
                            </div>
                            <div>
                              {fmtNum(results.selected_file_metrics.igg_median_shift)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.median_ratio")}
                            </div>
                            <div>
                              {fmtNum(results.selected_file_metrics.igg_median_ratio)}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.fluorescence_index")}
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_file_metrics
                                  .igg_fluorescence_index
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.igg_positive_fraction")}
                            </div>
                            <div>
                              {fmtPct01(
                                results.selected_file_metrics.igg_pos_fraction
                              )}
                            </div>
                          </div>
                        </div>
                      ) : null}

                      {results?.selected_sample_metrics ? (
                        <div className="rounded-xl border p-3 dark:border-neutral-800">
                          <div className="text-sm font-medium mb-2">
                            {t("FCXM.metrics_cards.combined_sample", {
                              label: results.selected_sample_metrics.label,
                            })}
                          </div>

                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.events")}
                            </div>
                            <div>{results.selected_sample_metrics.n_events}</div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.median_shift")}
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_sample_metrics.igg_median_shift
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.median_ratio")}
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_sample_metrics.igg_median_ratio
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.fluorescence_index")}
                            </div>
                            <div>
                              {fmtNum(
                                results.selected_sample_metrics
                                  .igg_fluorescence_index
                              )}
                            </div>

                            <div className="text-neutral-600 dark:text-neutral-400">
                              {t("FCXM.metrics.igg_positive_fraction")}
                            </div>
                            <div>
                              {fmtPct01(
                                results.selected_sample_metrics.igg_pos_fraction
                              )}
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
                  <div className="font-medium">{t("FCXM.gating.title")}</div>

                  <label className="flex items-center gap-2 select-none">
                    <input
                      type="checkbox"
                      checked={showGating}
                      onChange={(e) => setShowGating(e.target.checked)}
                      className="rounded border"
                      disabled={!selectedCard}
                    />
                    <span className="text-sm">{t("FCXM.gating.show")}</span>
                  </label>
                </div>

                {showGating ? (
                  <div className="mt-3 overflow-x-auto">
                    <div className="grid grid-flow-col auto-cols-[260px] gap-3">
                      {(results?.gating_plots ?? []).map((gp, idx) => (
                        <PlotCard
                          key={gp.title + idx}
                          title={gp.title}
                          subtitle={t("FCXM.plots.vs", {
                            x: gp.x_label,
                            y: gp.y_label,
                          })}
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
                                subtitle: t("FCXM.plots.vs", {
                                  x: gp.x_label,
                                  y: gp.y_label,
                                }),
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
                          {t("FCXM.gating.no_plots_yet")}
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
              <div className="font-medium">{t("FCXM.panels.report.heading")}</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("FCXM.panels.report.description")}
              </div>
            </div>

            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
              <div className="font-medium">{t("FCXM.report_settings.title")}</div>

              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 items-end">
                <div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                    {t("FCXM.report_settings.metric")}
                  </div>

                  <select
                    value={positivityMetric}
                    onChange={(e) =>
                      setPositivityMetric(e.target.value as PositivityMetric)
                    }
                    className="w-full h-11 rounded-xl border px-3 bg-white
                               dark:bg-neutral-900 dark:border-neutral-700"
                  >
                    <option value="Median Ratio">
                      {t("FCXM.metrics.median_ratio")}
                    </option>
                    <option value="Median Shift">
                      {t("FCXM.metrics.median_shift")}
                    </option>
                    <option value="Fluorescence Index">
                      {t("FCXM.metrics.fluorescence_index")}
                    </option>
                    <option value="% pos">
                      {t("FCXM.metrics.percent_positive")}
                    </option>
                  </select>
                </div>

                <div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
                    {t("FCXM.report_settings.positive_above")}
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
                  {t("FCXM.report_settings.current_rule")}
                </div>

                <div className="mt-1 font-medium">
                  {t("FCXM.report_settings.rule", {
                    metric: metricLabel(positivityMetric),
                    threshold:
                      positivityThreshold === ""
                        ? t("FCXM.empty_value")
                        : positivityThreshold,
                  })}
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
                    ? t("FCXM.report_settings.run_must_be_done")
                    : t("FCXM.report_settings.download_summary")
                }
              >
                {summaryBusy
                  ? t("FCXM.actions.preparing_summary")
                  : t("FCXM.actions.download_summary")}
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
