import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Toolbar } from "../components/Toolbar";
import { UploadCard } from "../components/UploadCard";
import PlateEditorWithOrder from "../components/PlateEditorWithOrder";
import { PlatePreview } from "../components/PlatePreview";
import { StepButton, type StepState } from "../components/StepButton";
import { StatusPill, type RunStatus } from "../components/StatusPill";
import { ImageMappingTable } from "../components/ImageMappingTable";
import { CrossmatchSummaryGrid } from "../components/PlateAssaySummaries";
import {
  ALL_WELLS,
  type ProcessResponse,
  type WellID,
  type WellMap,
} from "../types";
import {
  runProcess,
  fetchProgress,
  downloadCDCSummaryPdf,
  type BackendProgress,
} from "../api/crossmatch";
import { API_BASE } from "../App";
import {
  normalizeSavedNames,
  sameFiles,
  sameStringArray,
} from "../lib/upload";
import {
  buildDefaultColumnModes,
  buildDefaultCrossmatch,
  type CellMode,
} from "../lib/crossmatchDefaults";
import {
  buildImageOrder,
  buildImagesByWell,
  buildInitialWellStatus,
  buildThumbnailUrls,
  buildWellToFileMap,
  clampPercent,
  computeSummary,
  countWellsByType,
  extractImageScores,
  type WellRunStatus,
} from "../lib/plateAssay";

type ActiveStep = 1 | 2 | 3 | 4;

export default function CrossmatchApp() {
  const { t } = useTranslation();

  const [flip, setFlip] = useState(true);

  const [columnModes, setColumnModes] = useState<Record<number, CellMode>>(() =>
    buildDefaultColumnModes()
  );

  const [uploadResetKey, setUploadResetKey] = useState(0);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imageSavedNames, setImageSavedNames] = useState<string[]>([]);

  const [wells, setWells] = useState<WellMap>({} as WellMap);
  const [scanOrder, setScanOrder] = useState<WellID[]>([]);
  const [imageScores, setImageScores] = useState<Record<string, number>>({});
  const [proc, setProc] = useState<ProcessResponse | null>(null);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const [wellStatus, setWellStatus] = useState<Record<WellID, WellRunStatus>>(
    {} as any
  );

  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<RunStatus>("idle");
  const [jobStage, setJobStage] = useState<string | null>(null);

  const [activeStep, setActiveStep] = useState<ActiveStep>(1);
  const [plateVisited, setPlateVisited] = useState(false);
  const [autoJumpedToPlate, setAutoJumpedToPlate] = useState(false);

  const [processJobId, setProcessJobId] = useState<string | null>(null);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  useEffect(() => {
    setWells(buildDefaultCrossmatch());
  }, []);

  const handleImagesPicked = useCallback((files: File[]) => {
    const validFiles = files.filter((file) => file.type.startsWith("image/"));

    setImageFiles((prev) => (sameFiles(prev, validFiles) ? prev : validFiles));
  }, []);

  const handleImagesUploaded = useCallback((saved: any[]) => {
    const names = normalizeSavedNames(saved);

    setImageSavedNames((prev) => (sameStringArray(prev, names) ? prev : names));
    setPlateVisited(false);
    setAutoJumpedToPlate(false);
  }, []);

  const imageOrder = useMemo(
    () => buildImageOrder(scanOrder, wells),
    [scanOrder, wells]
  );

  const imageURLs = useMemo(
    () => buildThumbnailUrls(imageSavedNames, API_BASE),
    [imageSavedNames]
  );

  const imagesByWell = useMemo(
    () => buildImagesByWell(ALL_WELLS, imageOrder, imageURLs),
    [imageOrder, imageURLs]
  );

  const summary = computeSummary(proc);

  const hasImages = imageSavedNames.length > 0;
  const hasOrder = imageOrder.length > 0;

  const sampleCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "sample"),
    [wells]
  );

  const negativeCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "negative"),
    [wells]
  );

  const positiveCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "positive"),
    [wells]
  );

  const igmCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "igm"),
    [wells]
  );

  const mappedImageCount = Math.min(imageSavedNames.length, imageOrder.length);
  const unmappedImageCount = Math.max(
    0,
    imageSavedNames.length - imageOrder.length
  );
  const missingImageCount = Math.max(
    0,
    imageOrder.length - imageSavedNames.length
  );

  const uploadStepStatus: StepState = !hasImages ? "not_started" : "done";

  const plateStepStatus: StepState =
    !hasImages
      ? "not_started"
      : !hasOrder
      ? "needs_attention"
      : !plateVisited
      ? "needs_review"
      : missingImageCount > 0
      ? "needs_attention"
      : "done";

  const runStepStatus: StepState =
    jobStatus === "error"
      ? "error"
      : jobStatus === "queued" || jobStatus === "running" || busy
      ? "running"
      : jobStatus === "done"
      ? "done"
      : hasImages && hasOrder && missingImageCount === 0
      ? "ready"
      : "not_started";

  const reportStepStatus: StepState =
    jobStatus === "done" && summary ? "ready" : "not_started";

  const canRun =
    !busy &&
    hasImages &&
    hasOrder &&
    imageOrder.length > 0 &&
    missingImageCount === 0;

  const canDownloadSummary =
    jobStatus === "done" && !!processJobId && !summaryBusy;

  useEffect(() => {
    if (autoJumpedToPlate) return;
    if (!hasImages) return;
    if (!hasOrder) return;
    if (activeStep !== 1) return;

    setActiveStep(2);
    setPlateVisited(true);
    setAutoJumpedToPlate(true);
  }, [autoJumpedToPlate, hasImages, hasOrder, activeStep]);

  function goToStep(step: ActiveStep) {
    setActiveStep(step);
    if (step === 2) setPlateVisited(true);
  }

  function handleColumnMode(col: number, mode: CellMode) {
    setColumnModes((prev) => ({ ...prev, [col]: mode }));
  }

  function resetExperiment() {
    setUploadResetKey((x) => x + 1);

    setFlip(true);
    setColumnModes(buildDefaultColumnModes());

    setImageFiles([]);
    setImageSavedNames([]);

    setWells(buildDefaultCrossmatch());
    setScanOrder([]);
    setImageScores({});
    setProc(null);

    setBusy(false);
    setMsg(null);

    setWellStatus({} as any);
    setProgressPercent(null);
    setJobStatus("idle");
    setJobStage(null);

    setActiveStep(1);
    setPlateVisited(false);
    setAutoJumpedToPlate(false);

    setProcessJobId(null);
    setSummaryBusy(false);
    setSummaryError(null);
  }

  async function onDownloadSummary() {
    if (!processJobId) return;

    setSummaryBusy(true);
    setSummaryError(null);

    try {
      await downloadCDCSummaryPdf(processJobId);
    } catch (err: any) {
      setSummaryError(err?.message || "Could not download summary PDF.");
    } finally {
      setSummaryBusy(false);
    }
  }

  async function onRun() {
    setActiveStep(3);
    setBusy(true);
    setMsg(null);
    setProc(null);

    setWellStatus(buildInitialWellStatus(imageOrder));
    setProgressPercent(0);
    setJobStatus("queued");
    setImageScores({});
    setJobStage(null);

    setProcessJobId(null);
    setSummaryBusy(false);
    setSummaryError(null);

    const wellToFileAtRun = buildWellToFileMap(imageOrder, imageSavedNames);

    try {
      const { job_id } = await runProcess(wells, imageOrder, {
        templateFilename: null,
        imageFilenames: imageSavedNames,
      });

      setProcessJobId(job_id);

      const poll = async () => {
        try {
          const prog: BackendProgress = await fetchProgress(job_id);

          setJobStatus(prog.status as RunStatus);
          setJobStage(prog.stage ?? null);

          const pct = clampPercent(prog.done, prog.total);
          if (pct !== null) setProgressPercent(pct);

          setWellStatus((prev) => {
            const next = { ...prev };

            if (prog.done_wells) {
              prog.done_wells.forEach((w: WellID) => {
                next[w as WellID] = "done";
              });
            }

            if (prog.current_well) {
              next[prog.current_well as WellID] = "running";
            }

            return next;
          });

          if (prog.status === "done") {
            if (prog.result) {
              setProc(prog.result as ProcessResponse);
              setMsg(t("cdc_xm_app.messages.analysis_done"));
              setImageScores(extractImageScores(prog.result, wellToFileAtRun));
            } else {
              setMsg(t("cdc_xm_app.messages.analysis_done_no_result"));
            }

            setBusy(false);
            setProgressPercent(100);
            setJobStage(null);
            return;
          }

          if (prog.status === "error") {
            setMsg(t("cdc_xm_app.messages.process_failed"));
            setBusy(false);
            setProgressPercent(null);
            setJobStage(null);
            return;
          }

          setTimeout(poll, 800);
        } catch (err: any) {
          setMsg(err.message || t("cdc_xm_app.messages.process_failed"));
          setBusy(false);
          setJobStatus("error");
          setProgressPercent(null);
          setJobStage(null);
        }
      };

      poll();
    } catch (err: any) {
      setMsg(err.message || t("cdc_xm_app.messages.process_failed"));
      setBusy(false);
      setJobStatus("error");
      setProgressPercent(null);
    }
  }

  const stageMessage = jobStage
    ? t(`cdc_xm_app.stage_messages.${jobStage}`, {
        defaultValue: t("cdc_xm_app.stage_messages.default"),
      })
    : null;

  const statusMessage =
    jobStatus === "queued"
      ? t("cdc_xm_app.status.waiting_to_start")
      : jobStatus === "running"
      ? stageMessage || t("cdc_xm_app.status.processing_plate_images")
      : jobStatus === "done"
      ? msg || t("cdc_xm_app.status.analysis_done")
      : jobStatus === "error"
      ? msg || t("cdc_xm_app.status.process_failed")
      : msg;

  const tColumns = Object.values(columnModes).filter(
    (mode) => mode === "T"
  ).length;

  const bColumns = Object.values(columnModes).filter(
    (mode) => mode === "B"
  ).length;

  const tbColumns = Object.values(columnModes).filter(
    (mode) => mode === "T/B"
  ).length;

  return (
    <div className="h-screen overflow-hidden bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar title={t("cdc_xm_app.toolbar_title")} />

      <div className="flex-1 min-h-0 overflow-hidden p-4 grid grid-cols-1 xl:grid-cols-[280px_minmax(0,1fr)] gap-4">
        <aside className="min-h-0 overflow-y-auto pr-1">
          <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
            <div className="font-medium">{t("cdc_xm_app.workflow.title")}</div>
            <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
              {t("cdc_xm_app.workflow.description")}
            </div>
          </div>

          <div className="space-y-3">
            <StepButton
              number={1}
              title={t("cdc_xm_app.steps.upload.title")}
              status={uploadStepStatus}
              active={activeStep === 1}
              onClick={() => goToStep(1)}
              summary={
                !hasImages
                  ? t("cdc_xm_app.steps.upload.summary_missing")
                  : t("cdc_xm_app.steps.upload.summary_done", {
                      count: imageSavedNames.length,
                    })
              }
            />

            <StepButton
              number={2}
              title={t("cdc_xm_app.steps.review.title")}
              status={plateStepStatus}
              active={activeStep === 2}
              onClick={() => goToStep(2)}
              summary={
                !hasImages
                  ? t("cdc_xm_app.steps.review.summary_upload_first")
                  : !hasOrder
                  ? t("cdc_xm_app.steps.review.summary_set_scan_order")
                  : missingImageCount > 0
                  ? t("cdc_xm_app.steps.review.summary_missing_images", {
                      count: missingImageCount,
                    })
                  : t("cdc_xm_app.steps.review.summary_done", {
                      count: mappedImageCount,
                    })
              }
            />

            <StepButton
              number={3}
              title={t("cdc_xm_app.steps.run.title")}
              status={runStepStatus}
              active={activeStep === 3}
              onClick={() => goToStep(3)}
              summary={
                jobStatus === "done"
                  ? t("cdc_xm_app.steps.run.summary_done")
                  : jobStatus === "running" || jobStatus === "queued"
                  ? t("cdc_xm_app.steps.run.summary_running")
                  : canRun
                  ? t("cdc_xm_app.steps.run.summary_ready")
                  : t("cdc_xm_app.steps.run.summary_incomplete")
              }
            />

            <StepButton
              number={4}
              title={t("cdc_xm_app.steps.summary.title")}
              status={reportStepStatus}
              active={activeStep === 4}
              onClick={() => goToStep(4)}
              summary={
                jobStatus === "done"
                  ? t("cdc_xm_app.steps.summary.summary_ready")
                  : t("cdc_xm_app.steps.summary.summary_not_ready")
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
                {t("cdc_xm_app.actions.reset_experiment")}
              </button>
            </div>
          </div>
        </aside>

        <main className="min-h-0 overflow-y-auto pr-1">
          <section className={activeStep === 1 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">
                {t("cdc_xm_app.panels.upload.heading")}
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("cdc_xm_app.panels.upload.description")}
              </div>
            </div>

            <UploadCard
              key={`images-${uploadResetKey}`}
              title={t("cdc_xm_app.upload_cards.images.title")}
              accept="image/*"
              allowDirectory
              autoUpload
              fileFilter={(file) => file.type.startsWith("image/")}
              onPicked={handleImagesPicked}
              onUploaded={handleImagesUploaded}
              showUploadedList
              uploadedListLabel={t("cdc_xm_app.upload_cards.images.list_label")}
              hideSelectedList={true}
              className="h-[420px]"
            />

            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  {t("cdc_xm_app.stats.images_uploaded")}
                </div>
                <div className="mt-1 font-medium">{imageSavedNames.length}</div>
              </div>

              <div className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  {t("cdc_xm_app.stats.local_picked_files")}
                </div>
                <div className="mt-1 font-medium">{imageFiles.length}</div>
              </div>
            </div>
          </section>

          <section className={activeStep === 2 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">
                {t("cdc_xm_app.panels.review.heading")}
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("cdc_xm_app.panels.review.description")}
              </div>
            </div>

            <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_420px] gap-4 items-start">
              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <PlateEditorWithOrder
                  wells={wells}
                  setWells={setWells}
                  onOrderChange={setScanOrder}
                  flipVertical={flip}
                  onFlipChange={setFlip}
                  roleOptions={["sample", "positive", "negative", "igm", "empty"]}
                  buildDefault={buildDefaultCrossmatch}
                  columnModes={columnModes}
                  onColumnModeChange={handleColumnMode}
                />
              </div>

              <div className="2xl:h-0 2xl:min-h-full min-h-0 flex flex-col gap-4 overflow-hidden">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 shrink-0">
                  <div className="font-medium">
                    {t("cdc_xm_app.plate_checks.title")}
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.plate_checks.sample_wells")}
                      </div>
                      <div className="mt-1 font-semibold">{sampleCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.plate_checks.negative_controls")}
                      </div>
                      <div className="mt-1 font-semibold">{negativeCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.plate_checks.positive_controls")}
                      </div>
                      <div className="mt-1 font-semibold">{positiveCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.plate_checks.igm_wells")}
                      </div>
                      <div className="mt-1 font-semibold">{igmCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800 col-span-2">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.plate_checks.scan_order_wells")}
                      </div>
                      <div className="mt-1 font-semibold">{imageOrder.length}</div>
                    </div>
                  </div>

                  {negativeCount === 0 ? (
                    <div className="mt-3 text-xs text-red-700 dark:text-red-400 rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 px-3 py-2">
                      {t("cdc_xm_app.warnings.no_negative_controls")}
                    </div>
                  ) : null}

                  {positiveCount === 0 ? (
                    <div className="mt-3 text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                      {t("cdc_xm_app.warnings.no_positive_controls")}
                    </div>
                  ) : null}

                  {igmCount === 0 ? (
                    <div className="mt-3 text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                      {t("cdc_xm_app.warnings.no_igm_wells")}
                    </div>
                  ) : null}

                  {missingImageCount > 0 ? (
                    <div className="mt-3 text-xs text-red-700 dark:text-red-400 rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 px-3 py-2">
                      {t("cdc_xm_app.warnings.missing_ordered_images", {
                        count: missingImageCount,
                      })}
                    </div>
                  ) : null}

                  {unmappedImageCount > 0 ? (
                    <div className="mt-3 text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                      {t("cdc_xm_app.warnings.unmapped_images", {
                        count: unmappedImageCount,
                      })}
                    </div>
                  ) : null}
                </div>

                <ImageMappingTable
                  imageOrder={imageOrder}
                  imageSavedNames={imageSavedNames}
                />
              </div>
            </div>
          </section>

          <section className={activeStep === 3 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">
                {t("cdc_xm_app.panels.run.heading")}
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("cdc_xm_app.panels.run.description")}
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_220px] gap-3 items-center">
                  <StatusPill
                    status={jobStatus}
                    message={statusMessage}
                    busy={busy}
                    progressPercent={progressPercent}
                  />

                  <button
                    type="button"
                    onClick={onRun}
                    disabled={!canRun}
                    className="w-full py-3 rounded-xl border bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:hover:bg-blue-600
                               dark:border-blue-500"
                    title={
                      !hasImages
                        ? t("cdc_xm_app.run_button_titles.upload_images_first")
                        : !hasOrder
                        ? t("cdc_xm_app.run_button_titles.set_scan_order_first")
                        : missingImageCount > 0
                        ? t("cdc_xm_app.run_button_titles.missing_ordered_images")
                        : t("cdc_xm_app.run_button_titles.run_analysis")
                    }
                  >
                    {busy
                      ? t("cdc_xm_app.actions.running")
                      : t("cdc_xm_app.actions.run_analysis")}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[360px_minmax(0,1fr)] gap-4">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                  <div className="font-medium">
                    {t("cdc_xm_app.run_setup.title")}
                  </div>

                  <div className="mt-3 space-y-3 text-sm">
                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.run_setup.images_uploaded")}
                      </div>
                      <div className="mt-1 font-semibold">
                        {imageSavedNames.length}
                      </div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.run_setup.wells_to_process")}
                      </div>
                      <div className="mt-1 font-semibold">{imageOrder.length}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        {t("cdc_xm_app.run_setup.cell_mode_columns")}
                      </div>
                      <div className="mt-1 text-sm">
                        {t("cdc_xm_app.cell_modes.t")}: {tColumns} ·{" "}
                        {t("cdc_xm_app.cell_modes.b")}: {bColumns} ·{" "}
                        {t("cdc_xm_app.cell_modes.tb")}: {tbColumns}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="min-h-[620px]">
                  <PlatePreview
                    imagesByWell={imagesByWell}
                    result={proc}
                    flipVertical={flip}
                    wellStatus={wellStatus}
                    progressPercent={progressPercent ?? undefined}
                    jobStatus={jobStatus}
                    imageScores={imageScores}
                    key={flip ? "prev-flip-1" : "prev-flip-0"}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className={activeStep === 4 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">
                {t("cdc_xm_app.panels.summary.heading")}
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                {t("cdc_xm_app.panels.summary.description")}
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="flex items-center justify-between gap-3 mb-3">
                  <div>
                    <div className="font-medium">
                      {t("cdc_xm_app.summary_values.title")}
                    </div>

                    {summaryError ? (
                      <div className="mt-1 text-xs text-red-600 dark:text-red-400">
                        {summaryError}
                      </div>
                    ) : null}
                  </div>

                  <button
                    type="button"
                    onClick={onDownloadSummary}
                    disabled={!canDownloadSummary}
                    className="shrink-0 rounded-xl border px-3 py-2 text-sm
                               bg-white hover:bg-neutral-50 disabled:opacity-50
                               dark:bg-neutral-900 dark:hover:bg-neutral-800
                               dark:border-neutral-700 dark:text-neutral-200"
                  >
                    {summaryBusy ? "Preparing PDF..." : "Download Summary"}
                  </button>
                </div>

                <CrossmatchSummaryGrid summary={summary} />
              </div>

              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="font-medium mb-3">
                  {t("cdc_xm_app.plate_result.title")}
                </div>

                <div className="min-h-[620px]">
                  <PlatePreview
                    imagesByWell={imagesByWell}
                    result={proc}
                    flipVertical={flip}
                    wellStatus={wellStatus}
                    progressPercent={progressPercent ?? undefined}
                    jobStatus={jobStatus}
                    imageScores={imageScores}
                    key={flip ? "summary-prev-flip-1" : "summary-prev-flip-0"}
                  />
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
