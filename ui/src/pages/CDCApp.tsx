import { useCallback, useEffect, useMemo, useState } from "react";
import { Toolbar } from "../components/Toolbar";
import { UploadCard } from "../components/UploadCard";
import PlateEditorWithOrder from "../components/PlateEditorWithOrder";
import { PlatePreview } from "../components/PlatePreview";
import { StepButton, type StepState } from "../components/StepButton";
import { StatusPill, type RunStatus } from "../components/StatusPill";
import { ImageMappingTable } from "../components/ImageMappingTable";
import { PRASummaryGrid } from "../components/PlateAssaySummaries";
import {
  ALL_WELLS,
  type ProcessResponse,
  type WellID,
  type WellMap,
} from "../types";
import { runProcess, fetchProgress, type BackendProgress } from "../api/cdc";
import { API_BASE } from "../App";
import {
  getUploadedFilename,
  normalizeSavedNames,
  sameFiles,
  sameStringArray,
} from "../lib/upload";
import { buildDefaultCDC } from "../lib/cdcDefaults";
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
  plateStageMessage,
  type WellRunStatus,
} from "../lib/plateAssay";

type ActiveStep = 1 | 2 | 3 | 4;

export default function CDCApp() {
  const [flip, setFlip] = useState(true);

  const [layout, setLayout] = useState<any | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [uploadResetKey, setUploadResetKey] = useState(0);

  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imageSavedNames, setImageSavedNames] = useState<string[]>([]);
  const [wells, setWells] = useState<WellMap>({} as WellMap);
  const [scanOrder, setScanOrder] = useState<WellID[]>([]);
  const [proc, setProc] = useState<ProcessResponse | null>(null);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const [wellStatus, setWellStatus] = useState<Record<WellID, WellRunStatus>>(
    {} as any
  );

  const [imageScores, setImageScores] = useState<Record<string, number>>({});
  const [progressPercent, setProgressPercent] = useState<number | null>(null);

  const [jobStatus, setJobStatus] = useState<RunStatus>("idle");
  const [jobStage, setJobStage] = useState<string | null>(null);

  const [activeStep, setActiveStep] = useState<ActiveStep>(1);
  const [plateVisited, setPlateVisited] = useState(false);
  const [autoJumpedToPlate, setAutoJumpedToPlate] = useState(false);

  useEffect(() => {
    setWells(buildDefaultCDC());
  }, []);

  const handleLayoutUploaded = useCallback((saved: any[]) => {
    const first = saved?.[0] ?? null;
    const filename = getUploadedFilename(first);

    setUploadId(filename);
    setLayout(first);

    setPlateVisited(false);
    setAutoJumpedToPlate(false);
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

  const hasLayout = !!layout || !!uploadId;
  const hasImages = imageSavedNames.length > 0;
  const hasOrder = imageOrder.length > 0;

  const negativeCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "negative"),
    [wells]
  );

  const positiveCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "positive"),
    [wells]
  );

  const sampleCount = useMemo(
    () => countWellsByType(ALL_WELLS, wells, "sample"),
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

  const uploadStepStatus: StepState =
    !hasLayout && !hasImages
      ? "not_started"
      : !hasLayout || !hasImages
      ? "needs_attention"
      : "done";

  const plateStepStatus: StepState =
    !hasLayout || !hasImages
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
      : hasLayout && hasImages && hasOrder && missingImageCount === 0
      ? "ready"
      : "not_started";

  const reportStepStatus: StepState =
    jobStatus === "done" && summary ? "ready" : "not_started";

  const canRun =
    !busy &&
    hasLayout &&
    hasImages &&
    hasOrder &&
    imageOrder.length > 0 &&
    missingImageCount === 0;

  useEffect(() => {
    if (autoJumpedToPlate) return;
    if (!hasLayout) return;
    if (!hasImages) return;
    if (!hasOrder) return;
    if (activeStep !== 1) return;

    setActiveStep(2);
    setPlateVisited(true);
    setAutoJumpedToPlate(true);
  }, [autoJumpedToPlate, hasLayout, hasImages, hasOrder, activeStep]);

  function goToStep(step: ActiveStep) {
    setActiveStep(step);
    if (step === 2) setPlateVisited(true);
  }

  function resetExperiment() {
    setUploadResetKey((x) => x + 1);

    setFlip(true);
    setLayout(null);
    setUploadId(null);

    setImageFiles([]);
    setImageSavedNames([]);
    setWells(buildDefaultCDC());
    setScanOrder([]);
    setProc(null);

    setBusy(false);
    setMsg(null);

    setWellStatus({} as any);
    setImageScores({});
    setProgressPercent(null);
    setJobStatus("idle");
    setJobStage(null);

    setActiveStep(1);
    setPlateVisited(false);
    setAutoJumpedToPlate(false);
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

    const wellToFileAtRun = buildWellToFileMap(imageOrder, imageSavedNames);

    try {
      const { job_id } = await runProcess(wells, imageOrder, {
        templateFilename: uploadId,
        imageFilenames: imageSavedNames,
        assayType: "pra",
      });

      const poll = async () => {
        try {
          const prog: BackendProgress = await fetchProgress(job_id);

          setJobStage(prog.stage ?? null);
          setJobStatus(prog.status as RunStatus);

          const pct = clampPercent(prog.done, prog.total);
          if (pct !== null) setProgressPercent(pct);

          setWellStatus((prev) => {
            const next = { ...prev };

            if (prog.done_wells) {
              prog.done_wells.forEach((w) => {
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
              setMsg("Analysis done.");
              setImageScores(extractImageScores(prog.result, wellToFileAtRun));
            } else {
              setMsg("Analysis finished, but no result was returned.");
            }

            setBusy(false);
            setProgressPercent(100);
            setJobStage(null);
            return;
          }

          if (prog.status === "error") {
            setMsg("Process failed.");
            setBusy(false);
            setProgressPercent(null);
            setJobStage(null);
            return;
          }

          setTimeout(poll, 800);
        } catch (err: any) {
          setMsg(err.message || "Process failed");
          setBusy(false);
          setJobStatus("error");
          setProgressPercent(null);
          setJobStage(null);
        }
      };

      poll();
    } catch (err: any) {
      setMsg(err.message || "Process failed");
      setBusy(false);
      setJobStatus("error");
      setProgressPercent(null);
    }
  }

  const stageMessage = plateStageMessage(jobStage);

  const statusMessage =
    jobStatus === "queued"
      ? "Waiting to start."
      : jobStatus === "running"
      ? stageMessage || "Processing plate images."
      : jobStatus === "done"
      ? msg || "Analysis done."
      : jobStatus === "error"
      ? msg || "Process failed."
      : msg;

  return (
    <div className="h-screen overflow-hidden bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar title="AlloViewer - CDC panel reactivity assay" />

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
              title="Upload files"
              status={uploadStepStatus}
              active={activeStep === 1}
              onClick={() => goToStep(1)}
              summary={
                !hasLayout && !hasImages
                  ? "Upload layout and image folder"
                  : !hasLayout
                  ? `${imageSavedNames.length} image(s), layout missing`
                  : !hasImages
                  ? "Layout uploaded, images missing"
                  : `${imageSavedNames.length} image(s), layout uploaded`
              }
            />

            <StepButton
              number={2}
              title="Review plate"
              status={plateStepStatus}
              active={activeStep === 2}
              onClick={() => goToStep(2)}
              summary={
                !hasLayout || !hasImages
                  ? "Upload files first"
                  : !hasOrder
                  ? "Set scan order"
                  : missingImageCount > 0
                  ? `${missingImageCount} well(s) without image`
                  : `${mappedImageCount} image(s) mapped`
              }
            />

            <StepButton
              number={3}
              title="Run and results"
              status={runStepStatus}
              active={activeStep === 3}
              onClick={() => goToStep(3)}
              summary={
                jobStatus === "done"
                  ? "Analysis completed"
                  : jobStatus === "running" || jobStatus === "queued"
                  ? "Analysis running"
                  : canRun
                  ? "Ready to run"
                  : "Setup incomplete"
              }
            />

            <StepButton
              number={4}
              title="Summary"
              status={reportStepStatus}
              active={activeStep === 4}
              onClick={() => goToStep(4)}
              summary={
                jobStatus === "done"
                  ? "Review summary values"
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
              <div className="font-medium">Step 1 · Upload files</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Upload the plate layout and the image folder before checking the plate.
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <UploadCard
                key={`layout-${uploadResetKey}`}
                title="Plate layout (.xlsx)"
                accept=".xlsx,.xls"
                allowDirectory={false}
                mode="excel-layout"
                fileFilter={(file) => /\.(xlsx|xls)$/i.test(file.name)}
                onPicked={() => {}}
                onUploaded={handleLayoutUploaded}
                className="h-[320px]"
              />

              <UploadCard
                key={`images-${uploadResetKey}`}
                title="Images (Folder)"
                accept="image/*"
                allowDirectory
                autoUpload
                fileFilter={(file) => file.type.startsWith("image/")}
                onPicked={handleImagesPicked}
                onUploaded={handleImagesUploaded}
                showUploadedList
                uploadedListLabel="Uploaded images"
                hideSelectedList={true}
                className="h-[320px]"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  Layout
                </div>
                <div className="mt-1 font-medium">
                  {hasLayout ? "Uploaded" : "Missing"}
                </div>
              </div>

              <div className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  Images
                </div>
                <div className="mt-1 font-medium">{imageSavedNames.length}</div>
              </div>

              <div className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-3">
                <div className="text-xs text-neutral-600 dark:text-neutral-400">
                  Local picked files
                </div>
                <div className="mt-1 font-medium">{imageFiles.length}</div>
              </div>
            </div>
          </section>

          <section className={activeStep === 2 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">
                Step 2 · Review plate and scan order
              </div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Check well roles, scan order, and image-to-well mapping before running.
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
                  roleOptions={["sample", "positive", "negative", "empty"]}
                  buildDefault={buildDefaultCDC}
                />
              </div>

              <div className="2xl:h-0 2xl:min-h-full min-h-0 flex flex-col gap-4 overflow-hidden">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 shrink-0">
                  <div className="font-medium">Plate checks</div>

                  <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Sample wells
                      </div>
                      <div className="mt-1 font-semibold">{sampleCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Negative controls
                      </div>
                      <div className="mt-1 font-semibold">{negativeCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Positive controls
                      </div>
                      <div className="mt-1 font-semibold">{positiveCount}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Scan order wells
                      </div>
                      <div className="mt-1 font-semibold">{imageOrder.length}</div>
                    </div>
                  </div>

                  {negativeCount === 0 ? (
                    <div className="mt-3 text-xs text-red-700 dark:text-red-400 rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 px-3 py-2">
                      No negative control wells are selected.
                    </div>
                  ) : null}

                  {positiveCount === 0 ? (
                    <div className="mt-3 text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                      No positive control wells are selected.
                    </div>
                  ) : null}

                  {missingImageCount > 0 ? (
                    <div className="mt-3 text-xs text-red-700 dark:text-red-400 rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 px-3 py-2">
                      {missingImageCount} ordered well(s) do not have an image.
                    </div>
                  ) : null}

                  {unmappedImageCount > 0 ? (
                    <div className="mt-3 text-xs text-amber-700 dark:text-amber-400 rounded-xl border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950 px-3 py-2">
                      {unmappedImageCount} uploaded image(s) are not mapped to wells.
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
              <div className="font-medium">Step 3 · Run and inspect results</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Start the analysis, then inspect the plate preview and well status.
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
                      !hasLayout
                        ? "Upload the Excel layout first"
                        : !hasImages
                        ? "Upload images first"
                        : !hasOrder
                        ? "Set scan order first"
                        : missingImageCount > 0
                        ? "Some ordered wells do not have images"
                        : "Run analysis"
                    }
                  >
                    {busy ? "Running…" : "Run analysis"}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[360px_minmax(0,1fr)] gap-4">
                <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                  <div className="font-medium">Run setup</div>

                  <div className="mt-3 space-y-3 text-sm">
                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Images uploaded
                      </div>
                      <div className="mt-1 font-semibold">
                        {imageSavedNames.length}
                      </div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Wells to process
                      </div>
                      <div className="mt-1 font-semibold">{imageOrder.length}</div>
                    </div>

                    <div className="rounded-xl border p-3 dark:border-neutral-800">
                      <div className="text-xs text-neutral-600 dark:text-neutral-400">
                        Template
                      </div>
                      <div className="mt-1 font-semibold truncate">
                        {uploadId || "None"}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="min-h-[620px]">
                  <PlatePreview
                    imagesByWell={imagesByWell}
                    result={proc}
                    layout={layout as any}
                    flipVertical={flip}
                    wellStatus={wellStatus}
                    progressPercent={progressPercent ?? undefined}
                    jobStatus={jobStatus}
                    imageScores={imageScores}
                  />
                </div>
              </div>
            </div>
          </section>

          <section className={activeStep === 4 ? "block" : "hidden"}>
            <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 mb-4">
              <div className="font-medium">Step 4 · Summary</div>
              <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
                Review the final run summary after analysis.
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="font-medium mb-3">Summary values</div>
                <PRASummaryGrid summary={summary} />
              </div>

              <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
                <div className="font-medium mb-3">Plate result</div>

                <div className="min-h-[620px]">
                  <PlatePreview
                    imagesByWell={imagesByWell}
                    result={proc}
                    layout={layout as any}
                    flipVertical={flip}
                    wellStatus={wellStatus}
                    progressPercent={progressPercent ?? undefined}
                    jobStatus={jobStatus}
                    imageScores={imageScores}
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
