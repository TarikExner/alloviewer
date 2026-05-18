import React, { useEffect, useMemo, useState } from "react";
import { useParams, Navigate } from "react-router-dom";
import { Toolbar } from "../components/Toolbar";
import { UploadCard } from "../components/UploadCard";
import PlateEditorWithOrder from "../components/PlateEditorWithOrder";
import { PlatePreview } from "../components/PlatePreview";
import { ALL_WELLS, type ProcessResponse, type WellID, type WellMap } from "../types";
import { ASSAYS } from "../assays";

function defaultSumm(proc: any): Record<string, number> | null {
  if (!proc) return null;
  if (proc.summary) return proc.summary;
  if (proc.wells && typeof proc.wells === "object") {
    const vals = Object.values(proc.wells) as any[];
    const m = (xs: number[]) =>
      xs.length ? Math.round((xs.reduce((a, b) => a + b, 0) / xs.length) * 1000) / 1000 : 0;
    const fracBy = (prefix: string) =>
      vals.filter((w: any) => (w.role ?? "").toLowerCase().startsWith(prefix)).map((w: any) => Number(w.frac_pos ?? 0));
    return {
      total_wells: vals.length,
      pos_mean: m(fracBy("pos")),
      neg_mean: m(fracBy("neg")),
      sample_mean: m(fracBy("sa")),
      ok: vals.length,
      warn: 0,
      fail: 0,
    };
  }
  return null;
}

export default function AssayFlow() {
  const { assayId } = useParams();
  const config = assayId && ASSAYS[assayId as keyof typeof ASSAYS];
  if (!config) return <Navigate to="/" replace />;

  // NEW: parsed Excel layout + its upload_id (we use this as templateFilename in /api/process)
  const [layout, setLayout] = useState<any | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);

  // existing state
  const [templateFiles, setTemplateFiles] = useState<File[]>([]);
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [templateSavedName, setTemplateSavedName] = useState<string | null>(null);
  const [imageSavedNames, setImageSavedNames] = useState<string[]>([]);
  const [wells, setWells] = useState<WellMap>({} as WellMap);
  const [scanOrder, setScanOrder] = useState<WellID[]>([]);
  const [proc, setProc] = useState<ProcessResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // initialize wells with first role (usually "sample")
  useEffect(() => {
    setWells(Object.fromEntries(ALL_WELLS.map((w) => [w, config.roles[0] ?? "sample"])) as WellMap);
  }, [config]);

  const imageURLs = useMemo(() => imageFiles.map((f) => URL.createObjectURL(f)), [imageFiles]);
  useEffect(() => () => imageURLs.forEach((u) => URL.revokeObjectURL(u)), [imageURLs]);

  const imagesByWell = useMemo(() => {
    const map: Record<WellID, string | null> = Object.create(null);
    ALL_WELLS.forEach((w, i) => {
      map[w] = imageURLs[i] || null;
    });
    return map;
  }, [imageURLs]);

  const imageOrder: WellID[] = useMemo(() => {
    const isNonEmpty = (w: WellID) => (wells as any)[w] && (wells as any)[w] !== "empty";
    return scanOrder.filter(isNonEmpty);
  }, [scanOrder, wells]);

  async function onRun() {
    setBusy(true);
    setMsg(null);
    setProc(null);
    try {
      // per-assay validator if provided
      const err = config.validate?.(wells, imageOrder);
      if (err) {
        setMsg(err);
        setBusy(false);
        return;
      }
      // IMPORTANT: pass uploadId (from parsed Excel) as templateFilename
      const data = await config.run(wells, imageOrder, {
        templateFilename: uploadId,            // << use the parsed layout "upload_id"
        imageFilenames: imageSavedNames,
      });
      setProc(data);
      setMsg("Analysis done.");
    } catch (err: any) {
      setMsg(err.message || "Process failed");
    } finally {
      setBusy(false);
    }
  }

  // OLD: required templateSavedName (from /api/upload)
  // NEW: require uploadId (from /api/plate-layouts/parse)
  const canRun = !busy && !!uploadId && imageSavedNames.length > 0 && imageOrder.length > 0;

  const summarize = config.summarize ?? defaultSumm;
  const summary = summarize(proc);

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar title={config.title} />

      <div className="flex-1 p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* LEFT */}
        <div className="space-y-4 min-h-0 flex flex-col">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {/* Excel layout uploader (parse on backend) */}
            <UploadCard
              title="Plate layout (.xlsx)"
              accept="" // ignored in excel mode
              allowDirectory={false}
              mode="excel-layout"
              onPicked={() => {}}
              onUploaded={([parsed]) => {
                setLayout(parsed);
                setUploadId(parsed.upload_id);
              }}
            />
            {/* Images (to /api/upload) */}
            <UploadCard
              title={config.upload.imagesLabel}
              accept={config.upload.acceptImages}
              allowDirectory
              onPicked={setImageFiles}
              onUploaded={setImageSavedNames}
            />
          </div>

          <div className="flex-1 min-h-0">
            <PlateEditorWithOrder
              wells={wells}
              setWells={setWells}
              onOrderChange={setScanOrder}
              roles={config.roles}
              plate={config.plate}
            />
          </div>

          <button
            onClick={onRun}
            disabled={!canRun}
            className="w-full py-2.5 rounded-xl border bg-white hover:bg-neutral-50 disabled:opacity-50
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
            title={!canRun ? "Upload layout + images and set order first" : "Run analysis"}
          >
            {busy ? "Running…" : "Run Analysis"}
          </button>
          {msg && <p className="text-sm text-neutral-700 dark:text-neutral-300">{msg}</p>}
        </div>

        {/* RIGHT */}
        <div className="min-h-0 flex flex-col">
          <PlatePreview imagesByWell={imagesByWell} result={proc} summary={summary} layout={layout as any} />
        </div>
      </div>
    </div>
  );
}

