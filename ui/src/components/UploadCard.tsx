import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { uploadWithProgress } from "../api";
import { parseLayout } from "../api";

export function UploadCard({
  title,
  accept,
  allowDirectory,
  onPicked,
  onUploaded,
  mode = "generic",
  showUploadedList = false,
  uploadedListLabel,
  assignedFilenames = [],
  hideAssigned = false,
  hideSelectedList = false,
  renderUploadedItem,
  fileFilter,
  autoUpload = false,
  className = "",
}: {
  title: string;
  accept: string;
  allowDirectory?: boolean;
  onPicked: (files: File[]) => void;
  onUploaded?: (serverNamesOrLayout: any[]) => void;
  mode?: "generic" | "excel-layout";
  showUploadedList?: boolean;
  uploadedListLabel?: string;
  assignedFilenames?: string[];
  hideAssigned?: boolean;
  hideSelectedList?: boolean;
  renderUploadedItem?: (fname: string) => React.ReactNode;
  fileFilter?: (file: File) => boolean;
  autoUpload?: boolean;
  className?: string;
}) {
  const { t } = useTranslation();

  const [dragging, setDragging] = useState(false);
  const [selected, setSelected] = useState<File[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadedItems, setUploadedItems] = useState<any[]>([]);

  const assignedSet = new Set(assignedFilenames);
  const effectiveUploadedListLabel =
    uploadedListLabel ?? t("UploadCard.uploaded_list.default_label");

  useEffect(() => {
    if (mode === "excel-layout") return;

    if (allowDirectory && fileInputRef.current) {
      fileInputRef.current.setAttribute("webkitdirectory", "");
      fileInputRef.current.setAttribute("directory", "");
      fileInputRef.current.setAttribute("mozdirectory", "");
      fileInputRef.current.setAttribute("msdirectory", "");
      fileInputRef.current.setAttribute("odirectory", "");
    }
  }, [allowDirectory, mode]);

  const syncParent = useCallback(
    (files: File[]) => {
      setSelected(files);
      setUploadedItems([]);
      setMessage(null);
      onPicked(files);
    },
    [onPicked]
  );

  const applyFileFilter = useCallback(
    (files: File[]) => {
      if (!fileFilter) return files;

      const kept = files.filter(fileFilter);
      const skipped = files.length - kept.length;

      if (skipped > 0) {
        setMessage(t("UploadCard.messages.skipped_files", { count: skipped }));
      }

      return kept;
    },
    [fileFilter, t]
  );

  const onBrowse = useCallback(() => fileInputRef.current?.click(), []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const rawFiles = e.target.files ? Array.from(e.target.files) : [];

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      if (!rawFiles.length) return;

      const files = applyFileFilter(rawFiles);
      if (!files.length) {
        syncParent([]);
        return;
      }

      syncParent(mode === "excel-layout" ? [files[0]] : files);
    },
    [syncParent, mode, applyFileFilter]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragging(false);

      const rawFiles = Array.from(e.dataTransfer.files || []);
      if (!rawFiles.length) return;

      const files = applyFileFilter(rawFiles);
      if (!files.length) {
        syncParent([]);
        return;
      }

      syncParent(mode === "excel-layout" ? [files[0]] : files);
    },
    [syncParent, mode, applyFileFilter]
  );

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const startUpload = useCallback(async () => {
    if (!selected.length || uploading) return;

    setUploading(true);
    setProgress(0);
    setMessage(null);

    try {
      if (mode === "excel-layout") {
        const file = selected[0];
        const layout = await parseLayout(file);
        const wells = Object.keys(layout.wells || {}).length;

        setMessage(
          t("UploadCard.messages.layout_loaded", {
            lot: layout.lot_no ?? t("UploadCard.empty_value"),
            wells,
          })
        );

        onUploaded?.([layout]);
      } else {
        const names = await uploadWithProgress(selected, (p) => setProgress(p));

        setUploadedItems(names || []);
        setMessage(
          t("UploadCard.messages.uploaded_files", {
            count: names.length,
          })
        );
        onUploaded?.(names);
      }
    } catch (err: any) {
      setMessage(err?.message || t("UploadCard.messages.upload_failed"));
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(null), 600);
    }
  }, [selected, uploading, onUploaded, mode, t]);

  useEffect(() => {
    if (!autoUpload) return;
    if (uploading) return;
    if (!selected.length) return;
    if (uploadedItems.length > 0) return;

    startUpload();
  }, [autoUpload, selected, uploading, uploadedItems.length, startUpload]);

  const removeAt = useCallback(
    (idx: number) => {
      if (uploading) return;

      const next = selected.filter((_, i) => i !== idx);
      syncParent(next);
    },
    [selected, uploading, syncParent]
  );

  const multiple = mode !== "excel-layout";

  const hideSelectedNow =
    hideSelectedList && mode !== "excel-layout" && uploadedItems.length > 0;

  return (
    <div
      className={[
        "rounded-2xl border p-4 bg-white dark:bg-neutral-900 dark:border-neutral-800",
        "flex flex-col min-h-0",
        dragging ? "ring-2 ring-blue-500" : "",
        className,
      ].join(" ")}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
          {title}
        </h3>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onBrowse}
            disabled={uploading}
            className="text-sm px-3 py-1.5 rounded-lg border bg-white hover:bg-neutral-50
                       disabled:opacity-50 disabled:cursor-not-allowed
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
          >
            {t("UploadCard.actions.browse")}
          </button>

          {!autoUpload ? (
            <button
              type="button"
              onClick={startUpload}
              disabled={uploading || selected.length === 0}
              className={[
                "text-sm px-3 py-1.5 rounded-lg border",
                uploading || selected.length === 0
                  ? "opacity-50 cursor-not-allowed"
                  : "bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800",
                "dark:border-neutral-700 dark:text-neutral-200",
              ].join(" ")}
              title={
                selected.length === 0
                  ? t("UploadCard.actions.select_file_first")
                  : t("UploadCard.actions.upload_selected")
              }
            >
              {uploading
                ? t("UploadCard.actions.uploading")
                : mode === "excel-layout"
                ? t("UploadCard.actions.load_layout")
                : t("UploadCard.actions.upload")}
            </button>
          ) : null}
        </div>
      </div>

      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-3">
        {mode === "excel-layout"
          ? t("UploadCard.instructions.excel_layout")
          : t("UploadCard.instructions.generic", {
              target: allowDirectory
                ? t("UploadCard.instructions.folder")
                : t("UploadCard.instructions.files"),
            })}
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept={
          mode === "excel-layout"
            ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,.xlsx,.xlsm"
            : accept
        }
        multiple={multiple}
        className="hidden"
        onChange={handleChange}
      />

      <div className="flex-1 min-h-0 overflow-auto">
        {!hideSelectedNow && selected.length > 0 && (
          <div className="mt-2 flex-1 min-h-0 overflow-auto rounded-md border dark:border-neutral-800">
            <ul className="text-xs divide-y dark:divide-neutral-800">
              {selected.map((f, i) => (
                <li
                  key={f.name + i}
                  className="flex items-center justify-between gap-3 px-2 py-1"
                >
                  <span className="truncate">{f.name}</span>

                  <button
                    type="button"
                    aria-label={t("UploadCard.actions.remove_named", {
                      file: f.name,
                    })}
                    title={t("UploadCard.actions.remove")}
                    onClick={() => removeAt(i)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        removeAt(i);
                      }
                    }}
                    className="inline-flex items-center justify-center w-5 h-5 rounded
                               border text-neutral-600 hover:bg-neutral-100
                               dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {showUploadedList &&
          mode !== "excel-layout" &&
          uploadedItems.length > 0 && (
            <div className="mt-3 rounded-md border dark:border-neutral-800">
              <div className="px-2 py-1 text-xs text-neutral-600 dark:text-neutral-400">
                {t("UploadCard.uploaded_list.heading", {
                  label: effectiveUploadedListLabel,
                })}
              </div>

              <ul className="text-xs divide-y dark:divide-neutral-800">
                {uploadedItems
                  .map((x: any) => (typeof x === "string" ? x : x?.filename))
                  .filter(Boolean)
                  .filter(
                    (fname: string) => !(hideAssigned && assignedSet.has(fname))
                  )
                  .map((fname: string) => (
                    <li
                      key={fname}
                      className={[
                        "px-2 py-1 cursor-grab active:cursor-grabbing",
                        assignedSet.has(fname) ? "opacity-60" : "",
                      ].join(" ")}
                      draggable
                      onDragStart={(e) => {
                        const payload = JSON.stringify({
                          fname,
                          fromCardId: null,
                        });
                        e.dataTransfer.setData(
                          "application/x-allocviewer-fcsref",
                          payload
                        );
                        e.dataTransfer.setData(
                          "application/x-allocviewer-filename",
                          fname
                        );
                        e.dataTransfer.setData("text/plain", fname);
                        e.dataTransfer.effectAllowed = "copyMove";
                      }}
                      title={
                        assignedSet.has(fname)
                          ? t("UploadCard.uploaded_list.already_assigned")
                          : t("UploadCard.uploaded_list.drag")
                      }
                    >
                      {renderUploadedItem ? (
                        renderUploadedItem(fname)
                      ) : (
                        <div className="truncate">{fname}</div>
                      )}
                    </li>
                  ))}
              </ul>
            </div>
          )}

        {progress !== null && (
          <div className="mt-3">
            <div className="h-2 w-full bg-neutral-200 dark:bg-neutral-800 rounded">
              <div
                className="h-2 rounded bg-blue-500 dark:bg-sky-500 transition-[width]"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
              {progress}%
            </div>
          </div>
        )}

        {message && (
          <div className="mt-2 text-sm text-neutral-700 dark:text-neutral-300">
            {message}
          </div>
        )}
      </div>
    </div>
  );
}
