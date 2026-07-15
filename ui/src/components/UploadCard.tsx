import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";

import {
  parseLayout,
  uploadWithProgress,
  type JobUploadKind,
} from "../api";


type FileWithRelativePath = File & {
  __relativePath?: string;
};


type DroppedEntry = {
  isFile: boolean;
  isDirectory: boolean;
  name?: string;
  fullPath?: string;

  file?: (
    success: (file: File) => void,
    failure?: (error: DOMException) => void
  ) => void;

  createReader?: () => {
    readEntries: (
      success: (entries: DroppedEntry[]) => void,
      failure?: (error: DOMException) => void
    ) => void;
  };
};


type EntryDataTransferItem = DataTransferItem & {
  getAsEntry?: () => DroppedEntry | null;
  webkitGetAsEntry?: () => DroppedEntry | null;
};


type UploadCardProps = {
  title: string;
  accept: string;

  ensureJobId: () => Promise<string>;
  uploadKind?: JobUploadKind;

  allowDirectory?: boolean;

  onPicked: (files: File[]) => void;

  onUploaded?: (
    serverNamesOrLayout: any[],
    jobId: string
  ) => void;

  mode?: "generic" | "excel-layout";

  showUploadedList?: boolean;
  uploadedListLabel?: string;

  assignedFilenames?: string[];
  hideAssigned?: boolean;
  hideSelectedList?: boolean;

  renderUploadedItem?: (
    filename: string
  ) => React.ReactNode;

  fileFilter?: (file: File) => boolean;

  autoUpload?: boolean;
  className?: string;
};


function attachRelativePath(
  file: File,
  relativePath: string
): File {
  const normalized = relativePath
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");

  Object.defineProperty(
    file,
    "__relativePath",
    {
      value: normalized || file.name,
      configurable: true,
      enumerable: false,
      writable: false,
    }
  );

  return file;
}


function readFileEntry(
  entry: DroppedEntry
): Promise<File> {
  return new Promise(
    (resolve, reject) => {
      if (!entry.file) {
        reject(
          new Error(
            "Dropped file entry could not be read."
          )
        );
        return;
      }

      entry.file(
        (file) => {
          const relativePath =
            entry.fullPath?.replace(
              /^\/+/,
              ""
            ) || file.name;

          resolve(
            attachRelativePath(
              file,
              relativePath
            )
          );
        },
        reject
      );
    }
  );
}


function readDirectoryBatch(
  reader: ReturnType<
    NonNullable<
      DroppedEntry["createReader"]
    >
  >
): Promise<DroppedEntry[]> {
  return new Promise(
    (resolve, reject) => {
      reader.readEntries(
        resolve,
        reject
      );
    }
  );
}


async function readDroppedEntry(
  entry: DroppedEntry
): Promise<File[]> {
  if (entry.isFile) {
    return [
      await readFileEntry(entry),
    ];
  }

  if (
    !entry.isDirectory ||
    !entry.createReader
  ) {
    return [];
  }

  const reader =
    entry.createReader();

  const children:
    DroppedEntry[] = [];

  /*
   * Chromium may return directory
   * entries in several batches.
   */
  while (true) {
    const batch =
      await readDirectoryBatch(
        reader
      );

    if (batch.length === 0) {
      break;
    }

    children.push(...batch);
  }

  const nestedFiles =
    await Promise.all(
      children.map((child) =>
        readDroppedEntry(child)
      )
    );

  return nestedFiles.flat();
}


async function getDroppedFiles(
  dataTransfer: DataTransfer,
  allowDirectory: boolean
): Promise<File[]> {
  const items = Array.from(
    dataTransfer.items || []
  ).filter(
    (item) =>
      item.kind === "file"
  );

  const entries = items
    .map((item) => {
      const entryItem =
        item as EntryDataTransferItem;

      return (
        entryItem.getAsEntry?.() ??
        entryItem.webkitGetAsEntry?.() ??
        null
      );
    })
    .filter(
      (
        entry
      ): entry is DroppedEntry =>
        entry !== null
    );

  if (entries.length > 0) {
    const acceptedEntries =
      allowDirectory
        ? entries
        : entries.filter(
            (entry) =>
              entry.isFile
          );

    const nestedFiles =
      await Promise.all(
        acceptedEntries.map(
          (entry) =>
            readDroppedEntry(entry)
        )
      );

    return nestedFiles.flat();
  }

  /*
   * Fallback for browsers without
   * FileSystemEntry support.
   *
   * Files selected through a directory
   * input retain webkitRelativePath,
   * which src/api.ts reads directly.
   */
  return Array.from(
    dataTransfer.files || []
  );
}


function containsDraggedFiles(
  dataTransfer: DataTransfer
): boolean {
  return Array.from(
    dataTransfer.types || []
  ).includes("Files");
}


function selectedFileKey(
  file: File,
  index: number
): string {
  const extended =
    file as FileWithRelativePath & {
      webkitRelativePath?: string;
    };

  const relativePath =
    extended.__relativePath ||
    extended.webkitRelativePath ||
    file.name;

  return [
    relativePath,
    file.size,
    file.lastModified,
    index,
  ].join("-");
}


export function UploadCard({
  title,
  accept,
  ensureJobId,
  uploadKind,
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
}: UploadCardProps) {
  const { t } = useTranslation();

  const [dragging, setDragging] =
    useState(false);

  const [selected, setSelected] =
    useState<File[]>([]);

  const [progress, setProgress] =
    useState<number | null>(null);

  const [message, setMessage] =
    useState<string | null>(null);

  const [uploading, setUploading] =
    useState(false);

  const [uploadedItems, setUploadedItems] =
    useState<any[]>([]);

  const fileInputRef =
    useRef<HTMLInputElement | null>(
      null
    );

  const dragDepthRef =
    useRef(0);

  const assignedSet = useMemo(
    () =>
      new Set(
        assignedFilenames
      ),
    [assignedFilenames]
  );

  const effectiveUploadedListLabel =
    uploadedListLabel ??
    t(
      "UploadCard.uploaded_list.default_label"
    );


  useEffect(() => {
    const input =
      fileInputRef.current;

    if (!input) {
      return;
    }

    if (
      mode !== "excel-layout" &&
      allowDirectory
    ) {
      input.setAttribute(
        "webkitdirectory",
        ""
      );

      input.setAttribute(
        "directory",
        ""
      );

      input.setAttribute(
        "mozdirectory",
        ""
      );

      input.setAttribute(
        "msdirectory",
        ""
      );

      input.setAttribute(
        "odirectory",
        ""
      );

      return;
    }

    input.removeAttribute(
      "webkitdirectory"
    );

    input.removeAttribute(
      "directory"
    );

    input.removeAttribute(
      "mozdirectory"
    );

    input.removeAttribute(
      "msdirectory"
    );

    input.removeAttribute(
      "odirectory"
    );
  }, [
    allowDirectory,
    mode,
  ]);


  const syncParent = useCallback(
    (files: File[]) => {
      setSelected(files);
      setUploadedItems([]);
      onPicked(files);
    },
    [onPicked]
  );


  const applyFileFilter =
    useCallback(
      (files: File[]) => {
        if (!fileFilter) {
          return files;
        }

        const kept =
          files.filter(fileFilter);

        const skipped =
          files.length -
          kept.length;

        if (skipped > 0) {
          setMessage(
            t(
              "UploadCard.messages.skipped_files",
              {
                count: skipped,
              }
            )
          );
        }

        return kept;
      },
      [
        fileFilter,
        t,
      ]
    );


  const onBrowse =
    useCallback(() => {
      fileInputRef.current?.click();
    }, []);


  const handleChange =
    useCallback(
      (
        event: React.ChangeEvent<HTMLInputElement>
      ) => {
        const rawFiles =
          event.target.files
            ? Array.from(
                event.target.files
              )
            : [];

        if (
          fileInputRef.current
        ) {
          fileInputRef.current.value =
            "";
        }

        if (
          rawFiles.length === 0
        ) {
          return;
        }

        const files =
          applyFileFilter(rawFiles);

        if (
          files.length === 0
        ) {
          syncParent([]);
          return;
        }

        syncParent(
          mode === "excel-layout"
            ? [files[0]]
            : files
        );
      },
      [
        applyFileFilter,
        mode,
        syncParent,
      ]
    );


  const onDragEnter =
    useCallback(
      (
        event: React.DragEvent<HTMLDivElement>
      ) => {
        if (
          !containsDraggedFiles(
            event.dataTransfer
          )
        ) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        dragDepthRef.current += 1;
        setDragging(true);
      },
      []
    );


  const onDragOver =
    useCallback(
      (
        event: React.DragEvent<HTMLDivElement>
      ) => {
        if (
          !containsDraggedFiles(
            event.dataTransfer
          )
        ) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        event.dataTransfer.dropEffect =
          "copy";

        setDragging(true);
      },
      []
    );


  const onDragLeave =
    useCallback(
      (
        event: React.DragEvent<HTMLDivElement>
      ) => {
        if (
          !containsDraggedFiles(
            event.dataTransfer
          )
        ) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        dragDepthRef.current =
          Math.max(
            0,
            dragDepthRef.current -
              1
          );

        if (
          dragDepthRef.current ===
          0
        ) {
          setDragging(false);
        }
      },
      []
    );


  const onDrop =
    useCallback(
      async (
        event: React.DragEvent<HTMLDivElement>
      ) => {
        if (
          !containsDraggedFiles(
            event.dataTransfer
          )
        ) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();

        dragDepthRef.current = 0;
        setDragging(false);
        setMessage(null);

        try {
          const rawFiles =
            await getDroppedFiles(
              event.dataTransfer,
              Boolean(
                allowDirectory
              )
            );

          if (
            rawFiles.length === 0
          ) {
            return;
          }

          const files =
            applyFileFilter(
              rawFiles
            );

          if (
            files.length === 0
          ) {
            syncParent([]);
            return;
          }

          syncParent(
            mode ===
              "excel-layout"
              ? [files[0]]
              : files
          );
        } catch (error) {
          setMessage(
            error instanceof Error
              ? error.message
              : t(
                  "UploadCard.messages.upload_failed"
                )
          );
        }
      },
      [
        allowDirectory,
        applyFileFilter,
        mode,
        syncParent,
        t,
      ]
    );


  const startUpload =
    useCallback(async () => {
      if (
        selected.length === 0 ||
        uploading
      ) {
        return;
      }

      setUploading(true);
      setProgress(0);
      setMessage(null);

      try {
        const jobId =
          await ensureJobId();

        if (
          mode ===
          "excel-layout"
        ) {
          const file =
            selected[0];

          const layout =
            await parseLayout(
              jobId,
              file
            );

          const wells =
            Object.keys(
              layout.wells || {}
            ).length;

          setMessage(
            t(
              "UploadCard.messages.layout_loaded",
              {
                lot:
                  layout.lot_no ??
                  t(
                    "UploadCard.empty_value"
                  ),
                wells,
              }
            )
          );

          onUploaded?.(
            [layout],
            jobId
          );

          return;
        }

        if (!uploadKind) {
          throw new Error(
            "UploadCard requires uploadKind for generic file uploads."
          );
        }

        const names =
          await uploadWithProgress(
            jobId,
            uploadKind,
            selected,
            (percent) =>
              setProgress(
                percent
              )
          );

        setUploadedItems(
          names || []
        );

        setMessage(
          t(
            "UploadCard.messages.uploaded_files",
            {
              count: names.length,
            }
          )
        );

        onUploaded?.(
          names,
          jobId
        );
      } catch (error) {
        setMessage(
          error instanceof Error
            ? error.message
            : t(
                "UploadCard.messages.upload_failed"
              )
        );
      } finally {
        setUploading(false);

        window.setTimeout(
          () =>
            setProgress(null),
          600
        );
      }
    }, [
      ensureJobId,
      mode,
      onUploaded,
      selected,
      t,
      uploadKind,
      uploading,
    ]);


  useEffect(() => {
    if (!autoUpload) {
      return;
    }

    if (uploading) {
      return;
    }

    if (
      selected.length === 0
    ) {
      return;
    }

    if (
      uploadedItems.length >
      0
    ) {
      return;
    }

    void startUpload();
  }, [
    autoUpload,
    selected,
    startUpload,
    uploadedItems.length,
    uploading,
  ]);


  const removeAt =
    useCallback(
      (index: number) => {
        if (uploading) {
          return;
        }

        const next =
          selected.filter(
            (_, currentIndex) =>
              currentIndex !==
              index
          );

        syncParent(next);
      },
      [
        selected,
        syncParent,
        uploading,
      ]
    );


  const multiple =
    mode !== "excel-layout";

  const hideSelectedNow =
    hideSelectedList &&
    mode !== "excel-layout" &&
    uploadedItems.length > 0;


  return (
    <div
      className={[
        "rounded-2xl border p-4 bg-white dark:bg-neutral-900 dark:border-neutral-800",
        "flex flex-col min-h-0",
        dragging
          ? "ring-2 ring-blue-500"
          : "",
        className,
      ].join(" ")}
      onDragEnter={
        onDragEnter
      }
      onDragOver={onDragOver}
      onDragLeave={
        onDragLeave
      }
      onDrop={onDrop}
    >
      <div className="flex items-center justify-between gap-3 mb-2">
        <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
          {title}
        </h3>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onBrowse}
            disabled={
              uploading
            }
            className="text-sm px-3 py-1.5 rounded-lg border bg-white hover:bg-neutral-50
                       disabled:opacity-50 disabled:cursor-not-allowed
                       dark:bg-neutral-900 dark:hover:bg-neutral-800
                       dark:border-neutral-700 dark:text-neutral-200"
          >
            {t(
              "UploadCard.actions.browse"
            )}
          </button>

          {!autoUpload ? (
            <button
              type="button"
              onClick={() =>
                void startUpload()
              }
              disabled={
                uploading ||
                selected.length ===
                  0
              }
              className={[
                "text-sm px-3 py-1.5 rounded-lg border",
                uploading ||
                selected.length ===
                  0
                  ? "opacity-50 cursor-not-allowed"
                  : "bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800",
                "dark:border-neutral-700 dark:text-neutral-200",
              ].join(" ")}
              title={
                selected.length ===
                0
                  ? t(
                      "UploadCard.actions.select_file_first"
                    )
                  : t(
                      "UploadCard.actions.upload_selected"
                    )
              }
            >
              {uploading
                ? t(
                    "UploadCard.actions.uploading"
                  )
                : mode ===
                  "excel-layout"
                ? t(
                    "UploadCard.actions.load_layout"
                  )
                : t(
                    "UploadCard.actions.upload"
                  )}
            </button>
          ) : null}
        </div>
      </div>

      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-3">
        {mode ===
        "excel-layout"
          ? t(
              "UploadCard.instructions.excel_layout"
            )
          : t(
              "UploadCard.instructions.generic",
              {
                target:
                  allowDirectory
                    ? t(
                        "UploadCard.instructions.folder"
                      )
                    : t(
                        "UploadCard.instructions.files"
                      ),
              }
            )}
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept={
          mode ===
          "excel-layout"
            ? [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel.sheet.macroEnabled.12",
                ".xlsx",
                ".xlsm",
              ].join(",")
            : accept
        }
        multiple={multiple}
        className="hidden"
        onChange={
          handleChange
        }
      />

      {progress !== null ? (
        <div className="mt-3">
          <div className="h-2 w-full bg-neutral-200 dark:bg-neutral-800 rounded">
            <div
              className="h-2 rounded bg-blue-500 dark:bg-sky-500 transition-[width]"
              style={{
                width: `${progress}%`,
              }}
            />
          </div>

          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            {progress}%
          </div>
        </div>
      ) : null}

      {message ? (
        <div className="mt-2 text-sm text-neutral-700 dark:text-neutral-300 whitespace-pre-wrap break-words">
          {message}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-auto">
        {!hideSelectedNow &&
        selected.length > 0 ? (
          <div className="mt-2 flex-1 min-h-0 overflow-auto rounded-md border dark:border-neutral-800">
            <ul className="text-xs divide-y dark:divide-neutral-800">
              {selected.map(
                (
                  file,
                  index
                ) => (
                  <li
                    key={selectedFileKey(
                      file,
                      index
                    )}
                    className="flex items-center justify-between gap-3 px-2 py-1"
                  >
                    <span className="truncate">
                      {(
                        file as FileWithRelativePath & {
                          webkitRelativePath?: string;
                        }
                      )
                        .__relativePath ||
                        (
                          file as File & {
                            webkitRelativePath?: string;
                          }
                        )
                          .webkitRelativePath ||
                        file.name}
                    </span>

                    <button
                      type="button"
                      aria-label={t(
                        "UploadCard.actions.remove_named",
                        {
                          file:
                            file.name,
                        }
                      )}
                      title={t(
                        "UploadCard.actions.remove"
                      )}
                      onClick={() =>
                        removeAt(
                          index
                        )
                      }
                      onKeyDown={(
                        event
                      ) => {
                        if (
                          event.key ===
                            "Enter" ||
                          event.key ===
                            " "
                        ) {
                          event.preventDefault();

                          removeAt(
                            index
                          );
                        }
                      }}
                      className="inline-flex items-center justify-center w-5 h-5 rounded
                                 border text-neutral-600 hover:bg-neutral-100
                                 dark:border-neutral-700 dark:text-neutral-300
                                 dark:hover:bg-neutral-800"
                    >
                      ×
                    </button>
                  </li>
                )
              )}
            </ul>
          </div>
        ) : null}

        {showUploadedList &&
        mode !==
          "excel-layout" &&
        uploadedItems.length >
          0 ? (
          <div className="mt-3 rounded-md border dark:border-neutral-800">
            <div className="px-2 py-1 text-xs text-neutral-600 dark:text-neutral-400">
              {t(
                "UploadCard.uploaded_list.heading",
                {
                  label:
                    effectiveUploadedListLabel,
                }
              )}
            </div>

            <ul className="text-xs divide-y dark:divide-neutral-800">
              {uploadedItems
                .map(
                  (item: any) =>
                    typeof item ===
                    "string"
                      ? item
                      : item?.filename
                )
                .filter(Boolean)
                .filter(
                  (
                    filename: string
                  ) =>
                    !(
                      hideAssigned &&
                      assignedSet.has(
                        filename
                      )
                    )
                )
                .map(
                  (
                    filename: string
                  ) => (
                    <li
                      key={
                        filename
                      }
                      className={[
                        "px-2 py-1 cursor-grab active:cursor-grabbing",
                        assignedSet.has(
                          filename
                        )
                          ? "opacity-60"
                          : "",
                      ].join(" ")}
                      draggable
                      onDragStart={(
                        event
                      ) => {
                        const payload =
                          JSON.stringify(
                            {
                              fname:
                                filename,
                              fromCardId:
                                null,
                            }
                          );

                        event.dataTransfer.setData(
                          "application/x-allocviewer-fcsref",
                          payload
                        );

                        event.dataTransfer.setData(
                          "application/x-allocviewer-filename",
                          filename
                        );

                        event.dataTransfer.setData(
                          "text/plain",
                          filename
                        );

                        event.dataTransfer.effectAllowed =
                          "copyMove";
                      }}
                      title={
                        assignedSet.has(
                          filename
                        )
                          ? t(
                              "UploadCard.uploaded_list.already_assigned"
                            )
                          : t(
                              "UploadCard.uploaded_list.drag"
                            )
                      }
                    >
                      {renderUploadedItem ? (
                        renderUploadedItem(
                          filename
                        )
                      ) : (
                        <div className="truncate">
                          {
                            filename
                          }
                        </div>
                      )}
                    </li>
                  )
                )}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  );
}
