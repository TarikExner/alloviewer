export type RunStatus = "idle" | "queued" | "running" | "done" | "error";

export function StatusPill({
  status,
  message,
  jobId,
  busy,
  error,
  progressPercent,
  currentFile,
}: {
  status: RunStatus;
  message?: string | null;
  jobId?: string | null;
  busy?: boolean;
  error?: string | null;
  progressPercent?: number | null;
  currentFile?: string | null;
}) {
  const label =
    status === "queued"
      ? "Queued"
      : status === "running"
      ? "Running"
      : status === "done"
      ? "Completed"
      : status === "error"
      ? "Error"
      : "Idle";

  const dotClass =
    status === "done"
      ? "bg-emerald-500"
      : status === "error" || error
      ? "bg-red-500"
      : status === "running" || status === "queued" || busy
      ? "bg-blue-500"
      : "bg-neutral-400";

  const textClass =
    status === "done"
      ? "text-emerald-700 dark:text-emerald-400"
      : status === "error" || error
      ? "text-red-700 dark:text-red-400"
      : "text-neutral-700 dark:text-neutral-300";

  return (
    <div className="min-w-0 rounded-xl border bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-800 px-3 py-2">
      <div className="flex items-center gap-3">
        <span
          className={[
            "inline-block w-2.5 h-2.5 rounded-full shrink-0",
            dotClass,
            status === "running" || status === "queued" || busy
              ? "animate-pulse"
              : "",
          ].join(" ")}
        />

        <div className="min-w-0 flex-1">
          <div className={["text-sm font-medium", textClass].join(" ")}>
            {label}
          </div>

          {message || error ? (
            <div className="text-xs text-neutral-600 dark:text-neutral-400 truncate">
              {error || message}
            </div>
          ) : null}

          {currentFile && status === "running" ? (
            <div className="text-[11px] text-neutral-500 dark:text-neutral-500 truncate">
              Current file: {currentFile}
            </div>
          ) : null}
        </div>

        {typeof progressPercent === "number" ? (
          <div className="shrink-0 text-xs text-neutral-600 dark:text-neutral-400">
            {Math.round(progressPercent)}%
          </div>
        ) : jobId ? (
          <div
            className="shrink-0 max-w-[120px] truncate text-[11px] text-neutral-500 dark:text-neutral-500 font-mono"
            title={jobId}
          >
            job {jobId.slice(0, 8)}
          </div>
        ) : null}
      </div>

      {typeof progressPercent === "number" ? (
        <div className="mt-2 h-2 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-all"
            style={{
              width: `${Math.max(0, Math.min(100, progressPercent))}%`,
            }}
          />
        </div>
      ) : null}

      {jobId && typeof progressPercent === "number" ? (
        <div
          className="mt-1 truncate text-[11px] text-neutral-500 dark:text-neutral-500 font-mono"
          title={jobId}
        >
          job {jobId.slice(0, 8)}
        </div>
      ) : null}
    </div>
  );
}
