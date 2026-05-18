export type StepState =
  | "not_started"
  | "needs_attention"
  | "needs_review"
  | "ready"
  | "running"
  | "done"
  | "error";

function StepBadge({ status }: { status: StepState }) {
  const label =
    status === "not_started"
      ? "Not started"
      : status === "needs_attention"
      ? "Needs attention"
      : status === "needs_review"
      ? "Needs review"
      : status === "ready"
      ? "Ready"
      : status === "running"
      ? "Running"
      : status === "error"
      ? "Error"
      : "Done";

  const cls =
    status === "done"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300"
      : status === "error" || status === "needs_attention"
      ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300"
      : status === "needs_review"
      ? "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300"
      : status === "running"
      ? "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300"
      : status === "ready"
      ? "border-neutral-300 bg-white text-neutral-700 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200"
      : "border-neutral-200 bg-neutral-50 text-neutral-500 dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-400";

  return (
    <span className={["text-xs px-2 py-1 rounded-full border", cls].join(" ")}>
      {label}
    </span>
  );
}

export function StepButton({
  number,
  title,
  summary,
  status,
  active,
  onClick,
}: {
  number: number;
  title: string;
  summary: string;
  status: StepState;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "w-full rounded-2xl border p-3 text-left transition-colors",
        active
          ? "bg-white dark:bg-neutral-900 border-blue-300 dark:border-blue-600 ring-2 ring-blue-500"
          : "bg-white dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-800",
      ].join(" ")}
    >
      <div className="flex items-start gap-3">
        <div className="shrink-0 w-7 h-7 rounded-full border flex items-center justify-center text-sm font-semibold bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-700">
          {number}
        </div>

        <div className="min-w-0 flex-1">
          <div className="font-medium">{title}</div>
          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            {summary}
          </div>

          <div className="mt-2">
            <StepBadge status={status} />
          </div>
        </div>
      </div>
    </button>
  );
}
