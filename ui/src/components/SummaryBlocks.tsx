import React from "react";

export function formatSummaryValue(value: any): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return String(Math.round(value * 1000) / 1000);
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  return value ?? "—";
}

export function SummaryItem({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded-xl border p-3 dark:border-neutral-800">
      <div className="text-xs text-neutral-600 dark:text-neutral-400">
        {label}
      </div>

      <div className="mt-1 text-lg font-semibold">
        {formatSummaryValue(value)}
      </div>
    </div>
  );
}

export function SummarySection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4">
      <div className="font-medium mb-3">{title}</div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        {children}
      </div>
    </div>
  );
}

export function WarningList({
  title,
  warnings,
}: {
  title: string;
  warnings?: string[];
}) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300 p-3 text-sm">
      <div className="font-medium mb-1">{title}</div>

      <ul className="list-disc pl-5 space-y-1">
        {warnings.map((w, idx) => (
          <li key={`${w}-${idx}`}>{w}</li>
        ))}
      </ul>
    </div>
  );
}

export function WellList({
  title,
  wells,
}: {
  title: string;
  wells?: string[];
}) {
  if (!wells || wells.length === 0) return null;

  return (
    <div className="rounded-xl border p-3 dark:border-neutral-800 text-sm">
      <div className="font-medium mb-2">{title}</div>

      <div className="flex flex-wrap gap-1.5">
        {wells.map((w) => (
          <span
            key={w}
            className="inline-flex items-center rounded-md border px-2 py-0.5 text-xs bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-700"
          >
            {w}
          </span>
        ))}
      </div>
    </div>
  );
}

export function EmptySummary() {
  return (
    <div className="rounded-xl border p-3 dark:border-neutral-800 text-sm text-neutral-600 dark:text-neutral-400">
      No summary yet. Run the analysis first.
    </div>
  );
}
