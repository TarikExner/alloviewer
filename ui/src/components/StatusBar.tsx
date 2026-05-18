export function StatusBar({ label, percent }: { label: string; percent: number | null }) {
  if (percent === null) return null;
  return (
    <div className="mt-2">
      <div className="flex items-center justify-between text-xs text-neutral-600 dark:text-neutral-400 mb-1">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 bg-neutral-200 dark:bg-neutral-800 rounded-full overflow-hidden">
        <div className="h-full bg-blue-500" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

