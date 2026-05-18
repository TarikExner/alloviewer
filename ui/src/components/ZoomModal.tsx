import React from "react";

export function ZoomModal({
  open,
  onClose,
  title,
  subtitle,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      onMouseDown={onClose}
    >
      <div className="absolute inset-0 bg-black/40" />

      <div
        className="relative w-full max-w-5xl max-h-[90vh] rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 shadow-lg overflow-hidden flex flex-col"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="p-4 border-b dark:border-neutral-800 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="font-semibold truncate">{title}</div>
            {subtitle ? (
              <div className="text-sm text-neutral-600 dark:text-neutral-400 truncate">
                {subtitle}
              </div>
            ) : null}
          </div>

          <button
            type="button"
            onClick={onClose}
            className="text-sm px-3 py-1.5 rounded-lg border bg-white hover:bg-neutral-50
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
          >
            Close
          </button>
        </div>

        <div className="p-4 flex-1 min-h-0 overflow-hidden">
          <div className="w-full h-full flex items-center justify-center">
            <div className="aspect-square w-full max-w-[80vh] max-h-[80vh] min-h-[360px]">
              {children}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
