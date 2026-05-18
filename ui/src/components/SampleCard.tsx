import React from "react";

export type SampleType = "negative" | "positive" | "sample";

export type SampleCardModel = {
  id: string;
  sampleType: SampleType;
  title: string; // fallback label
  name: string; // shown in the top input
  fcsFiles: string[];
};

export function SampleCard({
  card,
  selected,
  onSelect,
  onRemoveCard,
  onNameChange,
  onDragOverFiles,
  onDropFile,
  onRemoveFile,
  fileDisplayNames = {},
}: {
  card: SampleCardModel;
  selected: boolean;
  onSelect: () => void;

  onRemoveCard?: () => void;
  onNameChange: (next: string) => void;

  onDragOverFiles: (e: React.DragEvent) => void;
  onDropFile: (e: React.DragEvent) => void;

  onRemoveFile: (fname: string) => void;

  /**
   * Maps the internal saved FCS filename to the label shown in the UI.
   * The internal value remains fname, so backend calls stay stable.
   */
  fileDisplayNames?: Record<string, string>;
}) {
  const nameLocked = card.sampleType !== "sample";

  function displayNameForFcs(fname: string) {
    return fileDisplayNames[fname] || fname;
  }

  return (
    <button
      type="button"
      onClick={onSelect}
      className={[
        "w-full text-left rounded-2xl border p-3",
        "bg-neutral-50 dark:bg-neutral-950 dark:border-neutral-800",
        "hover:bg-neutral-100 dark:hover:bg-neutral-900",
        "transition-colors",
        selected
          ? "ring-2 ring-blue-500 border-blue-300 dark:border-blue-500"
          : "border-neutral-200",
      ].join(" ")}
      title="Click to view results"
    >
      {/* Top row: Name input + optional Remove */}
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <div
            className={[
              "rounded-xl border px-3 py-2",
              "bg-white dark:bg-neutral-900 dark:border-neutral-700",
              nameLocked ? "opacity-90" : "",
            ].join(" ")}
            onClick={(e) => {
              if (!nameLocked) e.stopPropagation();
            }}
          >
            <input
              value={card.name}
              readOnly={nameLocked}
              onChange={(e) => onNameChange(e.target.value)}
              onClick={(e) => {
                if (!nameLocked) e.stopPropagation();
              }}
              className={[
                "w-full bg-transparent outline-none",
                "text-base font-bold",
                nameLocked ? "cursor-default" : "cursor-text",
              ].join(" ")}
              placeholder={card.title}
            />
          </div>
        </div>

        {card.sampleType === "sample" && onRemoveCard && (
          <span
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onRemoveCard();
            }}
            className="text-xs px-2 py-1 rounded-lg border cursor-pointer whitespace-nowrap
                       bg-white hover:bg-neutral-50
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
            title="Remove sample"
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                e.stopPropagation();
                onRemoveCard();
              }
            }}
          >
            Remove
          </span>
        )}
      </div>

      {/* FCS box */}
      <div className="mt-3">
        <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
          FCS files:
        </div>

        <div
          onClick={(e) => e.stopPropagation()}
          onDragOver={onDragOverFiles}
          onDrop={onDropFile}
          className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-700
                     px-3 py-2 min-h-[64px] max-h-[110px] overflow-auto"
          title="Drag uploaded FCS files here"
        >
          {card.fcsFiles.length === 0 ? (
            <div className="text-sm text-neutral-500 dark:text-neutral-400">
              Drag FCS files here after uploading
            </div>
          ) : (
            <ul className="space-y-1">
              {card.fcsFiles.map((fname) => {
                const displayName = displayNameForFcs(fname);
                const hasDisplayName = displayName !== fname;

                return (
                  <li
                    key={fname}
                    className="flex items-center justify-between gap-2 text-sm"
                    title={hasDisplayName ? fname : displayName}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate">{displayName}</div>

                      {hasDisplayName ? (
                        <div className="truncate text-[11px] text-neutral-500 dark:text-neutral-500">
                          {fname}
                        </div>
                      ) : null}
                    </div>

                    <button
                      type="button"
                      onClick={() => onRemoveFile(fname)}
                      className="inline-flex shrink-0 items-center justify-center w-6 h-6 rounded-lg border
                                 bg-white hover:bg-neutral-50
                                 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
                      title="Remove file"
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </button>
  );
}
