import React from "react";
import { useTranslation } from "react-i18next";

export type SampleType = "negative" | "positive" | "sample";

export type SampleCardModel = {
  id: string;
  sampleType: SampleType;
  title: string;
  name: string;
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

  fileDisplayNames?: Record<string, string>;
}) {
  const { t } = useTranslation();
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
      title={t("SampleCard.actions.view_results")}
    >
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
              aria-label={t("SampleCard.fields.sample_name")}
            />
          </div>
        </div>

        {card.sampleType === "sample" && onRemoveCard ? (
          <span
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onRemoveCard();
            }}
            className="text-xs px-2 py-1 rounded-lg border cursor-pointer whitespace-nowrap
                       bg-white hover:bg-neutral-50
                       dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
            title={t("SampleCard.actions.remove_sample")}
            role="button"
            tabIndex={0}
            aria-label={t("SampleCard.actions.remove_sample")}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                e.stopPropagation();
                onRemoveCard();
              }
            }}
          >
            {t("SampleCard.actions.remove")}
          </span>
        ) : null}
      </div>

      <div className="mt-3">
        <div className="text-xs text-neutral-600 dark:text-neutral-400 mb-1">
          {t("SampleCard.fields.fcs_files")}
        </div>

        <div
          onClick={(e) => e.stopPropagation()}
          onDragOver={onDragOverFiles}
          onDrop={onDropFile}
          className="rounded-xl border bg-white dark:bg-neutral-900 dark:border-neutral-700
                     px-3 py-2 min-h-[64px] max-h-[110px] overflow-auto"
          title={t("SampleCard.actions.drag_uploaded_files")}
        >
          {card.fcsFiles.length === 0 ? (
            <div className="text-sm text-neutral-500 dark:text-neutral-400">
              {t("SampleCard.empty")}
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
                      title={t("SampleCard.actions.remove_file")}
                      aria-label={t("SampleCard.actions.remove_file_named", {
                        file: displayName,
                      })}
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
