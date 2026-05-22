import React from "react";
import { useTranslation } from "react-i18next";
import { plateGridTemplate } from "../layout/grid";

type Props<R extends string | number, C extends string | number> = {
  rows: readonly R[];
  cols: readonly C[];
  className?: string;
  renderCell: (row: R, col: C, wellId: string) => React.ReactNode;
  header?: React.ReactNode;
  renderRowLabel?: (row: R) => React.ReactNode;
  renderColLabel?: (col: C) => React.ReactNode;
  cellMin?: string;
  cellMax?: string;
};

export function PlateMatrix<
  R extends string | number,
  C extends string | number
>({
  rows,
  cols,
  className,
  renderCell,
  header,
  renderRowLabel = (r) => (
    <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">
      {String(r)}
    </div>
  ),
  renderColLabel = (c) => (
    <div className="text-xs text-center text-neutral-600 dark:text-neutral-400">
      {String(c)}
    </div>
  ),
  cellMin = "1.8rem",
  cellMax = "2.25rem",
}: Props<R, C>) {
  const { t } = useTranslation();

  return (
    <div className={className}>
      {header}

      <div
        className="inline-grid gap-1"
        role="grid"
        aria-label={t("PlateMatrix.aria_label")}
        style={{
          gridTemplateColumns: plateGridTemplate(
            cols.length,
            cellMin,
            cellMax
          ),
        }}
      >
        <div aria-hidden="true" />

        {cols.map((c) => (
          <React.Fragment key={`col-${String(c)}`}>
            {renderColLabel(c)}
          </React.Fragment>
        ))}

        {rows.map((r) => (
          <React.Fragment key={`row-${String(r)}`}>
            {renderRowLabel(r)}

            {cols.map((c) => {
              const wellId = `${r}${c}`;
              return (
                <React.Fragment key={wellId}>
                  {renderCell(r, c, wellId)}
                </React.Fragment>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
