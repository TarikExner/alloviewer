import React from "react";
import { plateGridTemplate } from "../layout/grid";

type Props<R extends string | number, C extends string | number> = {
  rows: readonly R[];
  cols: readonly C[];
  // Put custom classes around the grid if you want
  className?: string;
  // Render a well cell; we pass row, col, and wellId string
  renderCell: (row: R, col: C, wellId: string) => React.ReactNode;
  // Optional: render a header above the grid (e.g., title, hint)
  header?: React.ReactNode;
  // Optional: component for the row/col labels if you need custom markup
  renderRowLabel?: (row: R) => React.ReactNode;
  renderColLabel?: (col: C) => React.ReactNode;
  // Sizing overrides (fallback to config values if not provided)
  cellMin?: string;
  cellMax?: string;
};

export function PlateMatrix<R extends string | number, C extends string | number>({
  rows,
  cols,
  className,
  renderCell,
  header,
  renderRowLabel = (r) => (
    <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">{String(r)}</div>
  ),
  renderColLabel = (c) => (
    <div className="text-xs text-center text-neutral-600 dark:text-neutral-400">{String(c)}</div>
  ),
  cellMin = "1.8rem",
  cellMax = "2.25rem",
}: Props<R, C>) {
  return (
    <div className={className}>
      {header}
      <div
        className="inline-grid gap-1"
        style={{ gridTemplateColumns: plateGridTemplate(cols.length, cellMin, cellMax) }}
      >
        {/* Empty corner cell */}
        <div />
        {/* Column labels */}
        {cols.map((c) => (
          <React.Fragment key={`col-${String(c)}`}>{renderColLabel(c)}</React.Fragment>
        ))}
        {/* Rows */}
        {rows.map((r) => (
          <React.Fragment key={`row-${String(r)}`}>
            {renderRowLabel(r)}
            {cols.map((c) => {
              const wellId = `${r}${c}`;
              return <React.Fragment key={wellId}>{renderCell(r, c, wellId)}</React.Fragment>;
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

