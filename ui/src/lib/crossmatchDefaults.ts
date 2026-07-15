import { COLS, ROWS } from "../plateConfig";
import type {
  CrossmatchCellMode,
  CrossmatchColumnModes,
  WellID,
  WellMap,
  WellType,
} from "../types";

export type CellMode = CrossmatchCellMode;

export function buildDefaultCrossmatch(): WellMap {
  const map: Record<WellID, WellType> = Object.create(null);

  ROWS.forEach((row) => {
    COLS.forEach((column) => {
      const id = `${row}${column}` as WellID;

      if (row === "F") map[id] = "positive";
      else if (row === "D") map[id] = "negative";
      else if (row === "E") map[id] = "igm";
      else map[id] = "sample";

      if (column === 10) map[id] = "empty";
    });
  });

  return map as WellMap;
}

export function buildDefaultColumnModes(): CrossmatchColumnModes {
  return {
    1: "T",
    2: "T",
    3: "T",
    4: "B",
    5: "B",
    6: "B",
    7: "T/B",
    8: "T/B",
    9: "T/B",
    10: "empty",
  };
}

