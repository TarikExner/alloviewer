import { ROWS, COLS } from "../plateConfig";
import type { WellID, WellMap, WellType } from "../types";

export type CellMode = "T" | "B" | "T/B" | "empty";

export function buildDefaultCrossmatch(): WellMap {
  const map: Record<WellID, WellType> = Object.create(null);

  ROWS.forEach((r) => {
    COLS.forEach((c) => {
      const id = `${r}${c}` as WellID;

      if (r === "F") map[id] = "positive";
      else if (r === "D") map[id] = "negative";
      else if (r === "E") map[id] = "igm";
      else map[id] = "sample";

      if (c === 10) map[id] = "empty";
    });
  });

  return map as WellMap;
}

export function buildDefaultColumnModes(): Record<number, CellMode> {
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
