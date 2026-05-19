import { ROWS, COLS } from "../plateConfig";
import type { WellID, WellMap, WellType } from "../types";

export function buildDefaultCDC(): WellMap {
  const map: Record<WellID, WellType> = Object.create(null);

  ROWS.forEach((r) =>
    COLS.forEach((c) => {
      map[`${r}${c}` as WellID] = "sample";
    })
  );

  (["A1", "B1"] as WellID[]).forEach((id) => {
    map[id] = "negative";
  });

  (["A10", "B10"] as WellID[]).forEach((id) => {
    map[id] = "positive";
  });

  return map as WellMap;
}
