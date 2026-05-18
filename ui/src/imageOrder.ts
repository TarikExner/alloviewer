import { type WellID } from "./types";
import { ROWS, COLS } from "./plateConfig";

export type SnakeOrientation = "horizontal" | "vertical";
export type StartCorner = "tl" | "tr" | "bl" | "br";

export function buildOrder(orientation: SnakeOrientation, start: StartCorner): WellID[] {

  // sequences for rows / cols based on start corner
  const rows = start.startsWith("t") ? [...ROWS] : [...ROWS].reverse();
  const cols = (start.endsWith("l") ? [...COLS] : [...COLS].reverse());

  const order: WellID[] = [];

  if (orientation === "horizontal") {
    // snake along rows
    rows.forEach((r, i) => {
      const walk = i % 2 === 0 ? cols : [...cols].reverse();
      walk.forEach(c => order.push(`${r}${c}` as WellID));
    });
  } else {
    // snake along cols
    cols.forEach((c, j) => {
      const walk = j % 2 === 0 ? rows : [...rows].reverse();
      walk.forEach(r => order.push(`${r}${c}` as WellID));
    });
  }
  return order;
}

