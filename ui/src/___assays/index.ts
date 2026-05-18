// src/assays/index.ts
// src/assays/index.ts


import type { WellID, WellMap, ProcessResponse } from "../types";
export type AssayID = "cdc" | "crossmatch";

export type AssayConfig = {
  id: AssayID;
  title: string;                // shown in toolbar/page
  roles: string[];              // e.g. ["sample","positive control","negative control","IgM control","empty"]
  plate: { rows: string[]; cols: number[] }; // 6x10, 8x12, etc.
  upload: {
    templateLabel: string;      // "Template (PDF)" or custom
    imagesLabel: string;        // "Images (Folder)" or custom
    acceptTemplate: string;     // "application/pdf"
    acceptImages: string;       // "image/*"
  };
  // optional per-assay validation for wells/order
  validate?: (wells: WellMap, order: WellID[]) => string | null;
  // run function (can hit different endpoints)
  run: (wells: WellMap, order: WellID[], files: {
    templateFilename: string | null;
    imageFilenames: string[];
  }) => Promise<ProcessResponse>;
  // summary calculator (defaults to a shared one if omitted)
  summarize?: (proc: ProcessResponse | null) => Record<string, number> | null;
};

import { CDC_CONFIG } from "./cdc";
import { CROSSMATCH_CONFIG } from "./crossmatch";
export { type AssayConfig, type AssayID } from "./index";
export const ASSAYS = {
  cdc: CDC_CONFIG,
  crossmatch: CROSSMATCH_CONFIG,
} as const;

