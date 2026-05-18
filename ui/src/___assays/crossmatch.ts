// src/assays/crossmatch.ts
import type { AssayConfig } from "./index";
import { runProcess as runXM } from "../api/crossmatch";
export const CROSSMATCH_CONFIG: AssayConfig = {
  id: "crossmatch",
  title: "Crossmatch Assay",
  roles: ["sample","positive control","negative control","empty"], // adjust as needed
  plate: { rows: ["A","B","C","D","E","F"], cols: [1,2,3,4,5,6,7,8,9,10] },
  upload: {
    templateLabel: "Template (PDF)",
    imagesLabel: "Images (Folder)",
    acceptTemplate: "application/pdf",
    acceptImages: "image/*",
  },
  run: runXM,
  // optionally override summarize or validate here
};

