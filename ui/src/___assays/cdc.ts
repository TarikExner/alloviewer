// src/assays/cdc.ts
import type { AssayConfig } from "./index";
import { runProcess as runCDC } from "../api/cdc";
export const CDC_CONFIG: AssayConfig = {
  id: "cdc",
  title: "CDC Assay",
  roles: ["sample","positive control","negative control","IgM control","empty"],
  plate: { rows: ["A","B","C","D","E","F"], cols: [1,2,3,4,5,6,7,8,9,10] },
  upload: {
    templateLabel: "Template (PDF)",
    imagesLabel: "Images (Folder)",
    acceptTemplate: "application/pdf",
    acceptImages: "image/*",
  },
  run: runCDC,
};

