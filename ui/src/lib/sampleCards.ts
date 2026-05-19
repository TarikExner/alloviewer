import type { DragEvent } from "react";
import { type SampleCardModel } from "../components/SampleCard";
import { createId } from "./id";

export function nextSampleTitle(existing: SampleCardModel[]) {
  const nums = existing
    .filter((c) => c.sampleType === "sample")
    .map((c) => c.title.replace(/^Sa/i, ""))
    .map((x) => parseInt(x, 10))
    .filter((n) => Number.isFinite(n));

  const next = (nums.length ? Math.max(...nums) : 0) + 1;
  return `Sa${String(next).padStart(3, "0")}`;
}

export function makeInitialCards(): SampleCardModel[] {
  return [
    {
      id: createId(),
      sampleType: "negative",
      title: "Negative Control",
      name: "Negative Control",
      fcsFiles: [],
    },
    {
      id: createId(),
      sampleType: "positive",
      title: "Positive Control",
      name: "Positive Control",
      fcsFiles: [],
    },
    {
      id: createId(),
      sampleType: "sample",
      title: "Sa001",
      name: "Sa001",
      fcsFiles: [],
    },
  ];
}

export function readDraggedFcs(
  e: DragEvent
): { fname: string; fromCardId: string | null } | null {
  const json = e.dataTransfer.getData("application/x-allocviewer-fcsref");

  if (json) {
    try {
      const parsed = JSON.parse(json);
      if (parsed?.fname) {
        return {
          fname: String(parsed.fname),
          fromCardId: parsed.fromCardId ?? null,
        };
      }
    } catch {
      // ignore malformed drag payloads
    }
  }

  const raw =
    e.dataTransfer.getData("application/x-allocviewer-filename") ||
    e.dataTransfer.getData("text/plain");

  const fname = (raw || "").trim();
  if (!fname) return null;

  return { fname, fromCardId: null };
}

export function mapSampleRole(sampleType: string): "NC" | "PC" | "SAMPLE" {
  if (sampleType === "negative") return "NC";
  if (sampleType === "positive") return "PC";
  return "SAMPLE";
}
