// src/api.ts
import type { ProcessRequest, ProcessResponse, WellID, WellMap } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

type SavedFile = { filename: string; size_mb: number };
type UploadResp = { saved: SavedFile[] };

export type ParsedPlateLayout = {
  upload_id: string;
  schema_version: string;
  sha256: string;
  lot_no?: string | null;
  compl_no?: string | null;
  plate_format?: string | null;
  warnings: string[];
  custom_loci: string[];
  wells: Record<string, {
    well_id: string;
    combo_id?: string | null;
    race?: string | null;
    loci: { data: Record<string, string[]> };
  }>;
  valid: boolean;
};

export async function parseLayout(file: File, base = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000") {
  const url = `${base}/api/plate-layouts/parse`;
  const form = new FormData();
  form.append("xlsx", file); // field name MUST be "xlsx"
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json(); // ParsedPlateLayout
}

export async function parseLayoutVerbose(
  file: File,
  base = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"
) {
  const url = `${base}/api/plate-layouts/parse`;
  console.log("[parseLayout] →", url, "file:", file.name, "size:", file.size);

  const form = new FormData();
  form.append("xlsx", file);
  const res = await fetch(url, { method: "POST", body: form });
  console.log("[parseLayout] status:", res.status, res.statusText);
  const text = await res.text();
  console.log("[parseLayout] raw body:", text.slice(0, 500));
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  try {
    const json = JSON.parse(text);
    console.log("[parseLayout] wells:", Object.keys(json?.wells ?? {}).length);
    return json;
  } catch (e) {
    console.error("[parseLayout] JSON parse error:", e);
    throw new Error("Bad JSON from /api/plate-layouts/parse");
  }
}

export function uploadWithProgress(
  files: File[],
  onProgress?: (percent: number) => void,
  base = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000")
): Promise<string[]> {
  return new Promise((resolve, reject) => {
    if (!files || files.length === 0) return resolve([]);

    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${base}/api/upload`);
    xhr.responseType = "text"; // safer to parse ourselves

    const report = (p: number) => {
      if (typeof onProgress === "function") {
        try { onProgress(Math.max(0, Math.min(100, Math.round(p)))); } catch {}
      }
    };

    let sawProgress = false;
    let fakeTimer: number | null = null;
    let fakeVal = 0;

    xhr.upload.onloadstart = () => {
      report(0);
      fakeTimer = window.setInterval(() => {
        if (sawProgress) return;
        // ease up to 95% while waiting for server reply
        fakeVal = Math.min(95, fakeVal + 4);
        report(fakeVal);
      }, 150);
    };

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable) {
        sawProgress = true;
        const pct = (evt.loaded / evt.total) * 100;
        report(pct);
      }
    };

    // extra guard: if onprogress never fires, we still finish on state change
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        // will be finalized again in onloadend, but calling is harmless
        report(100);
      }
    };

    // ALWAYS fires (after load/error/abort)
    xhr.onloadend = () => {
      if (fakeTimer) { clearInterval(fakeTimer); fakeTimer = null; }
      report(100);

      const status = xhr.status || 0;
      const ok = status >= 200 && status < 300;

      if (!ok) {
        return reject(new Error(`Upload failed: ${status} ${xhr.responseText || ""}`));
      }

      // tolerate empty body (e.g., 204) in dev
      const text = xhr.responseText || "";
      if (!text.trim()) return resolve([]);

      try {
        const parsed = JSON.parse(text) as { saved: { filename: string }[] };
        const names = (parsed?.saved ?? []).map((s) => s.filename);
        resolve(names);
      } catch {
        reject(new Error("Bad JSON from /api/upload"));
      }
    };

    xhr.onerror = () => {
      if (fakeTimer) clearInterval(fakeTimer);
      reject(new Error("Network error during upload"));
    };
    xhr.onabort = () => {
      if (fakeTimer) clearInterval(fakeTimer);
      reject(new Error("Upload aborted"));
    };

    xhr.send(fd);
  });
}

/** Upload without progress (simple fetch). */
export async function uploadFiles(files: File[]): Promise<string[]> {
  if (!files || files.length === 0) return [];
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));

  const res = await fetch(`${BASE}/api/upload`, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${res.status} ${text}`);
  }
  const data: UploadResp = await res.json();
  return data.saved.map((s) => s.filename);
}

/**
 * Run the backend. If you pass files, we upload them first and use the saved filenames.
 * You can also pass filenames directly if you already uploaded earlier.
 */
export async function runProcess(
  wells: WellMap,
  imageOrder: WellID[],
  opts: {
    templateFile?: File | null;
    imageFiles?: File[];
    templateFilename?: string | null;
    imageFilenames?: string[];
  } = {}
): Promise<ProcessResponse> {
  const layout = { wells };

  // upload if provided
  let template_filename = opts.templateFilename ?? null;
  let image_filenames = opts.imageFilenames ?? [];

  if (opts.templateFile) {
    const [saved] = await uploadFiles([opts.templateFile]); // or uploadWithProgress([file], cb)
    template_filename = saved ?? null;
  }
  if (opts.imageFiles && opts.imageFiles.length) {
    const saved = await uploadFiles(opts.imageFiles); // or uploadWithProgress(files, cb)
    image_filenames = saved;
  }

  const payload: ProcessRequest = {
    layout,
    image_order: imageOrder,
    template_filename,
    image_filenames,
  };

  const res = await fetch(`${BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Process failed: ${res.status} ${text}`);
  }
  return (await res.json()) as ProcessResponse;
}

