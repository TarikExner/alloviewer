// src/api.ts

const BASE =
  import.meta.env.VITE_API_BASE ??
  "http://127.0.0.1:8000";

export type JobType =
  | "pra"
  | "crossmatch"
  | "fcxm";

export type JobUploadKind =
  | "images"
  | "fcs";

export type JobResponse = {
  job_id: string;
  job_type: JobType;
  status: string;
  stage?: string | null;
  created_at: number;
  updated_at: number;
};

export type SavedFile = {
  filename: string;
  size_mb: number;
};

export type UploadResponse = {
  saved: SavedFile[];
};

export type ParsedPlateLayout = {
  upload_id: string;
  schema_version: string;
  sha256: string;

  lot_no?: string | null;
  compl_no?: string | null;
  plate_format?: string | null;

  warnings: string[];
  custom_loci: string[];

  wells: Record<
    string,
    {
      well_id: string;
      combo_id?: string | null;
      race?: string | null;

      loci: {
        data: Record<string, string[]>;
      };
    }
  >;

  valid: boolean;
};

type FastApiValidationError = {
  loc?: Array<string | number>;
  msg?: string;
  type?: string;
};

type FastApiErrorBody = {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
};

type FileWithRelativePath = File & {
  __relativePath?: string;
  webkitRelativePath?: string;
};


function formatValidationError(
  error: FastApiValidationError
): string {
  const location = Array.isArray(error.loc)
    ? error.loc
        .filter((part) => part !== "body")
        .map(String)
        .join(".")
    : "";

  const message =
    error.msg?.trim() ||
    "Invalid request value";

  return location
    ? `${location}: ${message}`
    : message;
}


function formatErrorDetail(
  detail: unknown
): string {
  if (typeof detail === "string") {
    return detail.trim();
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (
          item &&
          typeof item === "object"
        ) {
          return formatValidationError(
            item as FastApiValidationError
          );
        }

        return String(item);
      })
      .filter(Boolean)
      .join("\n");
  }

  if (
    detail &&
    typeof detail === "object"
  ) {
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }

  return "";
}


function extractApiErrorMessage(
  body: unknown
): string {
  if (
    !body ||
    typeof body !== "object"
  ) {
    return typeof body === "string"
      ? body.trim()
      : "";
  }

  const data =
    body as FastApiErrorBody;

  const detail = formatErrorDetail(
    data.detail
  );

  if (detail) {
    return detail;
  }

  const message = formatErrorDetail(
    data.message
  );

  if (message) {
    return message;
  }

  return formatErrorDetail(
    data.error
  );
}


async function readApiError(
  response: Response,
  fallback: string
): Promise<string> {
  const text = await response
    .text()
    .catch(() => "");

  if (!text.trim()) {
    return `${fallback} (${response.status})`;
  }

  try {
    const parsed = JSON.parse(text);
    const message =
      extractApiErrorMessage(parsed);

    if (message) {
      return message;
    }
  } catch {
    // Plain-text response.
  }

  return (
    text.trim() ||
    `${fallback} (${response.status})`
  );
}


function readXhrError(
  xhr: XMLHttpRequest,
  fallback: string
): string {
  const raw =
    xhr.responseText?.trim() || "";

  if (!raw) {
    return `${fallback} (${xhr.status || 0})`;
  }

  try {
    const parsed = JSON.parse(raw);
    const message =
      extractApiErrorMessage(parsed);

    if (message) {
      return message;
    }
  } catch {
    // Plain-text response.
  }

  return raw;
}


function relativePathForFile(
  file: File
): string {
  const extended =
    file as FileWithRelativePath;

  const relativePath =
    extended.__relativePath ||
    extended.webkitRelativePath ||
    file.name;

  return relativePath
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
}


export async function createJob(
  jobType: JobType,
  base = BASE
): Promise<JobResponse> {
  const response = await fetch(
    `${base}/api/jobs`,
    {
      method: "POST",
      headers: {
        "Content-Type":
          "application/json",
      },
      body: JSON.stringify({
        job_type: jobType,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Could not create job"
      )
    );
  }

  return (
    await response.json()
  ) as JobResponse;
}


export async function getJob(
  jobId: string,
  base = BASE
): Promise<JobResponse> {
  if (!jobId) {
    throw new Error(
      "Job request is missing job_id."
    );
  }

  const response = await fetch(
    `${base}/api/jobs/${encodeURIComponent(
      jobId
    )}`
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Could not load job"
      )
    );
  }

  return (
    await response.json()
  ) as JobResponse;
}


export async function deleteJob(
  jobId: string,
  base = BASE
): Promise<void> {
  if (!jobId) {
    return;
  }

  const response = await fetch(
    `${base}/api/jobs/${encodeURIComponent(
      jobId
    )}`,
    {
      method: "DELETE",
    }
  );

  if (
    !response.ok &&
    response.status !== 404
  ) {
    throw new Error(
      await readApiError(
        response,
        "Could not delete job"
      )
    );
  }
}


export async function parseLayout(
  jobId: string,
  file: File,
  base = BASE
): Promise<ParsedPlateLayout> {
  if (!jobId) {
    throw new Error(
      "Plate-layout upload is missing job_id."
    );
  }

  if (!file) {
    throw new Error(
      "No plate-layout file was selected."
    );
  }

  const form = new FormData();

  // Backend field name remains "xlsx".
  form.append(
    "xlsx",
    file,
    file.name
  );

  const response = await fetch(
    `${base}/api/jobs/${encodeURIComponent(
      jobId
    )}/plate-layout`,
    {
      method: "POST",
      body: form,
    }
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        "Could not parse plate layout"
      )
    );
  }

  return (
    await response.json()
  ) as ParsedPlateLayout;
}


export function uploadWithProgress(
  jobId: string,
  uploadKind: JobUploadKind,
  files: File[],
  onProgress?: (
    percent: number
  ) => void,
  base = BASE
): Promise<string[]> {
  return new Promise(
    (resolve, reject) => {
      if (!jobId) {
        reject(
          new Error(
            "Upload is missing job_id."
          )
        );
        return;
      }

      if (
        !files ||
        files.length === 0
      ) {
        resolve([]);
        return;
      }

      const form = new FormData();

      files.forEach((file) => {
        form.append(
          "files",
          file,
          file.name
        );

        form.append(
          "relative_paths",
          relativePathForFile(file)
        );
      });

      const xhr =
        new XMLHttpRequest();

      xhr.open(
        "POST",
        `${base}/api/jobs/${encodeURIComponent(
          jobId
        )}/uploads/${encodeURIComponent(
          uploadKind
        )}`
      );

      xhr.responseType = "text";

      let fakeTimer:
        | number
        | null = null;

      let fakeValue = 0;
      let sawNativeProgress = false;
      let settled = false;

      const stopFakeProgress = () => {
        if (fakeTimer !== null) {
          window.clearInterval(
            fakeTimer
          );

          fakeTimer = null;
        }
      };

      const report = (
        value: number
      ) => {
        if (
          typeof onProgress !==
          "function"
        ) {
          return;
        }

        const normalized = Math.max(
          0,
          Math.min(
            100,
            Math.round(value)
          )
        );

        try {
          onProgress(normalized);
        } catch {
          // Progress callbacks must not
          // interrupt the upload.
        }
      };

      const rejectOnce = (
        error: Error
      ) => {
        if (settled) {
          return;
        }

        settled = true;
        stopFakeProgress();
        reject(error);
      };

      const resolveOnce = (
        names: string[]
      ) => {
        if (settled) {
          return;
        }

        settled = true;
        stopFakeProgress();
        report(100);
        resolve(names);
      };

      xhr.upload.onloadstart = () => {
        report(0);

        fakeTimer =
          window.setInterval(() => {
            if (
              sawNativeProgress ||
              settled
            ) {
              return;
            }

            fakeValue = Math.min(
              95,
              fakeValue + 4
            );

            report(fakeValue);
          }, 150);
      };

      xhr.upload.onprogress = (
        event
      ) => {
        if (
          event.lengthComputable
        ) {
          sawNativeProgress = true;

          report(
            (event.loaded /
              event.total) *
              100
          );
        }
      };

      xhr.onload = () => {
        stopFakeProgress();

        const status =
          xhr.status || 0;

        if (
          status < 200 ||
          status >= 300
        ) {
          rejectOnce(
            new Error(
              readXhrError(
                xhr,
                "Upload failed"
              )
            )
          );

          return;
        }

        const text =
          xhr.responseText || "";

        if (!text.trim()) {
          resolveOnce([]);
          return;
        }

        try {
          const response =
            JSON.parse(
              text
            ) as UploadResponse;

          const names = (
            response.saved || []
          )
            .map(
              (saved) =>
                saved.filename
            )
            .filter(Boolean);

          resolveOnce(names);
        } catch {
          rejectOnce(
            new Error(
              "The upload endpoint returned invalid JSON."
            )
          );
        }
      };

      xhr.onerror = () => {
        rejectOnce(
          new Error(
            "Network error during upload."
          )
        );
      };

      xhr.onabort = () => {
        rejectOnce(
          new Error(
            "Upload was aborted."
          )
        );
      };

      xhr.ontimeout = () => {
        rejectOnce(
          new Error(
            "Upload timed out."
          )
        );
      };

      xhr.send(form);
    }
  );
}
