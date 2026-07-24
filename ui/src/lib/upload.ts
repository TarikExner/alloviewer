type FileWithRelativePath = File & {
  __relativePath?: string;
  webkitRelativePath?: string;
};

export type FileOrderDirection =
  | "ascending"
  | "descending";

const NATURAL_FILENAME_COLLATOR =
  new Intl.Collator("en", {
    numeric: true,
    sensitivity: "base",
  });

function normalizeSortText(
  value: string
): string {
  return value.normalize("NFKC");
}

export function getFileRelativePath(
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

export function sortFilesNaturally(
  files: File[],
  direction: FileOrderDirection =
    "ascending"
): File[] {
  const multiplier =
    direction === "ascending"
      ? 1
      : -1;

  return files
    .map((file, originalIndex) => ({
      file,
      originalIndex,
    }))
    .sort((left, right) => {
      const filenameResult =
        NATURAL_FILENAME_COLLATOR.compare(
          normalizeSortText(
            left.file.name
          ),
          normalizeSortText(
            right.file.name
          )
        );

      if (filenameResult !== 0) {
        return (
          filenameResult * multiplier
        );
      }

      const pathResult =
        NATURAL_FILENAME_COLLATOR.compare(
          normalizeSortText(
            getFileRelativePath(
              left.file
            )
          ),
          normalizeSortText(
            getFileRelativePath(
              right.file
            )
          )
        );

      if (pathResult !== 0) {
        return pathResult * multiplier;
      }

      return (
        left.originalIndex -
        right.originalIndex
      );
    })
    .map(({ file }) => file);
}

export function findDuplicateFilenames(
  files: File[]
): string[] {
  const firstSpelling =
    new Map<string, string>();

  const counts =
    new Map<string, number>();

  for (const file of files) {
    const normalizedName =
      normalizeSortText(
        file.name
      );

    const key =
      normalizedName.toLocaleLowerCase(
        "en-US"
      );

    if (!firstSpelling.has(key)) {
      firstSpelling.set(
        key,
        file.name
      );
    }

    counts.set(
      key,
      (counts.get(key) ?? 0) + 1
    );
  }

  return [...counts.entries()]
    .filter(
      ([, count]) => count > 1
    )
    .map(
      ([key]) =>
        firstSpelling.get(key) ?? key
    )
    .sort((left, right) =>
      NATURAL_FILENAME_COLLATOR.compare(
        left,
        right
      )
    );
}

export function normalizeSavedNames(
  saved: any[]
): string[] {
  return (saved || [])
    .map(getUploadedFilename)
    .filter(
      (
        filename
      ): filename is string =>
        Boolean(filename)
    );
}

export function sameStringArray(
  a: string[],
  b: string[]
) {
  if (a.length !== b.length) {
    return false;
  }

  return a.every(
    (value, index) =>
      value === b[index]
  );
}

export function sameFiles(
  a: File[],
  b: File[]
) {
  if (a.length !== b.length) {
    return false;
  }

  return a.every(
    (file, index) =>
      file.name === b[index]?.name &&
      file.size === b[index]?.size &&
      file.lastModified ===
        b[index]?.lastModified &&
      getFileRelativePath(file) ===
        (b[index]
          ? getFileRelativePath(
              b[index]
            )
          : "")
  );
}

export function getUploadedFilename(
  item: any
): string | null {
  if (!item) {
    return null;
  }

  if (typeof item === "string") {
    return item;
  }

  return (
    item.filename ??
    item.name ??
    null
  );
}

export const SUPPORTED_IMAGE_EXTENSIONS = [
  "tif",
  "tiff",
  "png",
  "jpg",
  "jpeg",
  "bmp",
  "webp",
] as const;

const SUPPORTED_IMAGE_PATTERN =
  new RegExp(
    `\\.(${SUPPORTED_IMAGE_EXTENSIONS.join(
      "|"
    )})$`,
    "i"
  );

export function isSupportedImageFile(
  file: File
): boolean {
  return SUPPORTED_IMAGE_PATTERN.test(
    file.name
  );
}
