export function normalizeSavedNames(saved: any[]): string[] {
  return (saved || [])
    .map((x: any) => (typeof x === "string" ? x : x?.filename))
    .filter(Boolean);
}

export function sameStringArray(a: string[], b: string[]) {
  if (a.length !== b.length) return false;
  return a.every((x, i) => x === b[i]);
}

export function sameFiles(a: File[], b: File[]) {
  if (a.length !== b.length) return false;

  return a.every(
    (f, i) =>
      f.name === b[i]?.name &&
      f.size === b[i]?.size &&
      f.lastModified === b[i]?.lastModified
  );
}

export function getUploadedFilename(item: any): string | null {
  if (!item) return null;
  if (typeof item === "string") return item;
  return item.filename ?? item.name ?? null;
}
