import type { WellType } from "./types";

/**
 * One place to tune colors. Stronger borders and higher saturation in dark mode.
 * swap these to any palette you like (e.g. teal/orange/purple).
 */
export const ROLE_STYLES: Record<WellType, string> = {
  positive:
    "border-rose-500 bg-rose-50 text-rose-800 " +
    "dark:border-rose-400 dark:bg-rose-400/10 dark:text-rose-100",

  negative:
    "border-emerald-500 bg-emerald-50 text-emerald-800 " +
    "dark:border-emerald-400 dark:bg-emerald-400/10 dark:text-emerald-100",

  igm:
    "border-amber-500 bg-amber-50 text-amber-800 " +
    "dark:border-amber-400 dark:bg-amber-400/10 dark:text-amber-100",

  sample:
    "border-sky-600 bg-sky-50 text-sky-800 " +
    "dark:border-sky-400 dark:bg-sky-400/10 dark:text-sky-100",

  empty:
    "border-neutral-300 bg-white text-neutral-700 " +
    "dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-300",
};

export const ROLE_SWATCH: Record<WellType, string> = {
  positive:
    "border-rose-500 bg-rose-500/40 dark:border-rose-400 dark:bg-rose-400/25",
  negative:
    "border-emerald-500 bg-emerald-500/40 dark:border-emerald-400 dark:bg-emerald-400/25",
  igm:
    "border-amber-500 bg-amber-500/50 dark:border-amber-400 dark:bg-amber-400/30",
  sample:
    "border-sky-600 bg-sky-600/40 dark:border-sky-400 dark:bg-sky-400/25",
  empty:
    "border-neutral-300 bg-neutral-200/40 dark:border-neutral-700 dark:bg-neutral-700/40",
};
