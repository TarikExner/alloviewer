export const ROWS = ["A", "B", "C", "D", "E", "F"] as const;
export const COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] as const;
export const WELL_TYPES = ["positive", "negative", "sample", "empty", "igm"] as const;

export type PlateLayout = {
  wells: WellMap;
};

export type Row = typeof ROWS[number];
export type Col = typeof COLS[number];
export type WellID = `${Row}${Col}`;

export const ALL_WELLS: WellID[] = ROWS.flatMap((r) =>
  COLS.map((c) => `${r}${c}` as WellID)
);

export type WellType = typeof WELL_TYPES[number];

export type WellMap = Record<WellID, WellType>;

export type WellResult = {
  well: WellID;
  role: WellType;
  score: number;
  status: "ok" | "warn" | "fail";
};

export type ProcessStartResponse = {
  job_id: string;
};

export const ROLE_LABEL: Record<WellType, string> = {
  positive: "positive control",
  negative: "negative control",
  sample:   "sample",
  igm:      "IgM control",
  empty:    "empty",
};


export type WellSummary = {
  well_id: WellID;
  n_rois: number;
  n_pos: number;
  frac_pos: number;
  frac_pos_corrected?: number | null;
  qc?: Record<string, unknown>;
  store_paths?: Record<string, string>;
  preview_path?: string | null;
  segmented_image_url?: string | null;
};

export type ProcessResponse = {
  calib?: unknown;
  wells?: Partial<Record<WellID, WellSummary>>;
  summary?: CDCSummary;
  pra_analysis?: PraAnalysis | null;
};

export type CDCControlStatus = "valid" | "warning" | "invalid";

export type CDCAssayType = "pra" | "crossmatch";

export type CDCRunValiditySummary = {
  status: CDCControlStatus | string;
  pc_mean_raw: number;
  nc_mean_raw: number;
  dynamic_range: number;
  pc_replicate_range?: number;
  nc_replicate_range?: number;
  n_positive_controls: number;
  n_negative_controls: number;
  control_warnings: string[];
};

export type CDCQCSummary = {
  total_wells: number;
  valid_wells: number;
  low_roi_wells: string[];
  high_uncertain_wells: string[];
  mean_n_rois: number;
  mean_uncertain_fraction: number;
  warnings: string[];
};

export type CDCPRAResultSummary = {
  pra_percent: number;
  positive_panel_wells: number;
  valid_panel_wells: number;
  mean_corrected_frac_pos: number;
  median_corrected_frac_pos: number;
  max_corrected_frac_pos: number;
  n_weak_positive: number;
  n_moderate_positive: number;
  n_strong_positive: number;
  positive_wells: string[];
};

export type CDCCrossmatchResultSummary = {
  final_call: string;
  sample_corrected_frac_pos: number;
  sample_raw_frac_pos: number;
  margin_from_cutoff: number;
  replicate_sd: number;
  replicate_range: number;
  replicate_discordant: boolean;
  sample_wells: string[];
};

export type CDCSummary = {
  assay_type: CDCAssayType | string;
  run_validity: CDCRunValiditySummary;
  assay_result: CDCPRAResultSummary | CDCCrossmatchResultSummary;
  qc: CDCQCSummary;
};

export type AlleleReactivityEvidence = {
  allele_key: string;
  locus: string;
  allele: string;

  positive_well_count: number;
  total_well_count: number;
  negative_well_count: number;

  positive_fraction: number;
  positive_ratio: string;

  positive_wells: string[];
  negative_wells: string[];
  missing_result_wells: string[];

  well_values: Record<string, number | null>;
};

export type PraReactivityScore = {
  positive_well_count: number;
  total_well_count: number;
  positive_fraction: number;
  score_percent: number;
  threshold: number;
  positive_wells: string[];
  negative_wells: string[];
  missing_result_wells: string[];
};

export type PraAnalysis = {
  positivity_threshold: number;
  included_well_type?: string;
  included_wells?: string[];
  reactivity_score: PraReactivityScore;
  alleles: AlleleReactivityEvidence[];
};
