import {
  EmptySummary,
  SummaryItem,
  SummarySection,
  WarningList,
  WellList,
} from "./SummaryBlocks";

function RunValiditySection({ run }: { run: any }) {
  return (
    <>
      <SummarySection title="Run validity">
        <SummaryItem label="Status" value={run.status} />
        <SummaryItem label="PC mean raw" value={run.pc_mean_raw} />
        <SummaryItem label="NC mean raw" value={run.nc_mean_raw} />
        <SummaryItem label="Dynamic range" value={run.dynamic_range} />
        <SummaryItem label="Positive controls" value={run.n_positive_controls} />
        <SummaryItem label="Negative controls" value={run.n_negative_controls} />
        <SummaryItem label="PC replicate range" value={run.pc_replicate_range} />
        <SummaryItem label="NC replicate range" value={run.nc_replicate_range} />
      </SummarySection>

      <WarningList title="Control warnings" warnings={run.control_warnings} />
    </>
  );
}

function QCSection({ qc, showWellLists }: { qc: any; showWellLists?: boolean }) {
  return (
    <>
      <SummarySection title="QC">
        <SummaryItem label="Total wells" value={qc.total_wells} />
        <SummaryItem label="Valid wells" value={qc.valid_wells} />
        <SummaryItem label="Mean ROI count" value={qc.mean_n_rois} />
        <SummaryItem
          label="Mean uncertain fraction"
          value={qc.mean_uncertain_fraction}
        />
        <SummaryItem
          label="Low ROI wells"
          value={qc.low_roi_wells?.length ?? 0}
        />
        <SummaryItem
          label="High uncertain wells"
          value={qc.high_uncertain_wells?.length ?? 0}
        />
      </SummarySection>

      <WarningList title="QC warnings" warnings={qc.warnings} />

      {showWellLists ? (
        <>
          <WellList title="Low ROI wells" wells={qc.low_roi_wells} />
          <WellList title="High uncertain wells" wells={qc.high_uncertain_wells} />
        </>
      ) : null}
    </>
  );
}

export function PRASummaryGrid({ summary }: { summary: any | null }) {
  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};

  return (
    <div className="space-y-4">
      <RunValiditySection run={run} />

      <SummarySection title="PRA result">
        <SummaryItem label="PRA %" value={assay.pra_percent} />
        <SummaryItem label="Positive panel wells" value={assay.positive_panel_wells} />
        <SummaryItem label="Valid panel wells" value={assay.valid_panel_wells} />
        <SummaryItem
          label="Mean corrected frac pos"
          value={assay.mean_corrected_frac_pos}
        />
        <SummaryItem
          label="Median corrected frac pos"
          value={assay.median_corrected_frac_pos}
        />
        <SummaryItem
          label="Max corrected frac pos"
          value={assay.max_corrected_frac_pos}
        />
        <SummaryItem label="Weak positive wells" value={assay.n_weak_positive} />
        <SummaryItem
          label="Moderate positive wells"
          value={assay.n_moderate_positive}
        />
        <SummaryItem label="Strong positive wells" value={assay.n_strong_positive} />
      </SummarySection>

      <QCSection qc={qc} />
    </div>
  );
}

export function CrossmatchSummaryGrid({ summary }: { summary: any | null }) {
  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};

  return (
    <div className="space-y-4">
      <RunValiditySection run={run} />

      <SummarySection title="Crossmatch result">
        <SummaryItem label="Final call" value={assay.final_call} />
        <SummaryItem
          label="Corrected frac pos"
          value={assay.sample_corrected_frac_pos}
        />
        <SummaryItem label="Raw frac pos" value={assay.sample_raw_frac_pos} />
        <SummaryItem label="Margin from cutoff" value={assay.margin_from_cutoff} />
        <SummaryItem label="Replicate SD" value={assay.replicate_sd} />
        <SummaryItem label="Replicate range" value={assay.replicate_range} />
        <SummaryItem
          label="Replicate discordant"
          value={assay.replicate_discordant}
        />
      </SummarySection>

      <WellList title="Sample wells" wells={assay.sample_wells} />

      <QCSection qc={qc} showWellLists />
    </div>
  );
}
