import { useTranslation } from "react-i18next";
import {
  EmptySummary,
  SummaryItem,
  SummarySection,
  WarningList,
  WellList,
} from "./SummaryBlocks";

function RunValiditySection({ run }: { run: any }) {
  const { t } = useTranslation();

  return (
    <>
      <SummarySection title={t("SummaryGrids.runValidity.title")}>
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.status")}
          value={run.status}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.pcMeanRaw")}
          value={run.pc_mean_raw}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.ncMeanRaw")}
          value={run.nc_mean_raw}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.dynamicRange")}
          value={run.dynamic_range}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.positiveControls")}
          value={run.n_positive_controls}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.negativeControls")}
          value={run.n_negative_controls}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.pcReplicateRange")}
          value={run.pc_replicate_range}
        />
        <SummaryItem
          label={t("SummaryGrids.runValidity.items.ncReplicateRange")}
          value={run.nc_replicate_range}
        />
      </SummarySection>

      <WarningList
        title={t("SummaryGrids.runValidity.controlWarnings")}
        warnings={run.control_warnings}
      />
    </>
  );
}

function QCSection({
  qc,
  showWellLists,
}: {
  qc: any;
  showWellLists?: boolean;
}) {
  const { t } = useTranslation();

  return (
    <>
      <SummarySection title={t("SummaryGrids.qc.title")}>
        <SummaryItem
          label={t("SummaryGrids.qc.items.totalWells")}
          value={qc.total_wells}
        />
        <SummaryItem
          label={t("SummaryGrids.qc.items.validWells")}
          value={qc.valid_wells}
        />
        <SummaryItem
          label={t("SummaryGrids.qc.items.meanRoiCount")}
          value={qc.mean_n_rois}
        />
        <SummaryItem
          label={t("SummaryGrids.qc.items.meanUncertainFraction")}
          value={qc.mean_uncertain_fraction}
        />
        <SummaryItem
          label={t("SummaryGrids.qc.items.lowRoiWells")}
          value={qc.low_roi_wells?.length ?? 0}
        />
        <SummaryItem
          label={t("SummaryGrids.qc.items.highUncertainWells")}
          value={qc.high_uncertain_wells?.length ?? 0}
        />
      </SummarySection>

      <WarningList
        title={t("SummaryGrids.qc.warnings")}
        warnings={qc.warnings}
      />

      {showWellLists ? (
        <>
          <WellList
            title={t("SummaryGrids.qc.lowRoiWells")}
            wells={qc.low_roi_wells}
          />
          <WellList
            title={t("SummaryGrids.qc.highUncertainWells")}
            wells={qc.high_uncertain_wells}
          />
        </>
      ) : null}
    </>
  );
}

export function PRASummaryGrid({ summary }: { summary: any | null }) {
  const { t } = useTranslation();

  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};

  return (
    <div className="space-y-4">
      <RunValiditySection run={run} />

      <SummarySection title={t("SummaryGrids.praResult.title")}>
        <SummaryItem
          label={t("SummaryGrids.praResult.items.praPercent")}
          value={assay.pra_percent}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.positivePanelWells")}
          value={assay.positive_panel_wells}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.validPanelWells")}
          value={assay.valid_panel_wells}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.meanCorrectedFracPos")}
          value={assay.mean_corrected_frac_pos}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.medianCorrectedFracPos")}
          value={assay.median_corrected_frac_pos}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.maxCorrectedFracPos")}
          value={assay.max_corrected_frac_pos}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.weakPositiveWells")}
          value={assay.n_weak_positive}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.moderatePositiveWells")}
          value={assay.n_moderate_positive}
        />
        <SummaryItem
          label={t("SummaryGrids.praResult.items.strongPositiveWells")}
          value={assay.n_strong_positive}
        />
      </SummarySection>

      <QCSection qc={qc} />
    </div>
  );
}

export function CrossmatchSummaryGrid({ summary }: { summary: any | null }) {
  const { t } = useTranslation();

  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};

  return (
    <div className="space-y-4">
      <RunValiditySection run={run} />

      <SummarySection title={t("SummaryGrids.crossmatchResult.title")}>
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.finalCall")}
          value={assay.final_call}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.correctedFracPos")}
          value={assay.sample_corrected_frac_pos}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.rawFracPos")}
          value={assay.sample_raw_frac_pos}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.marginFromCutoff")}
          value={assay.margin_from_cutoff}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.replicateSd")}
          value={assay.replicate_sd}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.replicateRange")}
          value={assay.replicate_range}
        />
        <SummaryItem
          label={t("SummaryGrids.crossmatchResult.items.replicateDiscordant")}
          value={assay.replicate_discordant}
        />
      </SummarySection>

      <WellList
        title={t("SummaryGrids.crossmatchResult.sampleWells")}
        wells={assay.sample_wells}
      />

      <QCSection qc={qc} showWellLists />
    </div>
  );
}
