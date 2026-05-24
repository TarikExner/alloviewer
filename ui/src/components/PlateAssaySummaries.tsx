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
function formatNumber(value: unknown, digits = 1) {
  if (value === null || value === undefined) return "—";

  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);

  return n.toFixed(digits);
}

function formatPercent(value: unknown, digits = 1) {
  if (value === null || value === undefined) return "—";

  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);

  return `${n.toFixed(digits)}%`;
}

function CompactMetric({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border bg-white px-3 py-2 dark:bg-neutral-900 dark:border-neutral-800">
      <div className="text-[11px] leading-4 text-neutral-500 dark:text-neutral-400">
        {label}
      </div>
      <div className="mt-0.5 text-lg font-semibold leading-6 text-neutral-950 dark:text-neutral-50">
        {value}
      </div>
      {sub ? (
        <div className="mt-0.5 text-[11px] leading-4 text-neutral-500 dark:text-neutral-400">
          {sub}
        </div>
      ) : null}
    </div>
  );
}

function CompactPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border bg-white p-3 dark:bg-neutral-900 dark:border-neutral-800">
      <div className="mb-2 text-sm font-medium text-neutral-900 dark:text-neutral-100">
        {title}
      </div>
      {children}
    </section>
  );
}

function RunQcCompact({
  run,
  qc,
}: {
  run: any;
  qc: any;
}) {
  return (
    <div className="grid grid-cols-1 2xl:grid-cols-2 gap-3">
      <CompactPanel title="Run validity">
        <div className="grid grid-cols-2 lg:grid-cols-4 2xl:grid-cols-4 gap-2">
          <CompactMetric label="Status" value={run.status ?? "—"} />
          <CompactMetric label="PC mean" value={formatNumber(run.pc_mean_raw, 3)} />
          <CompactMetric label="NC mean" value={formatNumber(run.nc_mean_raw, 3)} />
          <CompactMetric label="Range" value={formatNumber(run.dynamic_range, 3)} />
          <CompactMetric label="PC controls" value={run.n_positive_controls ?? "—"} />
          <CompactMetric label="NC controls" value={run.n_negative_controls ?? "—"} />
          <CompactMetric
            label="PC rep. range"
            value={formatNumber(run.pc_replicate_range, 3)}
          />
          <CompactMetric
            label="NC rep. range"
            value={formatNumber(run.nc_replicate_range, 3)}
          />
        </div>

        {run.control_warnings?.length ? (
          <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
            {run.control_warnings.join(", ")}
          </div>
        ) : null}
      </CompactPanel>

      <CompactPanel title="QC">
        <div className="grid grid-cols-2 lg:grid-cols-4 2xl:grid-cols-4 gap-2">
          <CompactMetric label="Total wells" value={qc.total_wells ?? "—"} />
          <CompactMetric label="Valid wells" value={qc.valid_wells ?? "—"} />
          <CompactMetric label="Mean ROI" value={formatNumber(qc.mean_n_rois, 1)} />
          <CompactMetric
            label="Uncertain"
            value={formatNumber(qc.mean_uncertain_fraction, 3)}
          />
          <CompactMetric
            label="Low ROI"
            value={qc.low_roi_wells?.length ?? 0}
          />
          <CompactMetric
            label="High uncertain"
            value={qc.high_uncertain_wells?.length ?? 0}
          />
        </div>

        {qc.warnings?.length ? (
          <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
            {qc.warnings.join(", ")}
          </div>
        ) : null}
      </CompactPanel>
    </div>
  );
}

function AlleleEvidenceTable({ alleles }: { alleles: any[] }) {
  const topAlleles = [...(alleles ?? [])]
    .sort((a, b) => {
      const aScore = Number(a.positive_fraction ?? 0);
      const bScore = Number(b.positive_fraction ?? 0);

      if (bScore !== aScore) return bScore - aScore;

      return Number(b.positive_well_count ?? 0) - Number(a.positive_well_count ?? 0);
    })
    .slice(0, 20);

  if (!topAlleles.length) {
    return (
      <div className="rounded-xl border px-3 py-3 text-sm text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
        No allele evidence available.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border dark:border-neutral-800">
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 z-10 bg-neutral-50 text-xs text-neutral-500 dark:bg-neutral-950 dark:text-neutral-400">
            <tr className="border-b dark:border-neutral-800">
              <th className="px-3 py-2 font-medium">Allele</th>
              <th className="px-3 py-2 font-medium">Positive score</th>
              <th className="px-3 py-2 font-medium">Fraction</th>
              <th className="px-3 py-2 font-medium">Positive wells</th>
              <th className="px-3 py-2 font-medium">Negative wells</th>
            </tr>
          </thead>

          <tbody className="divide-y dark:divide-neutral-800">
            {topAlleles.map((a) => {
              const pct = Number(a.positive_fraction ?? 0) * 100;

              return (
                <tr
                  key={a.allele_key}
                  className="bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-850"
                >
                  <td className="px-3 py-2 font-medium text-neutral-950 dark:text-neutral-50">
                    {a.allele_key}
                  </td>

                  <td className="px-3 py-2">
                    <span className="rounded-full border px-2 py-0.5 text-xs font-medium dark:border-neutral-700">
                      {a.positive_ratio ?? "—"}
                    </span>
                  </td>

                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
                        <div
                          className="h-full rounded-full bg-blue-600 dark:bg-blue-500"
                          style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
                        />
                      </div>
                      <span className="w-12 text-xs text-neutral-600 dark:text-neutral-400">
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                  </td>

                  <td className="px-3 py-2 text-xs text-neutral-600 dark:text-neutral-400">
                    {(a.positive_wells ?? []).join(", ") || "—"}
                  </td>

                  <td className="px-3 py-2 text-xs text-neutral-600 dark:text-neutral-400">
                    {(a.negative_wells ?? []).length}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PRASummaryGrid({
  summary,
  result,
}: {
  summary: any | null;
  result?: any | null;
}) {
  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};
  const pra = result?.pra_analysis ?? null;
  const reactivity = pra?.reactivity_score ?? null;
  const alleles = pra?.alleles ?? [];

  const praPercent = assay.pra_percent;
  const backendPanelPercent = reactivity?.score_percent;

  return (
    <div className="space-y-3">
      <CompactPanel title="PRA result">
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-2">
          <CompactMetric
            label="PRA"
            value={formatPercent(praPercent, 1)}
            sub={`${assay.positive_panel_wells ?? "—"} / ${
              assay.valid_panel_wells ?? "—"
            } panel wells`}
          />

          <CompactMetric
            label="Full panel reactivity"
            value={formatPercent(backendPanelPercent, 1)}
            sub={
              reactivity
                ? `${reactivity.positive_well_count} / ${reactivity.total_well_count} sample wells`
                : "—"
            }
          />

          <CompactMetric
            label="Mean corrected"
            value={formatNumber(assay.mean_corrected_frac_pos, 1)}
          />

          <CompactMetric
            label="Median corrected"
            value={formatNumber(assay.median_corrected_frac_pos, 1)}
          />

          <CompactMetric
            label="Max corrected"
            value={formatNumber(assay.max_corrected_frac_pos, 1)}
          />

          <CompactMetric
            label="Threshold"
            value={pra?.positivity_threshold ?? "—"}
          />

          <CompactMetric
            label="Weak"
            value={assay.n_weak_positive ?? 0}
          />

          <CompactMetric
            label="Moderate"
            value={assay.n_moderate_positive ?? 0}
          />

          <CompactMetric
            label="Strong"
            value={assay.n_strong_positive ?? 0}
          />
        </div>
      </CompactPanel>

      <CompactPanel title="Allele evidence">
        <div className="mb-2 text-xs text-neutral-500 dark:text-neutral-400">
          Ranked by positive carrier well fraction. Positive score means positive
          wells carrying the allele divided by all sample wells carrying the allele.
        </div>

        <AlleleEvidenceTable alleles={alleles} />
      </CompactPanel>

      <RunQcCompact run={run} qc={qc} />
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
