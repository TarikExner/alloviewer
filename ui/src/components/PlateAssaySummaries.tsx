import { EmptySummary } from "./SummaryBlocks";


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

function formatControlPercentWithRange({
  value,
  min,
  max,
  range,
  digits = 1,
}: {
  value: unknown;
  min?: unknown;
  max?: unknown;
  range?: unknown;
  digits?: number;
}) {
  const main = formatPercent(value, digits);

  const minN = Number(min);
  const maxN = Number(max);

  if (Number.isFinite(minN) && Number.isFinite(maxN)) {
    return `${main} (range: ${minN.toFixed(digits)}–${maxN.toFixed(digits)}%)`;
  }

  if (range === null || range === undefined) return main;

  const rangeN = Number(range);
  if (!Number.isFinite(rangeN)) {
    return `${main} (range: ${String(range)})`;
  }

  return `${main} (range: ${rangeN.toFixed(digits)}%)`;
}

function RunValidityCompactPanel({ run }: { run: any }) {
  return (
    <CompactPanel title="Run validity">
      <div className="overflow-x-auto">
        <div className="grid min-w-[1200px] grid-cols-5 gap-2">
          <CompactMetric label="Status" value={run.status ?? "—"} />

          <CompactMetric
            label="Positive Control % positive"
            value={formatControlPercentWithRange({
              value: run.pc_mean_raw,
              min: run.pc_min_raw ?? run.pc_replicate_min,
              max: run.pc_max_raw ?? run.pc_replicate_max,
              range: run.pc_replicate_range,
              digits: 1,
            })}
          />

          <CompactMetric
            label="Negative Control % positive"
            value={formatControlPercentWithRange({
              value: run.nc_mean_raw,
              min: run.nc_min_raw ?? run.nc_replicate_min,
              max: run.nc_max_raw ?? run.nc_replicate_max,
              range: run.nc_replicate_range,
              digits: 1,
            })}
          />

          <CompactMetric
            label="Dynamic range between Positive and Negative Control"
            value={formatPercent(run.dynamic_range, 1)}
          />

          <CompactMetric
            label="Controls"
            value={`${run.n_positive_controls ?? "—"} positive control(s) · ${
              run.n_negative_controls ?? "—"
            } negative control(s)`}
          />
        </div>
      </div>

      {run.control_warnings?.length ? (
        <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          {run.control_warnings.join(", ")}
        </div>
      ) : null}
    </CompactPanel>
  );
}

function QCCompactPanel({ qc }: { qc: any }) {
  return (
    <CompactPanel title="QC">
      <div className="overflow-x-auto">
        <div className="grid min-w-[1200px] grid-cols-6 gap-2">
          <CompactMetric label="Total wells" value={qc.total_wells ?? "—"} />

          <CompactMetric label="Valid wells" value={qc.valid_wells ?? "—"} />

          <CompactMetric
            label="Mean cell count"
            value={formatNumber(qc.mean_n_rois, 1)}
          />

          <CompactMetric
            label="Mean uncertain fraction"
            value={formatNumber(qc.mean_uncertain_fraction, 3)}
          />

          <CompactMetric
            label="Low cell count wells"
            value={qc.low_roi_wells?.length ?? 0}
          />

          <CompactMetric
            label="High uncertain wells"
            value={qc.high_uncertain_wells?.length ?? 0}
          />
        </div>
      </div>

      {qc.warnings?.length ? (
        <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          {qc.warnings.join(", ")}
        </div>
      ) : null}

      <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
        {qc.low_roi_wells?.length ? (
          <div className="rounded-xl border px-3 py-2 text-xs dark:border-neutral-800">
            <div className="font-medium text-neutral-700 dark:text-neutral-300">
              Low cell count wells
            </div>
            <div className="mt-1 text-neutral-600 dark:text-neutral-400">
              {qc.low_roi_wells.join(", ")}
            </div>
          </div>
        ) : null}

        {qc.high_uncertain_wells?.length ? (
          <div className="rounded-xl border px-3 py-2 text-xs dark:border-neutral-800">
            <div className="font-medium text-neutral-700 dark:text-neutral-300">
              High uncertain wells
            </div>
            <div className="mt-1 text-neutral-600 dark:text-neutral-400">
              {qc.high_uncertain_wells.join(", ")}
            </div>
          </div>
        ) : null}
      </div>
    </CompactPanel>
  );
}

function PRASummaryValuesPanel({
  assay,
  pra,
  reactivity,
}: {
  assay: any;
  pra: any;
  reactivity: any;
}) {
  const praPercent = assay.pra_percent;
  const backendPanelPercent = reactivity?.score_percent;

  return (
    <CompactPanel title="Summary values">
      <div className="overflow-x-auto">
        <div className="grid min-w-[1700px] grid-cols-9 gap-2">
          <CompactMetric
            label="Panel reactivity"
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

          <CompactMetric label="Weak" value={assay.n_weak_positive ?? 0} />

          <CompactMetric
            label="Moderate"
            value={assay.n_moderate_positive ?? 0}
          />

          <CompactMetric label="Strong" value={assay.n_strong_positive ?? 0} />
        </div>
      </div>
    </CompactPanel>
  );
}

export function PRASummaryGrid({
  summary,
  result,
  onDownloadSummary,
  canDownloadSummary = false,
  summaryBusy = false,
  summaryError = null,
}: {
  summary: any | null;
  result?: any | null;
  onDownloadSummary?: () => void;
  canDownloadSummary?: boolean;
  summaryBusy?: boolean;
  summaryError?: string | null;
}) {
  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const assay = summary.assay_result ?? {};
  const qc = summary.qc ?? {};
  const pra = result?.pra_analysis ?? null;
  const reactivity = pra?.reactivity_score ?? null;
  const alleles = pra?.alleles ?? [];

  return (
    <div className="space-y-4">
      <PRASummaryValuesPanel
        assay={assay}
        pra={pra}
        reactivity={reactivity}
      />

      <CompactPanel title="Allele evidence">
        <div className="mb-2 text-xs text-neutral-500 dark:text-neutral-400">
          Ranked by positive carrier well fraction. Positive score means positive
          wells carrying the allele divided by all sample wells carrying the allele.
        </div>

        <AlleleEvidenceTable alleles={alleles} />
      </CompactPanel>

      <RunValidityCompactPanel run={run} />

      <QCCompactPanel qc={qc} />

      {onDownloadSummary ? (
        <button
          type="button"
          onClick={onDownloadSummary}
          disabled={!canDownloadSummary}
          className="w-full py-3 rounded-xl border bg-blue-600 hover:bg-blue-700 text-white
                     disabled:opacity-50 disabled:hover:bg-blue-600
                     dark:border-blue-500"
        >
          {summaryBusy ? "Preparing PDF..." : "Download Summary"}
        </button>
      ) : null}

      {summaryError ? (
        <div className="text-sm text-red-600 dark:text-red-400">
          {summaryError}
        </div>
      ) : null}
    </div>
  );
}

function formatPercentWithRange(
  value: unknown,
  range: unknown,
  digits = 1
) {
  const main = formatPercent(value, digits);

  if (range === null || range === undefined) return main;

  const n = Number(range);
  if (!Number.isFinite(n)) return `${main} (range: ${String(range)})`;

  return `${main} (range: ${n.toFixed(digits)}%)`;
}

function CrossmatchRunValiditySection({ run }: { run: any }) {
  return (
    <>
      <CompactPanel title="Run validity">
        <div className="overflow-x-auto">
          <div className="grid min-w-[1100px] grid-cols-5 gap-2">
            <CompactMetric label="Status" value={run.status ?? "—"} />

            <CompactMetric
              label="Positive Control % positive"
              value={formatPercentWithRange(
                run.pc_mean_raw,
                run.pc_replicate_range,
                1
              )}
            />

            <CompactMetric
              label="Negative Control % positive"
              value={formatPercentWithRange(
                run.nc_mean_raw,
                run.nc_replicate_range,
                1
              )}
            />

            <CompactMetric
              label="Dynamic range between Positive and Negative Control"
              value={formatPercent(run.dynamic_range, 1)}
            />

            <CompactMetric
              label="Controls"
              value={`${run.n_positive_controls ?? "—"} PC · ${
                run.n_negative_controls ?? "—"
              } NC`}
            />
          </div>
        </div>

        {run.control_warnings?.length ? (
          <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
            {run.control_warnings.join(", ")}
          </div>
        ) : null}
      </CompactPanel>
    </>
  );
}

function CrossmatchQCSection({ qc }: { qc: any }) {
  return (
    <CompactPanel title="QC">
      <div className="overflow-x-auto">
        <div className="grid min-w-[1100px] grid-cols-6 gap-2">
          <CompactMetric label="Total wells" value={qc.total_wells ?? "—"} />

          <CompactMetric label="Valid wells" value={qc.valid_wells ?? "—"} />

          <CompactMetric
            label="Mean cell count"
            value={formatNumber(qc.mean_n_rois, 1)}
          />

          <CompactMetric
            label="Mean uncertain fraction"
            value={formatNumber(qc.mean_uncertain_fraction, 3)}
          />

          <CompactMetric
            label="Low cell count wells"
            value={qc.low_roi_wells?.length ?? 0}
          />

          <CompactMetric
            label="High uncertain wells"
            value={qc.high_uncertain_wells?.length ?? 0}
          />
        </div>
      </div>

      {qc.warnings?.length ? (
        <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300">
          {qc.warnings.join(", ")}
        </div>
      ) : null}

      <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
        {qc.low_roi_wells?.length ? (
          <div className="rounded-xl border px-3 py-2 text-xs dark:border-neutral-800">
            <div className="font-medium text-neutral-700 dark:text-neutral-300">
              Low cell count wells
            </div>
            <div className="mt-1 text-neutral-600 dark:text-neutral-400">
              {qc.low_roi_wells.join(", ")}
            </div>
          </div>
        ) : null}

        {qc.high_uncertain_wells?.length ? (
          <div className="rounded-xl border px-3 py-2 text-xs dark:border-neutral-800">
            <div className="font-medium text-neutral-700 dark:text-neutral-300">
              High uncertain wells
            </div>
            <div className="mt-1 text-neutral-600 dark:text-neutral-400">
              {qc.high_uncertain_wells.join(", ")}
            </div>
          </div>
        ) : null}
      </div>
    </CompactPanel>
  );
}

export function CrossmatchSummaryGrid({
  summary,
  onDownloadSummary,
  canDownloadSummary = false,
  summaryBusy = false,
  summaryError = null,
}: {
  summary: any | null;
  onDownloadSummary?: () => void;
  canDownloadSummary?: boolean;
  summaryBusy?: boolean;
  summaryError?: string | null;
}) {
  if (!summary) return <EmptySummary />;

  const run = summary.run_validity ?? {};
  const qc = summary.qc ?? {};

  return (
    <div className="space-y-4">
      <CrossmatchRunValiditySection run={run} />

      <CrossmatchQCSection qc={qc} />

      {onDownloadSummary ? (
        <button
          type="button"
          onClick={onDownloadSummary}
          disabled={!canDownloadSummary}
          className="w-full py-3 rounded-xl border bg-blue-600 hover:bg-blue-700 text-white
                     disabled:opacity-50 disabled:hover:bg-blue-600
                     dark:border-blue-500"
        >
          {summaryBusy ? "Preparing PDF..." : "Download Summary"}
        </button>
      ) : null}

      {summaryError ? (
        <div className="text-sm text-red-600 dark:text-red-400">
          {summaryError}
        </div>
      ) : null}
    </div>
  );
}
