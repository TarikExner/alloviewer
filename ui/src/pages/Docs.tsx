// ui/src/pages/Docs.tsx
import { Toolbar } from "../components/Toolbar";

export default function Docs() {
  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10 space-y-10">
        <h1 className="text-2xl font-semibold mb-4">Documentation</h1>

        {/* --- Overview --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">Overview</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            AlloViewer is a web-based analysis platform for evaluating lymphocyte
            cytotoxicity assays. It quantifies cell-mediated lysis and summarizes
            donor-specific responses in standardized reports.
          </p>
        </section>

        {/* --- Modules --- */}
        <section>
          <h2 className="text-lg font-medium mb-4">Modules</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-900/60 p-6">
              <h3 className="font-semibold text-base mb-2">CDC-PRA</h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                Estimates panel-reactive antibodies (PRA) based on fluorescence
                or brightfield images from multiple donor wells.
              </p>
              <ul className="text-sm text-neutral-600 dark:text-neutral-400 list-disc list-inside space-y-1">
                <li>Input: one or more images per donor well</li>
                <li>Output: % cytotoxicity, donor specificity distribution, PRA summary</li>
              </ul>
            </div>

            <div className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-900/60 p-6">
              <h3 className="font-semibold text-base mb-2">CDC-XM</h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                Performs crossmatch analysis between specific donor and recipient
                wells using the same cytotoxicity quantification pipeline.
              </p>
              <ul className="text-sm text-neutral-600 dark:text-neutral-400 list-disc list-inside space-y-1">
                <li>Input: paired donor–recipient well images</li>
                <li>Output: % cytotoxicity, overall crossmatch result</li>
              </ul>
            </div>
          </div>
        </section>

        {/* --- Data format --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">Image Format</h2>
          <ul className="text-sm text-neutral-600 dark:text-neutral-400 list-disc list-inside space-y-1">
            <li>Accepted file types: <code>.tif</code>, <code>.png</code></li>
            <li>Single- or multi-channel supported</li>
            <li>Maximum file size: 50 MB per image</li>
            <li>Recommended naming: <code>sampleID_condition.tif</code></li>
          </ul>
        </section>

        {/* --- Output --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">Output Files</h2>
          <ul className="text-sm text-neutral-600 dark:text-neutral-400 list-disc list-inside space-y-1">
            <li><code>.csv</code> summary table (donor, % cytotoxicity, result)</li>
            <li><code>.png</code> or <code>.jpg</code> visualization (optional)</li>
            <li>All outputs include a date and version identifier</li>
          </ul>
        </section>

        {/* --- Reproducibility --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">Reproducibility</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            Each analysis run is processed using a fixed version of the AlloViewer
            pipeline. Version numbers are embedded in result files to ensure that
            future updates remain traceable.
          </p>
        </section>

        {/* --- Disclaimer --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">Disclaimer</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            AlloViewer is designed for research use only. Do not upload
            patient-identifying data or use the results for direct clinical
            decision-making.
          </p>
        </section>
      </main>
    </div>
  );
}

