// ui/src/pages/Tutorial.tsx
import { useState } from "react";
import { Toolbar } from "../components/Toolbar";

export default function Tutorial() {
  const [tab, setTab] = useState<"cdc" | "xm">("cdc");

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold mb-6 text-center">Tutorial</h1>

        {/* Tab buttons */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex border rounded-xl overflow-hidden dark:border-neutral-700">
            <button
              onClick={() => setTab("cdc")}
              className={`px-6 py-2 text-sm font-medium transition ${
                tab === "cdc"
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                  : "bg-white dark:bg-neutral-900 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              }`}
            >
              CDC-PRA
            </button>
            <button
              onClick={() => setTab("xm")}
              className={`px-6 py-2 text-sm font-medium transition ${
                tab === "xm"
                  ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
                  : "bg-white dark:bg-neutral-900 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              }`}
            >
              Crossmatch
            </button>
          </div>
        </div>

        {tab === "cdc" ? <CDCTutorial /> : <CrossmatchTutorial />}
      </main>
    </div>
  );
}

/* ============ CDC-PRA tutorial ============ */
function CDCTutorial() {
  return (
    <div className="space-y-10">
      <h2 className="text-xl font-semibold mb-2 text-center">CDC-PRA Analysis</h2>

      {/* Step 1 */}
      <section>
        <h3 className="text-base font-medium mb-2">1. Upload plate layout</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          Start by uploading your <code>.xlsx</code> layout file. Each well should
          define whether it is a sample, positive, or negative control. The default
          layout can be generated automatically if no template is available.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder1.jpg"
            alt="Upload layout step"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>

      {/* Step 2 */}
      <section>
        <h3 className="text-base font-medium mb-2">2. Upload images</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          Upload the entire folder containing your well images. Image order will be
          detected automatically, but you can review and reorder them if needed.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder2.jpg"
            alt="Upload images step"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>

      {/* Step 3 */}
      <section>
        <h3 className="text-base font-medium mb-2">3. Review and run analysis</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          Verify that each well is assigned correctly. Adjust any wells if needed
          and press <strong>Run Analysis</strong>. Once processing completes, the
          summary and cytotoxicity maps will appear on the right.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder3.jpg"
            alt="Results step"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>
    </div>
  );
}

/* ============ Crossmatch tutorial ============ */
function CrossmatchTutorial() {
  return (
    <div className="space-y-10">
      <h2 className="text-xl font-semibold mb-2 text-center">Crossmatch Analysis</h2>

      {/* Step 1 */}
      <section>
        <h3 className="text-base font-medium mb-2">1. Upload images</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          Select and upload your image folder. Wells are automatically labeled as
          positive, negative, IgM, or sample based on their position in the plate.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder4.jpg"
            alt="Upload images crossmatch"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>

      {/* Step 2 */}
      <section>
        <h3 className="text-base font-medium mb-2">2. Adjust layout and column modes</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          Use the editor to adjust well roles or column modes (T, B, or T/B). The
          default layout sets rows A–C as controls and columns 1–9 as test samples.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder5.jpg"
            alt="Layout step crossmatch"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>

      {/* Step 3 */}
      <section>
        <h3 className="text-base font-medium mb-2">3. Run analysis</h3>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
          When all wells and modes are set, click <strong>Run Analysis</strong>. The
          cytotoxicity results and summarized statistics will appear automatically.
        </p>
        <div className="rounded-xl overflow-hidden border dark:border-neutral-800">
          <img
            src="/tutorial-placeholder6.jpg"
            alt="Run analysis crossmatch"
            className="w-full object-cover max-h-[300px] dark:opacity-90"
          />
        </div>
      </section>
    </div>
  );
}

