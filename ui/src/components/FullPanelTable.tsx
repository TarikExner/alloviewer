import { type PanelRow } from "./PanelTable";

export function FullPanelTable({
  rows,
  onChange,
}: {
  rows: PanelRow[];
  onChange: (next: PanelRow[]) => void;
}) {
  function setCell(
    idx: number,
    key: "role" | "antibody" | "population",
    value: string
  ) {
    onChange(rows.map((r, i) => (i === idx ? { ...r, [key]: value } : r)));
  }

  function removeRow(idx: number) {
    onChange(rows.filter((_, i) => i !== idx));
  }

  return (
    <div className="rounded-xl border dark:border-neutral-800 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-white dark:bg-neutral-900">
          <tr className="border-b dark:border-neutral-800">
            <th className="text-left px-3 py-2 font-medium">Channel</th>
            <th className="text-left px-3 py-2 font-medium">Type</th>
            <th className="text-left px-3 py-2 font-medium">Antibody</th>
            <th className="text-left px-3 py-2 font-medium">Population name</th>
            <th className="w-10 px-2 py-2" aria-label="Remove channel" />
          </tr>
        </thead>

        <tbody className="bg-white dark:bg-neutral-900">
          {rows.length === 0 ? (
            <tr>
              <td
                colSpan={5}
                className="px-3 py-3 text-neutral-600 dark:text-neutral-400"
              >
                Upload FCS files to load panel.
              </td>
            </tr>
          ) : (
            rows.map((r, idx) => (
              <tr
                key={r.channel + idx}
                className="border-b dark:border-neutral-800 last:border-b-0"
              >
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="font-mono text-xs">{r.channel}</span>
                </td>

                <td className="px-3 py-2">
                  <select
                    value={r.role}
                    onChange={(e) => setCell(idx, "role", e.target.value)}
                    className="w-full rounded-lg border px-2 py-1 bg-white
                               dark:bg-neutral-950 dark:border-neutral-700"
                  >
                    <option value="Scatter">Scatter</option>
                    <option value="Population Marker">Population Marker</option>
                    <option value="IgG Marker">IgG Marker</option>
                  </select>
                </td>

                <td className="px-3 py-2">
                  <input
                    value={r.antibody}
                    onChange={(e) => setCell(idx, "antibody", e.target.value)}
                    className="w-full rounded-lg border px-2 py-1 bg-white
                               dark:bg-neutral-950 dark:border-neutral-700"
                    placeholder="e.g. CD3"
                  />
                </td>

                <td className="px-3 py-2">
                  <input
                    value={r.population}
                    onChange={(e) => setCell(idx, "population", e.target.value)}
                    className="w-full rounded-lg border px-2 py-1 bg-white
                               dark:bg-neutral-950 dark:border-neutral-700"
                    placeholder="e.g. T cells"
                  />
                </td>

                <td className="px-2 py-2 text-right">
                  <button
                    type="button"
                    onClick={() => removeRow(idx)}
                    className="inline-flex items-center justify-center w-6 h-6 rounded-lg border
                               bg-white hover:bg-neutral-50 text-neutral-600
                               dark:bg-neutral-950 dark:hover:bg-neutral-800
                               dark:border-neutral-700 dark:text-neutral-300"
                    title={`Remove channel ${r.channel}`}
                    aria-label={`Remove channel ${r.channel}`}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
