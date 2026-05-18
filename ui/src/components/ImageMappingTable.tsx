import type { WellID } from "../../types";

export function ImageMappingTable({
  imageOrder,
  imageSavedNames,
}: {
  imageOrder: WellID[];
  imageSavedNames: string[];
}) {
  return (
    <div className="rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 p-4 flex-1 min-h-0 overflow-hidden flex flex-col">
      <div className="shrink-0">
        <div className="font-medium">Image mapping</div>
        <div className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
          Images are assigned by upload order against the selected scan order.
        </div>
      </div>

      <div className="mt-3 flex-1 min-h-0 overflow-auto rounded-xl border dark:border-neutral-800">
        <table className="w-full text-sm">
          <thead className="bg-white dark:bg-neutral-900 sticky top-0 z-10">
            <tr className="border-b dark:border-neutral-800">
              <th className="text-left px-3 py-2 font-medium">#</th>
              <th className="text-left px-3 py-2 font-medium">Well</th>
              <th className="text-left px-3 py-2 font-medium">Image</th>
            </tr>
          </thead>

          <tbody>
            {imageOrder.length === 0 ? (
              <tr>
                <td
                  colSpan={3}
                  className="px-3 py-3 text-neutral-600 dark:text-neutral-400"
                >
                  No scan order set.
                </td>
              </tr>
            ) : (
              imageOrder.map((well, idx) => (
                <tr
                  key={well}
                  className="border-b dark:border-neutral-800 last:border-b-0"
                >
                  <td className="px-3 py-2 text-xs text-neutral-500">
                    {idx + 1}
                  </td>

                  <td className="px-3 py-2 font-mono text-xs">{well}</td>

                  <td className="px-3 py-2 truncate">
                    {imageSavedNames[idx] || "—"}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
