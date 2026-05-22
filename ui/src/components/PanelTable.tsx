import { useTranslation } from "react-i18next";

export type ChannelRole = "Scatter" | "Population Marker" | "IgG Marker";

export type PanelRow = {
  channel: string;
  role: ChannelRole;
  antibody: string;
  population: string;
};

export function PanelTable({
  rows,
  onChange,
}: {
  rows: PanelRow[];
  onChange: (next: PanelRow[]) => void;
}) {
  const { t } = useTranslation();

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

  function roleLabel(role: ChannelRole) {
    return t(`PanelTable.roles.${role}`);
  }

  return (
    <div className="rounded-xl border dark:border-neutral-800 overflow-hidden">
      <div className="max-h-56 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white dark:bg-neutral-900 z-10">
            <tr className="border-b dark:border-neutral-800">
              <th className="text-left px-3 py-2 font-medium">
                {t("PanelTable.columns.channel")}
              </th>
              <th className="text-left px-3 py-2 font-medium">
                {t("PanelTable.columns.type")}
              </th>
              <th className="text-left px-3 py-2 font-medium">
                {t("PanelTable.columns.antibody")}
              </th>
              <th className="text-left px-3 py-2 font-medium">
                {t("PanelTable.columns.population_name")}
              </th>
              <th
                className="w-10 px-2 py-2"
                aria-label={t("PanelTable.actions.remove_channel")}
              />
            </tr>
          </thead>

          <tbody className="bg-white dark:bg-neutral-900">
            {rows.length === 0 ? (
              <tr>
                <td
                  colSpan={5}
                  className="px-3 py-3 text-neutral-600 dark:text-neutral-400"
                >
                  {t("PanelTable.empty")}
                </td>
              </tr>
            ) : (
              rows.map((r, idx) => (
                <tr
                  key={r.channel + idx}
                  className="border-b dark:border-neutral-800"
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
                      <option value="Scatter">{roleLabel("Scatter")}</option>
                      <option value="Population Marker">
                        {roleLabel("Population Marker")}
                      </option>
                      <option value="IgG Marker">
                        {roleLabel("IgG Marker")}
                      </option>
                    </select>
                  </td>

                  <td className="px-3 py-2">
                    <input
                      value={r.antibody}
                      onChange={(e) => setCell(idx, "antibody", e.target.value)}
                      className="w-full rounded-lg border px-2 py-1 bg-white
                                 dark:bg-neutral-950 dark:border-neutral-700"
                      placeholder={t("PanelTable.placeholders.antibody")}
                    />
                  </td>

                  <td className="px-3 py-2">
                    <input
                      value={r.population}
                      onChange={(e) =>
                        setCell(idx, "population", e.target.value)
                      }
                      className="w-full rounded-lg border px-2 py-1 bg-white
                                 dark:bg-neutral-950 dark:border-neutral-700"
                      placeholder={t("PanelTable.placeholders.population")}
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
                      title={t("PanelTable.actions.remove_channel_named", {
                        channel: r.channel,
                      })}
                      aria-label={t("PanelTable.actions.remove_channel_named", {
                        channel: r.channel,
                      })}
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
    </div>
  );
}
