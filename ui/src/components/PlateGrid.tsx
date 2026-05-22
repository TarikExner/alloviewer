import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ALL_WELLS, type WellMap, type WellType } from "../types";
import { ROWS, COLS } from "../plateConfig";
import { ROLE_STYLES, ROLE_SWATCH } from "../roleStyles";

const MENU_TYPES: WellType[] = ["positive", "negative", "sample"];

export function PlateGridEditor({
  wells,
  setWells,
}: {
  wells: WellMap;
  setWells: (next: WellMap) => void;
}) {
  const { t } = useTranslation();

  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });
  const [menuWell, setMenuWell] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    }

    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpen(false);
    }

    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);

    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, []);

  function roleLabel(role: WellType) {
    return t(`PlateGrid.roles.${role}`);
  }

  function openMenu(wellId: string, evt: React.MouseEvent) {
    evt.preventDefault();
    setMenuWell(wellId);
    setMenuPos({ x: evt.clientX, y: evt.clientY });
    setMenuOpen(true);
  }

  function applyType(t: WellType) {
    if (!menuWell) return;
    setWells({ ...wells, [menuWell]: t } as WellMap);
    setMenuOpen(false);
  }

  function clearAll() {
    setWells({} as WellMap);
  }

  function resetAll() {
    setWells(
      Object.fromEntries(ALL_WELLS.map((w) => [w, "sample"])) as WellMap
    );
  }

  return (
    <div className="relative rounded-2xl border bg-white p-4 dark:bg-neutral-900 dark:border-neutral-800">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-neutral-900 dark:text-neutral-100">
          {t("PlateGrid.title")}
        </h3>

        <div className="flex items-center gap-3">
          <Legend label={roleLabel("positive")} cls={ROLE_SWATCH.positive} />
          <Legend label={roleLabel("negative")} cls={ROLE_SWATCH.negative} />
          <Legend label={roleLabel("sample")} cls={ROLE_SWATCH.sample} />
          <Legend label={roleLabel("empty")} cls={ROLE_SWATCH.empty} />
        </div>
      </div>

      <div className="overflow-auto">
        <div className="inline-grid grid-cols-[auto_repeat(12,minmax(1.8rem,2.25rem))] gap-1">
          <div />

          {COLS.map((c) => (
            <div
              key={c}
              className="text-xs text-center text-neutral-600 dark:text-neutral-400"
            >
              {c}
            </div>
          ))}

          {ROWS.map((r) => (
            <React.Fragment key={r}>
              <div className="text-xs text-right pr-1 text-neutral-600 dark:text-neutral-400">
                {r}
              </div>

              {COLS.map((c) => {
                const id = `${r}${c}` as const;
                const role = ((wells as any)[id] ?? "empty") as WellType;

                return (
                  <button
                    key={id}
                    type="button"
                    onClick={(e) => openMenu(id, e)}
                    onContextMenu={(e) => openMenu(id, e)}
                    title={t("PlateGrid.well_title", {
                      well: id,
                      role: roleLabel(role),
                    })}
                    className={[
                      "rounded-md border text-[10px] flex items-center justify-center aspect-square min-w-7 min-h-7",
                      ROLE_STYLES[role],
                      "hover:ring-2 hover:ring-offset-0 hover:ring-neutral-300 dark:hover:ring-neutral-600 select-none",
                    ].join(" ")}
                  >
                    {r}
                    {c}
                  </button>
                );
              })}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <button
          type="button"
          onClick={clearAll}
          className="px-3 py-1.5 border rounded-lg bg-white hover:bg-neutral-50
                     dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
          title={t("PlateGrid.actions.clear_title")}
        >
          {t("PlateGrid.actions.clear")}
        </button>

        <button
          type="button"
          onClick={resetAll}
          className="px-3 py-1.5 border rounded-lg bg-white hover:bg-neutral-50
                     dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700 dark:text-neutral-200"
          title={t("PlateGrid.actions.reset_title")}
        >
          {t("PlateGrid.actions.reset")}
        </button>
      </div>

      {menuOpen ? (
        <div
          ref={menuRef}
          style={{ left: menuPos.x, top: menuPos.y }}
          className="fixed z-50 bg-white border rounded-lg shadow-md p-1 text-sm
                     dark:bg-neutral-900 dark:border-neutral-700 dark:text-neutral-200"
        >
          {MENU_TYPES.map((tRole) => (
            <button
              key={tRole}
              type="button"
              onClick={() => applyType(tRole)}
              className="px-3 py-1.5 w-full text-left hover:bg-neutral-100 rounded-md
                         dark:hover:bg-neutral-800"
            >
              {roleLabel(tRole)}
            </button>
          ))}

          <div className="h-px bg-neutral-200 dark:bg-neutral-800 my-1" />

          <button
            type="button"
            onClick={() => applyType("empty")}
            className="px-3 py-1.5 w-full text-left hover:bg-neutral-100 rounded-md
                       dark:hover:bg-neutral-800"
          >
            {roleLabel("empty")}
          </button>
        </div>
      ) : null}
    </div>
  );
}

function Legend({ cls, label }: { cls: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-neutral-700 dark:text-neutral-300">
      <span className={["w-3.5 h-3.5 rounded border", cls].join(" ")} />
      <span>{label}</span>
    </div>
  );
}
