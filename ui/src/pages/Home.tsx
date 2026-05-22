import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Toolbar } from "../components/Toolbar";
import { useTranslation } from "react-i18next";

type HomeItem = {
  label: string;
  to: string;
};

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const a = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}

function wedgePath(
  cx: number,
  cy: number,
  r: number,
  startAngleDeg: number,
  endAngleDeg: number
) {
  const start = polarToCartesian(cx, cy, r, endAngleDeg);
  const end = polarToCartesian(cx, cy, r, startAngleDeg);
  const largeArcFlag = endAngleDeg - startAngleDeg <= 180 ? "0" : "1";

  return [
    `M ${cx} ${cy}`,
    `L ${end.x} ${end.y}`,
    `A ${r} ${r} 0 ${largeArcFlag} 1 ${start.x} ${start.y}`,
    "Z",
  ].join(" ");
}

function CircleMenu({ items }: { items: HomeItem[] }) {
  const nav = useNavigate();

  const slices = useMemo(() => {
    const n = items.length;
    const step = 360 / n;
    return items.map((it, i) => {
      const start = i * step;
      const end = (i + 1) * step;
      const mid = start + step / 2;
      return { ...it, start, end, mid };
    });
  }, [items]);

  const cx = 50;
  const cy = 50;
  const r = 48;
  const labelR = 28;

  return (
    <div className="w-full flex items-center justify-center">
      <svg
        viewBox="0 0 100 100"
        className="w-[280px] h-[280px] sm:w-[320px] sm:h-[320px]"
        aria-label="Choose an app mode"
      >
        <circle
          cx={cx}
          cy={cy}
          r={r}
          className="fill-white dark:fill-neutral-900 stroke-neutral-200 dark:stroke-neutral-800"
          strokeWidth="1.2"
        />

        {slices.map((s) => {
          const d = wedgePath(cx, cy, r, s.start, s.end);
          const lp = polarToCartesian(cx, cy, labelR, s.mid);

          return (
            <g
              key={s.to}
              role="link"
              tabIndex={0}
              onClick={() => nav(s.to)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") nav(s.to);
              }}
              className="cursor-pointer outline-none"
            >
              <path
                d={d}
                className="fill-white dark:fill-neutral-900 stroke-neutral-200 dark:stroke-neutral-800
                           hover:fill-neutral-50 dark:hover:fill-neutral-800 transition-colors"
                strokeWidth="1.2"
              />
              <text
                x={lp.x}
                y={lp.y}
                textAnchor="middle"
                dominantBaseline="middle"
                className="select-none fill-neutral-900 dark:fill-neutral-100 font-medium"
                style={{ fontSize: 5.8 }}
              >
                {s.label}
              </text>
            </g>
          );
        })}

        <circle
          cx={cx}
          cy={cy}
          r={9}
          className="fill-neutral-50 dark:fill-neutral-950 stroke-neutral-200 dark:stroke-neutral-800"
          strokeWidth="1.2"
        />
      </svg>
    </div>
  );
}

export default function Home() {
  const nav = useNavigate();
  const { t } = useTranslation();

  const items: HomeItem[] = [
    { label: "CDC-PRA", to: "/cdc" },
    { label: "CDC-XM", to: "/crossmatch" },
    { label: "FC-XM", to: "/fcxm" },
  ];

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <div className="flex-1 flex items-center justify-center">
        <div className="w-full max-w-xl p-6">
          <h1 className="text-center text-2xl font-semibold mb-6">
            AlloViewer
          </h1>

          <div className="rounded-2xl border dark:border-neutral-800 bg-white/60 dark:bg-neutral-900/30 p-6">
            <CircleMenu items={items} />

            <div className="mt-6 flex justify-center">
              <button
                type="button"
                onClick={() => nav("/tutorial")}
                className="rounded-xl border px-5 py-2.5 text-sm font-medium
                           bg-neutral-900 text-white hover:bg-neutral-800
                           dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200
                           dark:border-neutral-700 transition-colors"
              >
                {t("home.get_started")}
              </button>
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
            <strong>{t("home.disclaimer_strong")}</strong>{" "}
            {t("home.disclaimer_text")}{" "}
            <button
              type="button"
              onClick={() => nav("/use-policy")}
              className="font-medium underline underline-offset-4 hover:opacity-80"
            >
              {t("home.disclaimer_link")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
