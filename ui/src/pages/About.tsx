// ui/src/pages/About.tsx
import { Toolbar } from "../components/Toolbar";
import { useTranslation } from "react-i18next";

export default function About() {
  const { t } = useTranslation();

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10 space-y-10">
        <h1 className="text-2xl font-semibold mb-4">
          {t("about.title")}
        </h1>

        {/* --- Purpose --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">{t("about.purpose_heading")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            {t("about.purpose_text")}
          </p>
        </section>

        {/* --- Why it matters --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">{t("about.why_heading")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            {t("about.why_text")}
          </p>
        </section>

        {/* --- How it works --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">{t("about.how_heading")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            {t("about.how_text")}
          </p>
        </section>

        {/* --- Development --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">{t("about.dev_heading")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            {t("about.dev_text", {
              cassian: "Dr. med. Dr. rer. nat. Cassian Afting",
              tarik: "Dr. med. Tarik Exner",
            })}
          </p>
        </section>

        {/* --- License / Use --- */}
        <section>
          <h2 className="text-lg font-medium mb-2">{t("about.license_heading")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
            {t("about.license_text")}
          </p>
        </section>
      </main>
    </div>
  );
}

