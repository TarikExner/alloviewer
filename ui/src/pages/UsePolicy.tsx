// src/pages/UsePolicy.tsx
import { Toolbar } from "../components/Toolbar";
import { useTranslation } from "react-i18next";

function PolicySection({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <section>
      <h2 className="text-lg font-medium mb-2">{title}</h2>
      <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl">
        {body}
      </p>
    </section>
  );
}

export default function UsePolicy() {
  const { t } = useTranslation();

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10 space-y-10">
        <div>
          <h1 className="text-2xl font-semibold mb-4">
            {t("use_policy.title")}
          </h1>

          <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200 max-w-3xl">
            <strong>{t("use_policy.banner_strong")}</strong>{" "}
            {t("use_policy.banner_text")}
          </div>
        </div>

        <PolicySection
          title={t("use_policy.research_heading")}
          body={t("use_policy.research_text")}
        />

        <PolicySection
          title={t("use_policy.no_clinical_heading")}
          body={t("use_policy.no_clinical_text")}
        />

        <PolicySection
          title={t("use_policy.review_heading")}
          body={t("use_policy.review_text")}
        />

        <PolicySection
          title={t("use_policy.validation_heading")}
          body={t("use_policy.validation_text")}
        />

        <PolicySection
          title={t("use_policy.liability_heading")}
          body={t("use_policy.liability_text")}
        />

        <PolicySection
          title={t("use_policy.data_heading")}
          body={t("use_policy.data_text")}
        />
      </main>
    </div>
  );
}
