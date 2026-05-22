// ui/src/pages/Tutorial.tsx
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Toolbar } from "../components/Toolbar";

type TutorialTab = "cdcPra" | "cdcXm" | "fcxm";

type StepSectionProps = {
  title: string;
  body: string;
  children?: React.ReactNode;
};

function StepSection({ title, body, children }: StepSectionProps) {
  return (
    <section>
      <h3 className="text-base font-medium mb-2">{title}</h3>

      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-3xl">
        {body}
      </p>

      {children ? <div>{children}</div> : null}
    </section>
  );
}

function TutorialButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-6 py-2 text-sm font-medium transition ${
        active
          ? "bg-neutral-900 text-white dark:bg-white dark:text-neutral-900"
          : "bg-white dark:bg-neutral-900 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800"
      }`}
    >
      {children}
    </button>
  );
}

export default function Tutorial() {
  const { t } = useTranslation();
  const [tab, setTab] = useState<TutorialTab>("cdcPra");

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold mb-6 text-center">
          {t("tutorial.title")}
        </h1>

        <div className="flex justify-center mb-6">
          <div className="inline-flex border rounded-xl overflow-hidden dark:border-neutral-700">
            <TutorialButton
              active={tab === "cdcPra"}
              onClick={() => setTab("cdcPra")}
            >
              {t("tutorial.tabs.cdc_pra")}
            </TutorialButton>

            <TutorialButton
              active={tab === "cdcXm"}
              onClick={() => setTab("cdcXm")}
            >
              {t("tutorial.tabs.cdc_xm")}
            </TutorialButton>

            <TutorialButton
              active={tab === "fcxm"}
              onClick={() => setTab("fcxm")}
            >
              {t("tutorial.tabs.fcxm")}
            </TutorialButton>
          </div>
        </div>

        <div className="max-w-3xl mx-auto mb-10 rounded-2xl border bg-white dark:bg-neutral-900 dark:border-neutral-800 px-4 py-3 text-sm text-neutral-600 dark:text-neutral-400">
          {t("tutorial.video.text")}{" "}
          <a
            href={t("tutorial.video.href")}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-neutral-900 underline underline-offset-4 hover:text-neutral-700 dark:text-neutral-100 dark:hover:text-neutral-300"
          >
            {t("tutorial.video.link")}
          </a>
          .
        </div>

        {tab === "cdcPra" ? <CDCPRATutorial /> : null}
        {tab === "cdcXm" ? <CDCXMTutorial /> : null}
        {tab === "fcxm" ? <FCXMTutorial /> : null}
      </main>
    </div>
  );
}

/* ============ CDC-PRA tutorial ============ */
function CDCPRATutorial() {
  const { t } = useTranslation();

  return (
    <div className="space-y-10">
      <header className="text-center">
        <h2 className="text-xl font-semibold mb-2">
          {t("tutorial.cdc_pra.title")}
        </h2>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto">
          {t("tutorial.cdc_pra.intro")}
        </p>
      </header>

      <StepSection
        title={t("tutorial.cdc_pra.step1_title")}
        body={t("tutorial.cdc_pra.step1_body")}
      >
        <a
          href="/examples/cdc-pra-layout-example.xlsx"
          download
          className="inline-flex items-center rounded-xl border px-3 py-2 text-sm bg-white hover:bg-neutral-50 dark:bg-neutral-900 dark:hover:bg-neutral-800 dark:border-neutral-700"
        >
          {t("tutorial.cdc_pra.download_example")}
        </a>
      </StepSection>

      <StepSection
        title={t("tutorial.cdc_pra.step2_title")}
        body={t("tutorial.cdc_pra.step2_body")}
      />

      <StepSection
        title={t("tutorial.cdc_pra.step3_title")}
        body={t("tutorial.cdc_pra.step3_body")}
      />

      <StepSection
        title={t("tutorial.cdc_pra.step4_title")}
        body={t("tutorial.cdc_pra.step4_body")}
      />
    </div>
  );
}

/* ============ CDC-XM tutorial ============ */
function CDCXMTutorial() {
  const { t } = useTranslation();

  return (
    <div className="space-y-10">
      <header className="text-center">
        <h2 className="text-xl font-semibold mb-2">
          {t("tutorial.cdc_xm.title")}
        </h2>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto">
          {t("tutorial.cdc_xm.intro")}
        </p>
      </header>

      <StepSection
        title={t("tutorial.cdc_xm.step1_title")}
        body={t("tutorial.cdc_xm.step1_body")}
      />

      <StepSection
        title={t("tutorial.cdc_xm.step2_title")}
        body={t("tutorial.cdc_xm.step2_body")}
      />

      <StepSection
        title={t("tutorial.cdc_xm.step3_title")}
        body={t("tutorial.cdc_xm.step3_body")}
      />

      <StepSection
        title={t("tutorial.cdc_xm.step4_title")}
        body={t("tutorial.cdc_xm.step4_body")}
      />
    </div>
  );
}

/* ============ FC-XM tutorial ============ */
function FCXMTutorial() {
  const { t } = useTranslation();

  return (
    <div className="space-y-10">
      <header className="text-center">
        <h2 className="text-xl font-semibold mb-2">
          {t("tutorial.fcxm.title")}
        </h2>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-3xl mx-auto">
          {t("tutorial.fcxm.intro")}
        </p>
      </header>

      <StepSection
        title={t("tutorial.fcxm.step1_title")}
        body={t("tutorial.fcxm.step1_body")}
      />

      <StepSection
        title={t("tutorial.fcxm.step2_title")}
        body={t("tutorial.fcxm.step2_body")}
      />

      <StepSection
        title={t("tutorial.fcxm.step3_title")}
        body={t("tutorial.fcxm.step3_body")}
      />

      <StepSection
        title={t("tutorial.fcxm.step4_title")}
        body={t("tutorial.fcxm.step4_body")}
      />
    </div>
  );
}
