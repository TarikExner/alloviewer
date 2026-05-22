import { useState, useEffect } from "react";
import { Toolbar } from "../components/Toolbar";
import { useTranslation } from "react-i18next";


// const CONTACT_PERSONS = [
//   {
//     name: "Dr. med. Dr. rer. nat. Cassian Fucking Afting",
//     role: "Project lead",
//     unit: "DRK Frankfurt",
//     email: "c.afting@blutspende.de",
//     img: "/CA.jpg",
//   },
//   {
//     name: "Dr. med. Tarik Exner",
//     role: "Image analysis / pipeline",
//     unit: "University Hospital Heidelberg",
//     email: "Tarik.Exner@med.uni-heidelberg.de",
//     img: "/TE.jpg",
//   },
// ];
// 
// const GITHUB = {
//   url: "https://github.com/TarikExner/alloviewer",
//   text: "View GitHub repository",
//   note:
//     "This is mainly for people who are used to React / TypeScript / imaging. If that’s not you, just send us an email.",
// };


export default function Contact() {
  const [openImage, setOpenImage] = useState<string | null>(null);
  const { t } = useTranslation();

  // close on ESC
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenImage(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  //  Get translated persons from JSON
  const persons = t("contact.persons", { returnObjects: true }) as {
    name: string;
    role: string;
    unit: string;
  }[];

  const GITHUB = {
    url: "https://github.com/TarikExner/alloviewer",
    text: t("contact.github_text"),
    note: t("contact.github_note"),
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100 flex flex-col">
      <Toolbar />

      <main className="flex-1 w-full max-w-5xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold mb-6">{t("contact.heading")}</h1>

        <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-8 max-w-2xl">
          {t("contact.intro")}
        </p>

        <div className="grid gap-6 md:grid-cols-2">
          {persons.map((person, i) => {
            const img = i === 0 ? "/CA.jpg" : "/TE.jpg";
            const email =
              i === 0
                ? "c.afting@blutspende.de"
                : "Tarik.Exner@med.uni-heidelberg.de";

            return (
              <div
                key={email}
                className="rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white/70 dark:bg-neutral-900/60 p-6"
              >
                <div className="flex items-start gap-4 mb-4">
                  <button
                    type="button"
                    onClick={() => img && setOpenImage(img)}
                    className="h-14 w-14 rounded-full overflow-hidden bg-neutral-200 dark:bg-neutral-700 flex-shrink-0 focus:outline-none focus:ring-2 focus:ring-blue-500/70"
                  >
                    <img
                      src={img}
                      alt={person.name}
                      className="h-full w-full object-cover"
                    />
                  </button>

                  <div>
                    <h2 className="text-lg font-medium leading-tight">
                      {person.name}
                    </h2>
                    <p className="text-sm text-neutral-600 dark:text-neutral-400">
                      {person.unit}
                    </p>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <p>
                    Email:{" "}
                    <a
                      href={`mailto:${email}`}
                      className="text-blue-600 dark:text-blue-400 hover:underline"
                    >
                      {email}
                    </a>
                  </p>
                  <p>{t("contact.role_label", { defaultValue: "Role" })}: {person.role}</p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Developer / GitHub block */}
        <div className="mt-10 rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-900/40 p-6">
          <h2 className="text-base font-medium mb-2">{t("contact.developers")}</h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4 max-w-2xl">
            {GITHUB.note}
          </p>
          <a
            href={GITHUB.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 text-sm px-4 py-2 rounded-lg bg-neutral-900 text-white dark:bg-white dark:text-neutral-900 hover:opacity-90 transition"
          >
            {GITHUB.text}
            <span aria-hidden="true">↗</span>
          </a>
        </div>

        <div className="mt-10 text-xs text-neutral-500 dark:text-neutral-500">
          {t("contact.disclaimer")}
        </div>
      </main>

      {/* lightbox */}
      {openImage && (
        <div
          onClick={() => setOpenImage(null)}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 px-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative max-w-3xl w-full bg-neutral-950/60 rounded-xl overflow-hidden flex items-center justify-center"
          >
            <img src={openImage} alt="Contact" className="max-h-[85vh] max-w-full object-contain" />
            <button
              onClick={() => setOpenImage(null)}
              className="absolute top-3 right-3 bg-black/50 text-white text-xs px-3 py-1 rounded-full hover:bg-black"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

