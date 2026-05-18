// ui/src/components/Toolbar.tsx
import React from "react";
import { ThemeToggle } from "./ThemeToggle";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

export function Toolbar({ title = "AlloViewer" }: { title?: string }) {
  const { t, i18n } = useTranslation();

  const changeLang = (lang: string) => {
    i18n.changeLanguage(lang);
    localStorage.setItem("lang", lang);
  };

  const LANGUAGES = [
    { code: "en", flag: "🇬🇧", label: "English" },
    { code: "de", flag: "🇩🇪", label: "Deutsch" },
    { code: "es", flag: "🇪🇸", label: "Español" },
    { code: "fr", flag: "🇫🇷", label: "Français" }
  ];

  return (
    <div className="w-full border-b bg-white/90 dark:bg-neutral-900/90 dark:border-neutral-800 backdrop-blur-md">
      <div className="w-full px-4 py-3 flex items-center justify-between">
        <Link to="/" className="font-semibold text-neutral-900 dark:text-neutral-100">
          {title}
        </Link>

        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-6 text-sm text-neutral-700 dark:text-neutral-300">
            <Link to="/about" className="hover:underline">
              {t("toolbar.about")}
            </Link>
            <Link to="/tutorial" className="hover:underline">
              {t("toolbar.tutorial")}
            </Link>
            <Link to="/docs" className="hover:underline">
              {t("toolbar.docs")}
            </Link>
            <Link to="/contact" className="hover:underline">
              {t("toolbar.contact")}
            </Link>
          </nav>

          {/* 🌍 Language flags */}
          <div className="flex items-center gap-1">
            {LANGUAGES.map((lang) => (
              <button
                key={lang.code}
                onClick={() => changeLang(lang.code)}
                title={lang.label}
                className={`px-1 text-lg transition hover:scale-110 ${
                  i18n.language === lang.code
                    ? "opacity-100"
                    : "opacity-60 hover:opacity-100"
                }`}
              >
                <span
                  role="img"
                  aria-label={lang.label}
                  className="select-none"
                >
                  {lang.flag}
                </span>
              </button>
            ))}
          </div>

          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}

