# Configuration file for the Sphinx documentation builder.
#
# Docs:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# -- Path setup --------------------------------------------------------------

# docs/source/conf.py -> repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# -- Project information -----------------------------------------------------

project = "AlloViewer"
copyright = "2026, Tarik Exner, Cassian Afting"
author = "Tarik Exner, Cassian Afting"
release = "0.1.0"


# -- General configuration ---------------------------------------------------

extensions = [
    # Core Sphinx API documentation
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",

    # Copy button for code blocks
    "sphinx_copybutton",

    # Type hints in API docs
    # Keep this after napoleon.
    "sphinx_autodoc_typehints",

    # MyST Markdown + notebooks.
    # Do not also load myst_parser, nbsphinx, or nbsphinx_link.
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"


# -- Autodoc / Autosummary ---------------------------------------------------

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}

autodoc_typehints = "description"
autoclass_content = "class"


# -- Napoleon: NumPy-style docstrings ----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True


# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


# -- MyST / notebook configuration ------------------------------------------

# Allows .md files in addition to .rst files.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Do not execute notebooks during the docs build.
# This is safer for Read the Docs and avoids failures from missing local data.
nb_execution_mode = "off"

# Suppress notebook execution warnings when execution is disabled.
nb_execution_show_tb = False


# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "AlloViewer"
html_static_path = ["_static"]


# -- sphinx-book-theme options ----------------------------------------------

html_theme_options = {
    "repository_url": "https://github.com/TarikExner/alloviewer",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "path_to_docs": "docs/source",
    "home_page_in_toc": True,
    "show_navbar_depth": 3,
}


# -- Copybutton options ------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |PS C:\\.*> "
copybutton_prompt_is_regexp = True
