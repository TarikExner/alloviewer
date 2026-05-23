Installation
============

This page describes how to install AlloViewer for local development and
documentation builds.

Repository Setup
----------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/TarikExner/alloviewer.git
   cd alloviewer

Replace the repository URL if the final public repository uses a different
address.

Python Environment
------------------

Create and activate a virtual environment.

On Windows PowerShell:

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

On Linux or macOS:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

Then upgrade ``pip``:

.. code-block:: bash

   python -m pip install --upgrade pip

Install the Python Package
--------------------------

Install the package from the repository root:

.. code-block:: bash

   python -m pip install .

For editable development installs, use:

.. code-block:: bash

   python -m pip install -e .

Verify that the package can be imported:

.. code-block:: bash

   python -c "import alloviewer"

If this command fails, fix the package installation before building the
documentation.

Install Documentation Dependencies
----------------------------------

Install the documentation dependencies:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

A minimal ``docs/requirements.txt`` should contain:

.. code-block:: text

   sphinx
   sphinx-book-theme
   sphinx-autodoc-typehints
   myst-parser
   myst-nb

If your documentation imports modules that require additional runtime
dependencies, add those dependencies to ``docs/requirements.txt`` as well.

Build the Documentation Locally
-------------------------------

From the repository root, run:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

For a stricter build that treats warnings as errors, run:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html -W

On Windows, open the built documentation with:

.. code-block:: powershell

   start docs/build/html/index.html

On Linux or macOS, open:

.. code-block:: bash

   xdg-open docs/build/html/index.html

Documentation Folder Structure
------------------------------

The documentation source files are stored in:

.. code-block:: text

   docs/source/

The local HTML build output is written to:

.. code-block:: text

   docs/build/html/

The build output should not be committed to Git.

Read the Docs
-------------

Read the Docs uses the ``.readthedocs.yaml`` file in the repository root.

A suitable minimal configuration is:

.. code-block:: yaml

   version: 2

   build:
     os: ubuntu-24.04
     tools:
       python: "3.12"

   sphinx:
     configuration: docs/source/conf.py

   python:
     install:
       - method: pip
         path: .
       - requirements: docs/requirements.txt

Common Installation Problems
----------------------------

ModuleNotFoundError
~~~~~~~~~~~~~~~~~~~~~~

This usually means that either the package itself is not installed, or one of
its dependencies is missing.

First check:

.. code-block:: bash

   python -c "import alloviewer"

If that fails, install the package:

.. code-block:: bash

   python -m pip install -e .

If a dependency is missing during the documentation build, add it to
``docs/requirements.txt``.

Import-Time Side Effects
~~~~~~~~~~~~~~~~~~~~~~~~

Sphinx imports Python modules when using autodoc. Module imports should not load
large model files, access local data folders, start web servers, or require
frontend build artifacts.

If a module does this, move that behavior behind a function call or command-line
entry point.

Large Files
~~~~~~~~~~~

Model weights, temporary uploads, generated reports, microscopy images, FCS
files, and local build outputs should not be committed to the repository unless
there is a clear reason.

Recommended ``.gitignore`` entries:

.. code-block:: text

   __pycache__/
   *.py[cod]
   *.pyo

   docs/build/
   tree.txt

   *.pth
   *.pt

   app/data/
   app/data/**

Frontend Development
--------------------

The frontend lives in ``ui/``.

Typical frontend commands are:

.. code-block:: bash

   cd ui
   npm install
   npm run dev
   npm run build

Backend Development
-------------------

The FastAPI backend lives in ``app/``.

A typical development command is:

.. code-block:: bash

   uvicorn app.main:app --reload --port 8000

Depending on your packaging setup, you may need to run this from the repository
root with the virtual environment activated.
