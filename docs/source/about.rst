About
=====

AlloViewer is a web-based software platform for standardized and traceable
interpretation of physical HLA antibody diagnostics.

It supports automated analysis of:

* CDC-based HLA antibody specificity testing using commercial test-cell panels,
  also referred to as CDC-PRA
* complement-dependent cytotoxicity crossmatch assays, also referred to as CDC-XM
* flow cytometry crossmatch data from ``.fcs`` files, also referred to as FC-XM

The project contains a Python/FastAPI backend and a TypeScript/React frontend.
This documentation focuses on the Python ``alloviewer`` package.

Purpose
-------

AlloViewer was built to provide a standardized and traceable way to analyze
physical HLA antibody diagnostics.

Manual interpretation of CDC microscopy and FC-XM data can vary between
operators, runs, local protocols, and software settings. AlloViewer aims to
reduce this variability by applying fixed analysis steps, preserving quantitative
intermediate outputs, and making assay-specific workflows accessible through a
web interface and API.

AlloViewer is not intended to replace expert laboratory judgment. Results require
independent review, local validation, and compliance with applicable
institutional and regulatory rules before any operational or clinical use.

Research Use Only
-----------------

.. warning::

   AlloViewer is intended for research use only.

   It is an experimental software tool. It is not certified or approved for
   clinical diagnostics, treatment decisions, transplant allocation,
   donor-recipient compatibility decisions, or patient management.

Supported Workflows
-------------------

CDC-PRA
~~~~~~~

The CDC-PRA workflow supports CDC-based HLA antibody specificity analysis using
commercial test-cell panels.

Users upload an Excel-based panel layout and a folder of microscopy images.
AlloViewer maps images to wells, segments lymphocytes, classifies cells using
positive and negative controls, calculates well-level cytotoxicity, and maps the
observed reactivity pattern to the uploaded panel layout.

The goal is to support candidate HLA antibody specificity inference from CDC
panel reactivity patterns.

CDC-XM
~~~~~~

The CDC-XM workflow supports image-based complement-dependent cytotoxicity
crossmatch analysis.

Users upload microscopy images, define well roles, review the plate layout and
scan order, and run automated well-level analysis.

Supported well roles include:

* sample
* positive control
* negative control
* IgM control
* empty

The IgM control can be used to check whether Dithiothreitol, or DTT, treatment
effectively reduces IgM-mediated reactivity.

FC-XM
~~~~~

The FC-XM workflow supports flow cytometry crossmatch analysis from ``.fcs``
files.

Users upload FCS files, assign them to negative control, positive control, or
sample cards, review the detected marker panel, and run automated analysis.

The FC-XM workflow includes:

* FCS upload and sample assignment
* marker panel review
* quality control
* lymphocyte selection
* marker-informed population detection
* population-specific IgG readout extraction
* summary report generation

Technology Stack
----------------

Backend
~~~~~~~

* Python
* FastAPI
* Uvicorn
* NumPy
* PyTorch
* scikit-image / OpenCV-style image processing
* flow cytometry processing tools
* ReportLab for PDF output
* Render deployment

Frontend
~~~~~~~~

* TypeScript
* React
* Vite
* Tailwind CSS
* i18next
* Vercel deployment

Links
-----

* Public website: https://alloviewer.org
* Backend API: https://alloviewer.onrender.com
* Documentation: https://alloviewer.readthedocs.io
* Video tutorials: https://www.youtube.com/@AlloViewer

Known Limitations
-----------------

* The current version is intended for research and workflow evaluation.
* The software has not been established as a stand-alone clinical decision
  system.
* External and prospective multi-center validation is required before routine
  clinical decision support.
* FC-XM analysis may depend on marker panel design, compensation, instrument
  settings, sample quality, and local interpretation rules.
* CDC image analysis may be affected by staining failures, poor focus, debris,
  rare artifacts, unusual lymphocyte morphology, or local imaging conditions.
* Free hosting tiers may sleep or restart services.
* Large uploads may hit hosting memory, time, or storage limits.
