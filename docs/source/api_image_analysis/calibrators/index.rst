Calibrators
===========

Calibrators estimate decision parameters from positive and negative control
wells.

The calibrators in this module operate on nested well data. Each well contains
ROI dictionaries with red- and green-channel measurements, usually stored as
``mean_r`` and ``mean_g``.

.. toctree::
   :maxdepth: 1

   pcnc_median_calibrator
   pcnc_mean_calibrator
   pcnc_gaussian_rg_calibrator
   pcnc_gaussian_2d_calibrator
