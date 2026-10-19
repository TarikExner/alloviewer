from typing import Protocol, Dict, Any, List
import numpy as np

class ISegmenter(Protocol):
    """
    Define the callable interface for image segmentation.
    """
    def __call__(self, img: np.ndarray) -> Dict[str, Any]: """
        Specify a segmentation call accepting an image and returning a result mapping.

        Parameters
        ----------
        img : numpy.ndarray
            Input image array.

        Returns
        -------
        dict[str, Any]
            Segmentation result mapping defined by the implementing segmenter.
        """; ...

class IFeatureExtractor(Protocol):
    """
    Define the callable interface for extracting per-instance image features.
    """
    def __call__(self, img: np.ndarray, inst: np.ndarray) -> List[Dict[str, Any]]: """
        Specify extraction of ROI feature dictionaries from image and instance labels.

        Parameters
        ----------
        img : numpy.ndarray
            Input image array.
        inst : numpy.ndarray
            Integer instance-label image with background label zero.

        Returns
        -------
        list of dict
            Per-instance feature mappings extracted from the supplied image and labels.
        """; ...

class ICalibrator(Protocol):
    """
    Define the control-well calibration interface.
    """
    def fit(self, pc_wells: List[List[Dict[str, Any]]], nc_wells: List[List[Dict[str, Any]]]) -> Dict[str, Any]: """
        Specify fitting calibration data from positive and negative control ROIs.

        Parameters
        ----------
        pc_wells : List[List[Dict[str, Any]]]
            Positive-control wells, each containing a list of ROI feature dictionaries.
        nc_wells : List[List[Dict[str, Any]]]
            Negative-control wells, each containing a list of ROI feature dictionaries.

        Returns
        -------
        dict[str, Any]
            Calibration values fitted from positive- and negative-control ROI measurements.
        """; ...

class IClassifier(Protocol):
    """
    Define the calibrated ROI classifier interface.
    """
    def __init__(self, calib: Dict[str, Any]): """
        Specify construction from a calibration mapping.

        Parameters
        ----------
        calib : Dict[str, Any]
            Fitted calibration values used by the classifier.
        """; ...
    def __call__(self, rois: List[Dict[str, Any]]) -> List[Dict[str, Any]]: """
        Specify classification of ROI feature dictionaries.

        Parameters
        ----------
        rois : List[Dict[str, Any]]
            Per-cell ROI feature dictionaries.

        Returns
        -------
        list of dict
            Classified ROI mappings as defined by the implementing classifier.
        """; ...

