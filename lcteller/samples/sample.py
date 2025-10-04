import numpy as np
from typing import Literal

class WellImage:
    """
    WellImage class as a structure to store information about the
    respective well. This class is supposed to store all image
    and segmentation information as well as the final readouts.
    This class will be passed to and modified by the respective
    analysis functions.

    Attributes
    ----------
    id
        Well id, normally one of A01 to H12
    img
        image of the well as an np.ndarray in RGB
    assignment
        Specification if this well is either positive control (PC),
        negative control (NC) or a sample well (sample)
    mask
        Stores the mask of the well that filters out the non-cell
        area
    segmented
        stores the segmented image, gets calculated multiplying
        self.img*self.mask


    """

    def __init__(self,
                 id: str,
                 img: np.ndarray,
                 assignment: Literal["PC", "NC", "sample"]):

        self.img = img
        self.id = id
        self.sample_type = assignment
        self.mask: np.ndarray = np.zeros(shape = self.img.shape)
        self.segmented: np.ndarray = np.zeros(shape = self.img.shape)

    def _check_input_image(self,
                           img: np.ndarray):
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(
                f"Image needs to be supplied as RGB image. Shape is currently {img.shape}"
            )

    def create_segmented_image(self):
        if np.sum(self.mask) == 0:
            raise ValueError("Segmentation attempted using an empty mask")
        self.segmented = self.img*self.mask

    @property
    def segmented(self):
        return self._segmented

    @segmented.setter
    def segmented(self,
                  segmented: np.ndarray):
        self._segmented = segmented

    @property
    def default_gate(self):
        return self._default_gate

    @default_gate.setter
    def default_gate(self,
                     gate: str):
        self._default_gate = gate

        
