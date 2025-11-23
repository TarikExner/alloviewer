import numpy as np
from typing import List

from .structs import Plate, PlateLayout, WellImage, WellResult

PRA_GENERIC_LAYOUT = PlateLayout(
    wells={
        'A1': 'negative', 'A2': 'sample',  'A3': 'sample',  'A4': 'sample',  'A5': 'sample',
        'A6': 'sample',   'A7': 'sample',  'A8': 'sample',  'A9': 'sample',  'A10': 'positive',

        'B1': 'negative', 'B2': 'sample',  'B3': 'sample',  'B4': 'sample',  'B5': 'sample',
        'B6': 'sample',   'B7': 'sample',  'B8': 'sample',  'B9': 'sample',  'B10': 'positive',

        'C1': 'sample',   'C2': 'sample',  'C3': 'sample',  'C4': 'sample',  'C5': 'sample',
        'C6': 'sample',   'C7': 'sample',  'C8': 'sample',  'C9': 'sample',  'C10': 'sample',

        'D1': 'sample',   'D2': 'sample',  'D3': 'sample',  'D4': 'sample',  'D5': 'sample',
        'D6': 'sample',   'D7': 'sample',  'D8': 'sample',  'D9': 'sample',  'D10': 'sample',

        'E1': 'sample',   'E2': 'sample',  'E3': 'sample',  'E4': 'sample',  'E5': 'sample',
        'E6': 'sample',   'E7': 'sample',  'E8': 'sample',  'E9': 'sample',  'E10': 'sample',

        'F1': 'sample',   'F2': 'sample',  'F3': 'sample',  'F4': 'sample',  'F5': 'sample',
        'F6': 'sample',   'F7': 'sample',  'F8': 'sample',  'F9': 'sample',  'F10': 'sample',
    }
)

PRA_GENERIC_IMAGE_ORDER=[
    'A1', 'B1', 'C1', 'D1', 'E1', 'F1',
    'F2', 'E2', 'D2', 'C2', 'B2', 'A2',
    'A3', 'B3', 'C3', 'D3', 'E3', 'F3',
    'F4', 'E4', 'D4', 'C4', 'B4', 'A4',
    'A5', 'B5', 'C5', 'D5', 'E5', 'F5',
    'F6', 'E6', 'D6', 'C6', 'B6', 'A6',
    'A7', 'B7', 'C7', 'D7', 'E7', 'F7',
    'F8', 'E8', 'D8', 'C8', 'B8', 'A8',
    'A9', 'B9', 'C9', 'D9', 'E9', 'F9',
    'F10', 'E10', 'D10', 'C10', 'B10', 'A10',
]

def frac_pos_raw(wr: WellResult) -> float:
    """Raw fraction positive in percent (0–100) for a WellResult."""
    n_pos = sum(1 for r in wr.rois if r.label == "pos")
    n_rois = len(wr.rois)
    if n_rois == 0:
        return np.nan
    return 100.0 * (n_pos / n_rois)

def convert_frac_pos_to_score(frac_pos: int) -> int:
    if frac_pos <= 10:
        return 1
    elif frac_pos <= 20:
        return 2
    elif frac_pos <= 50:
        return 4
    elif frac_pos <= 80:
        return 6
    return 8

def create_plate(layout: PlateLayout,
                 images: List[np.ndarray],
                 image_order: List[str],
                 image_paths: List[str]) -> Plate:
    plate = Plate(plate_id="SIM001")
    for i, well_id in enumerate(image_order):
        role = layout.wells[well_id]
        plate.add(
            WellImage(
                well_id,
                role=role,
                image=images[i],
                path=image_paths[i]
            )
        )

    return plate
