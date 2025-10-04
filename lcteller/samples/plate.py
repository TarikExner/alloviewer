from . import WellImage
from typing import Optional

class Plate:

    def __init__(self,
                 plate_id: str):
        self.id = plate_id
        self.wells: dict[str, WellImage] = {}


    def add_well(self,
                 well: WellImage,
                 well_id: Optional[str] = ""):
        if not well_id:
            well_id = well.id
        self.wells[well_id] = well

    def remove_well(self,
                    well_id: str):
        self.wells.pop(well_id)
        

