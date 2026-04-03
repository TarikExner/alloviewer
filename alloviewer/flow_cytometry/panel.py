from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class Panel:
    # scatter
    fsc_a: Optional[str] = None
    fsc_h: Optional[str] = None
    ssc_a: Optional[str] = None

    # readout channel (anti-IgG)
    igg: Optional[str] = None

    # markers used for cell typing (exclude IgG)
    markers: Optional[Dict[str, str]] = None  # marker name -> channel label

    def marker_names(self) -> List[str]:
        return list((self.markers or {}).keys())

    def marker_channels(self) -> List[str]:
        return list((self.markers or {}).values())

