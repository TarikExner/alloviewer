from dataclasses import dataclass
from typing import Dict, List, Optional, Mapping, Any


@dataclass
class Panel:
    fsc_a: Optional[str] = None
    fsc_h: Optional[str] = None
    ssc_a: Optional[str] = None
    igg: Optional[str] = None

    # marker name -> channel label
    markers: Optional[Dict[str, str]] = None

    def marker_names(self) -> List[str]:
        return list((self.markers or {}).keys())

    def marker_channels(self) -> List[str]:
        return list((self.markers or {}).values())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Panel":
        known_fields = {"fsc_a", "fsc_h", "ssc_a", "igg", "markers"}

        explicit_markers = dict(data.get("markers") or {})

        inferred_markers = {
            key: value
            for key, value in data.items()
            if key not in known_fields
        }

        return cls(
            fsc_a=data.get("fsc_a"),
            fsc_h=data.get("fsc_h"),
            ssc_a=data.get("ssc_a"),
            igg=data.get("igg"),
            markers={**explicit_markers, **inferred_markers},
        )
