from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class Panel:
    """Flow cytometry panel channel mapping.

    Parameters
    ----------
    fsc_a : str or None, optional
        Channel label for forward scatter area.
    fsc_h : str or None, optional
        Channel label for forward scatter height.
    ssc_a : str or None, optional
        Channel label for side scatter area.
    igg : str or None, optional
        Channel label for IgG.
    markers : dict or None, optional
        Mapping from marker names to channel labels.
    """

    fsc_a: Optional[str] = None
    fsc_h: Optional[str] = None
    ssc_a: Optional[str] = None
    igg: Optional[str] = None

    markers: Optional[Dict[str, str]] = None

    def marker_names(self) -> List[str]:
        """Return marker names.

        Returns
        -------
        list of str
            Marker names from ``markers``. Returns an empty list if no markers
            are defined.
        """
        return list((self.markers or {}).keys())

    def marker_channels(self) -> List[str]:
        """Return marker channel labels.

        Returns
        -------
        list of str
            Channel labels from ``markers``. Returns an empty list if no
            markers are defined.
        """
        return list((self.markers or {}).values())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Panel":
        """Create a panel from a dictionary.

        Parameters
        ----------
        data : mapping
            Panel definition. Known top-level keys are ``"fsc_a"``, ``"fsc_h"``,
            ``"ssc_a"``, ``"igg"``, and ``"markers"``. Unknown top-level keys
            are treated as marker-name to channel-label pairs.

        Returns
        -------
        Panel
            Parsed panel object.
        """
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
