import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from flowio import FlowData
from pydantic import BaseModel

from .panel import Panel


ChannelRole = Literal["IgG Marker", "Population Marker", "Scatter", "Time"]


class PanelRow(BaseModel):
    """Single channel row used for panel setup.

    Parameters
    ----------
    channel : str
        FCS channel name, usually the PnN value.
    role : ChannelRole
        Functional role assigned to the channel.
    antibody : str, optional
        Antibody or marker name, usually derived from the PnS value.
    population : str, optional
        User-facing population label linked to the marker.
    """

    channel: str
    role: ChannelRole
    antibody: str = ""
    population: str = ""


_PANEL_CACHE: Dict[
    Tuple[str, int, int],
    Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]],
] = {}

_MARKER_TO_POP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bCD3\b", re.IGNORECASE), "T-cells"),
    (re.compile(r"\bCD4\b", re.IGNORECASE), "CD4-T-cells"),
    (re.compile(r"\bCD8\b", re.IGNORECASE), "CD8-T-cells"),
    (re.compile(r"\bCD19\b", re.IGNORECASE), "B-cells"),
    (re.compile(r"\bCD20\b", re.IGNORECASE), "B-cells"),
    (re.compile(r"\bCD56\b", re.IGNORECASE), "NK-cells"),
    (re.compile(r"\bCD16\b", re.IGNORECASE), "NK-cells / neutrophils"),
    (re.compile(r"\bCD14\b", re.IGNORECASE), "Monocytes"),
    (re.compile(r"\bCD66B\b", re.IGNORECASE), "Neutrophils"),
    (re.compile(r"\bCD45\b", re.IGNORECASE), "Leukocytes"),
    (re.compile(r"\bHLA-DR\b", re.IGNORECASE), "Antigen-presenting-cells"),
]


def guess_role(pnn: str, pns: Optional[str] = None) -> ChannelRole:
    """Infer a channel role from FCS channel labels.

    Parameters
    ----------
    pnn : str
        FCS PnN channel name.
    pns : str or None, optional
        FCS PnS channel description.

    Returns
    -------
    ChannelRole
        Guessed channel role.

    Notes
    -----
    Scatter and time channels are detected from PnN/PnS text. IgG-like or
    isotype markers are assigned to ``"IgG Marker"``. All other channels are
    assigned to ``"Population Marker"``.
    """
    text = f"{pnn or ''} {pns or ''}".strip()

    if re.search(r"\b(FSC|SSC)\b", text, re.IGNORECASE):
        return "Scatter"
    if re.search(r"\bTIME\b", text, re.IGNORECASE):
        return "Time"
    if re.search(r"\b(IGG|ISOTYPE|ISO)\b", text, re.IGNORECASE):
        return "IgG Marker"

    return "Population Marker"


def guess_population_name(
    pnn: str,
    pns: Optional[str],
    role: ChannelRole,
) -> str:
    """Infer a population label for a channel.

    Parameters
    ----------
    pnn : str
        FCS PnN channel name.
    pns : str or None
        FCS PnS channel description.
    role : ChannelRole
        Channel role.

    Returns
    -------
    str
        Guessed population label. Returns an empty string when no useful label
        can be inferred.
    """
    if role == "Scatter":
        return "Lymphocytes"
    if role == "IgG Marker":
        return "Reacting Antibodies"
    if role == "Time":
        return ""

    text = f"{pnn or ''} {pns or ''}".strip()

    for pat, pop in _MARKER_TO_POP:
        if pat.search(text):
            return pop

    return ""


def is_time_channel(pnn: str) -> bool:
    """Return whether a PnN label is a time channel.

    Parameters
    ----------
    pnn : str
        FCS PnN channel name.

    Returns
    -------
    bool
        ``True`` if the channel name is exactly ``"time"`` ignoring case and
        surrounding whitespace.
    """
    return bool(re.fullmatch(r"\s*time\s*", (pnn or ""), flags=re.IGNORECASE))


def _file_key(path: str) -> Tuple[str, int, int]:
    """Build a cache key for an FCS file.

    Parameters
    ----------
    path : str
        Path to an FCS file.

    Returns
    -------
    tuple
        Absolute path, modification time in nanoseconds, and file size.
    """
    p = os.path.abspath(path)
    st = os.stat(p)
    return p, int(st.st_mtime_ns), int(st.st_size)


def _sanitize_pns(pns: Optional[str]) -> str:
    """Normalize a PnS value.

    Parameters
    ----------
    pns : str or None
        Raw PnS value.

    Returns
    -------
    str
        Stripped PnS value, or an empty string if missing.
    """
    if not pns:
        return ""

    return str(pns).strip()


def _extract_panel_rows_from_fcs(
    abs_path: str,
) -> Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]]:
    """Extract panel setup rows from FCS text metadata.

    Parameters
    ----------
    abs_path : str
        Path to an FCS file.

    Returns
    -------
    signature : tuple
        Tuple of ``(PnN, PnS)`` pairs in channel order.
    rows : list of PanelRow
        Initial panel rows with channel names, guessed roles, antibody labels,
        and empty population labels.

    Notes
    -----
    The FCS file is read with ``only_text=True``. If strict offset parsing
    fails, loading is retried with ``ignore_offset_error=True``.
    """
    path_str = str(abs_path)

    try:
        fcs = FlowData(path_str, ignore_offset_error=False, only_text=True)
    except Exception:
        fcs = FlowData(path_str, ignore_offset_error=True, only_text=True)

    ch_dict = fcs.channels
    ch_keys = sorted(ch_dict.keys(), key=lambda k: int(k))

    signature: List[Tuple[str, str]] = []
    rows: List[PanelRow] = []

    for k in ch_keys:
        info = ch_dict[k]
        pnn = str(info.get("PnN", "")).strip()
        pns = _sanitize_pns(info.get("PnS", ""))

        signature.append((pnn, pns))

        rows.append(
            PanelRow(
                channel=pnn,
                role=guess_role(pnn),
                antibody=pns,
                population="",
            )
        )

    return tuple(signature), rows


def get_panel_rows_cached(
    path: str,
) -> Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]]:
    """Return cached panel rows for an FCS file.

    Parameters
    ----------
    path : str
        Path to an FCS file.

    Returns
    -------
    signature : tuple
        Tuple of ``(PnN, PnS)`` pairs in channel order.
    rows : list of PanelRow
        Panel rows extracted from the FCS metadata.

    Notes
    -----
    The cache key includes absolute path, modification time, and file size.
    """
    key = _file_key(path)
    hit = _PANEL_CACHE.get(key)

    if hit is not None:
        return hit

    sig, rows = _extract_panel_rows_from_fcs(path)
    _PANEL_CACHE[key] = (sig, rows)

    return sig, rows


def _get(r: Any, key: str, default: str = "") -> str:
    """Read a string field from a dictionary or object.

    Parameters
    ----------
    r : Any
        Dictionary-like object or object with attributes.
    key : str
        Field name.
    default : str, optional
        Fallback value. The default is an empty string.

    Returns
    -------
    str
        Field value converted to string, or ``default`` when missing.
    """
    if isinstance(r, dict):
        return str(r.get(key, default) or default)

    return str(getattr(r, key, default) or default)


def build_panel_from_rows(panel_rows: List[Any]) -> Tuple["Panel", Dict[str, str]]:
    """Build a panel and marker-to-population map from panel rows.

    Parameters
    ----------
    panel_rows : list
        List of dictionaries or :class:`PanelRow` objects. Each row must provide
        ``channel``, ``role``, ``antibody``, and ``population`` fields.

    Returns
    -------
    panel : Panel
        Panel object containing scatter channels, IgG channel, and marker
        channel mapping.
    marker_to_population : dict
        Mapping from marker name to population label.

    Raises
    ------
    ValueError
        If exactly one IgG marker is not present, or if no population markers
        with antibody names are present.
    """
    rows = []

    for r in panel_rows:
        rows.append(
            {
                "channel": _get(r, "channel").strip(),
                "role": _get(r, "role"),
                "antibody": _get(r, "antibody").strip(),
                "population": _get(r, "population").strip(),
            }
        )

    igg_rows = [r for r in rows if r["role"] == "IgG Marker"]

    if len(igg_rows) != 1:
        got = [r["role"] for r in rows]
        raise ValueError(
            f"Expected exactly 1 IgG channel. Got {len(igg_rows)}. "
            f"Roles seen: {got}"
        )

    igg_channel = igg_rows[0]["channel"]

    def pick(name: str) -> Optional[str]:
        """Return a channel matching a preferred scatter name.

        Parameters
        ----------
        name : str
            Preferred channel name.

        Returns
        -------
        str or None
            Matching channel name, or ``None`` if absent.
        """
        for r in rows:
            if r["channel"] == name:
                return r["channel"]

        for r in rows:
            if r["channel"].lower() == name.lower():
                return r["channel"]

        return None

    fsc_a = pick("FSC-A")
    fsc_h = pick("FSC-H")
    ssc_a = pick("SSC-A")

    markers: Dict[str, str] = {}
    marker_to_population: Dict[str, str] = {}

    for r in rows:
        if r["role"] != "Population Marker":
            continue

        marker_name = r["antibody"]

        if not marker_name:
            continue

        markers[marker_name] = r["channel"]
        marker_to_population[marker_name] = r["population"]

    if not markers:
        raise ValueError(
            "No population markers. Please set at least one row to "
            "'Population Marker' with antibody filled."
        )

    panel = Panel(
        fsc_a=fsc_a,
        fsc_h=fsc_h,
        ssc_a=ssc_a,
        igg=igg_channel,
        markers=markers,
    )

    return panel, marker_to_population
