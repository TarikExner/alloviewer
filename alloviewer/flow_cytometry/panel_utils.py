import os
import re
from pydantic import BaseModel

from typing import Dict, Tuple, List, Optional, Literal, Any

from flowio import FlowData

from .panel import Panel

ChannelRole = Literal["IgG Marker", "Population Marker", "Scatter", "Time"]

class PanelRow(BaseModel):
    channel: str
    role: ChannelRole
    antibody: str = ""
    population: str = ""

_PANEL_CACHE: Dict[Tuple[str, int, int], Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]]] = {}

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
    text = f"{pnn or ''} {pns or ''}".strip()

    if re.search(r"\b(FSC|SSC)\b", text, re.IGNORECASE):
        return "Scatter"
    if re.search(r"\bTIME\b", text, re.IGNORECASE):
        return "Time"
    if re.search(r"\b(IGG|ISOTYPE|ISO)\b", text, re.IGNORECASE):
        return "IgG Marker"
    return "Population Marker"

def guess_population_name(pnn: str, pns: Optional[str], role: ChannelRole) -> str:
    """
    Returns a population label guess (for UI convenience).
    - Scatter -> "Lymphocytes"
    - IgG/Time -> ""
    - Marker -> based on common marker names found in PnS/PnN
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
    return bool(re.fullmatch(r"\s*time\s*", (pnn or ""), flags=re.IGNORECASE))

def _file_key(path: str) -> Tuple[str, int, int]:
    p = os.path.abspath(path)
    st = os.stat(p)
    return (p, int(st.st_mtime_ns), int(st.st_size))

def _sanitize_pns(pns: Optional[str]) -> str:
    if not pns:
        return ""
    # keep it simple, no heavy editing here
    return str(pns).strip()

def _extract_panel_rows_from_fcs(abs_path: str) -> Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]]:
    """
    Returns:
      signature: tuple of (PnN, PnS) in channel order
      rows: PanelRow list with channel=PnN, antibody=PnS
    """

    path_str = str(abs_path)

    try:
        fcs = FlowData(path_str, ignore_offset_error=False, only_text=True)
    except Exception:
        # mimic your current behavior
        fcs = FlowData(path_str, ignore_offset_error=True, only_text=True)

    # flowio uses channel numbers as keys ("1","2",...)
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

def get_panel_rows_cached(path: str) -> Tuple[Tuple[Tuple[str, str], ...], List[PanelRow]]:
    key = _file_key(path)
    hit = _PANEL_CACHE.get(key)
    if hit is not None:
        return hit

    sig, rows = _extract_panel_rows_from_fcs(path)
    _PANEL_CACHE[key] = (sig, rows)
    return sig, rows


def _get(r: Any, key: str, default: str = "") -> str:
    if isinstance(r, dict):
        return str(r.get(key, default) or default)
    return str(getattr(r, key, default) or default)


def build_panel_from_rows(panel_rows: List[Any]) -> Tuple["Panel", Dict[str, str]]:
    """
    Accepts list of dicts OR Pydantic PanelRow objects.
    Uses:
      - channel (PnN) as channel key
      - antibody as marker name (for Population Marker)
      - population as population label
    """
    # normalize roles
    rows = []
    for r in panel_rows:
        rows.append({
            "channel": _get(r, "channel").strip(),
            "role": _get(r, "role"),
            "antibody": _get(r, "antibody").strip(),
            "population": _get(r, "population").strip(),
        })

    # IgG channel
    igg_rows = [r for r in rows if r["role"] == "IgG Marker"]
    if len(igg_rows) != 1:
        got = [r["role"] for r in rows]
        raise ValueError(f"Expected exactly 1 IgG channel. Got {len(igg_rows)}. Roles seen: {got}")
    igg_channel = igg_rows[0]["channel"]

    # scatter channels (optional)
    def pick(name: str) -> Optional[str]:
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

    # markers: antibody name -> channel
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
        raise ValueError("No population markers. Please set at least one row to 'Population Marker' with antibody filled.")

    panel = Panel(
        fsc_a=fsc_a,
        fsc_h=fsc_h,
        ssc_a=ssc_a,
        igg=igg_channel,
        markers=markers,
    )
    return panel, marker_to_population

