from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure, morphology, filters


# ===================== Config =====================

@dataclass
class QCMonitorConfig:
    # --- touching rule ---
    allowed_neighbors: int = 0          # 1 => doublets allowed, triplets+ excluded
    connectivity: int = 8               # 8-connected touching (use 4 to be stricter)

    # --- border ---
    exclude_border_touching: bool = True

    # --- area outliers (robust) ---
    area_z_max_isolated: float = 3.5
    area_z_max_touching: float = 4.5
    min_instance_area: int = 0

    # --- shape ---
    min_circularity: float = 0.4
    max_eccentricity: float = 0.95
    min_solidity: float = 0.80

    # --- boundary confidence gap (needs probs) ---
    use_boundary_gap: bool = True
    min_boundary_gap: float = 0.15

    # --- intensity / focus (optional) ---
    use_intensity_focus: bool = True
    max_saturation_frac: float = 0.02
    min_snr: float = 3.0
    min_focus_varlap: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QCMonitorConfig":
        d = dict(d or {})
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class QCMonitor:
    """
    Crowding:
      - Build label adjacency with 8-connected touching.
      - Let K = allowed_neighbors.
      - Any ROI with degree > K is a violator.
      - For every violating edge (u,v) where deg(u)>K or deg(v)>K,
        mark BOTH u and v with reason 'touching_violation' and exclude them.
    Other QC rules stay as before.
    """
    def __init__(self, cfg: Optional[QCMonitorConfig] = None):
        self.cfg = cfg or QCMonitorConfig()

    def __call__(
        self,
        instance_labels: np.ndarray,
        probs: Optional[Dict[str, np.ndarray]] = None,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        lab = instance_labels.astype(np.int32, copy=False)
        max_id = int(lab.max())

        if max_id <= 0:
            return {
                "instances_filtered": lab.copy(),
                "exclude_ids": [],
                "roi_table": [],
                "well": {
                    "n_instances": 0,
                    "border_touch_frac": 0.0,
                    "boundary_gap_median": np.nan,
                    "focus_vlap": np.nan,
                    "snr": np.nan,
                    "saturation_frac": np.nan,
                    "excluded_frac": 0.0,
                    "touching_pairs_violating": [],
                },
            }

        # ---------- geometry ----------
        props = measure.regionprops(lab)
        areas = np.array([p.area for p in props], dtype=np.float32)
        perims = np.array([p.perimeter if p.perimeter > 0 else 1.0 for p in props], dtype=np.float32)
        circularity = 4.0 * np.pi * areas / (perims ** 2)
        eccentricity = np.array([getattr(p, "eccentricity", 0.0) for p in props], dtype=np.float32)
        solidity = np.array([getattr(p, "solidity", 1.0) for p in props], dtype=np.float32)
        centroids = np.array([p.centroid for p in props], dtype=np.float32)  # (y, x)

        # ---------- border touching ----------
        border_touch = self._border_touching(lab, max_id)

        # ---------- adjacency (touching) ----------
        neighbors, edges = self._adjacency(lab, connectivity=self.cfg.connectivity)
        degrees = np.array([len(neighbors.get(i+1, set())) for i in range(max_id)], dtype=np.int32)

        # Find violating edges (any endpoint degree > allowed_neighbors)
        K = int(self.cfg.allowed_neighbors)
        violating_edges: List[Tuple[int, int]] = []
        violating_ids: set[int] = set()
        for a, b in edges:
            if (len(neighbors.get(a, ())) > K) or (len(neighbors.get(b, ())) > K):
                violating_edges.append((a, b))
                violating_ids.add(a); violating_ids.add(b)

        # ---------- NN distance (info only) ----------
        nn_dist = self._nearest_neighbor_distances(centroids)

        # ---------- boundary confidence gap ----------
        gap = None
        if self.cfg.use_boundary_gap and probs is not None:
            pc = probs.get("cell", None)
            pb = probs.get("bound", None)
            if (pc is not None) and (pb is not None):
                gap = self._boundary_gap(lab, pc, pb)

        # ---------- intensity / focus ----------
        sat_frac = None
        snr = None
        vlap = None
        if self.cfg.use_intensity_focus and (image is not None):
            sat_frac, snr, vlap = self._intensity_focus_metrics(image, lab > 0)

        # ---------- area robust z (split by isolated vs touching) ----------
        isolated = (degrees == 0)
        area_z = np.zeros_like(areas, dtype=np.float32)
        if np.any(isolated):
            area_z[isolated] = self._robust_z(areas[isolated])
        if np.any(~isolated):
            area_z[~isolated] = self._robust_z(areas[~isolated])

        # ---------- reasons + exclusions ----------
        exclude_ids: List[int] = []
        reasons: Dict[int, List[str]] = {}
        neighbor_ids_per_roi: Dict[int, List[int]] = {rid: sorted(list(neighbors.get(rid, set()))) for rid in range(1, max_id+1)}

        for i, p in enumerate(props):
            rid = p.label
            r: List[str] = []

            # touching violation (propagate to both endpoints of violating edges)
            if rid in violating_ids:
                r.append("touching_violation")  # flagging means excluded

            if self.cfg.min_instance_area > 0 and areas[i] < self.cfg.min_instance_area:
                r.append("too_small")

            if self.cfg.exclude_border_touching and border_touch[i]:
                r.append("border_touch")

            if isolated[i]:
                if abs(area_z[i]) > self.cfg.area_z_max_isolated:
                    r.append("area_outlier")
            else:
                if abs(area_z[i]) > self.cfg.area_z_max_touching:
                    r.append("area_outlier_touching")

            if circularity[i] < self.cfg.min_circularity and degrees[i] >= 1 and solidity[i] < self.cfg.min_solidity:
                r.append("merge_shape")

            if eccentricity[i] > self.cfg.max_eccentricity and degrees[i] == 0:
                r.append("too_elongated")

            if (gap is not None) and (gap[i] < self.cfg.min_boundary_gap):
                r.append("weak_boundary")

            if r:
                reasons[rid] = r

        # exclude anything that was flagged for any reason
        for rid, r in reasons.items():
            if len(r) > 0:
                exclude_ids.append(rid)

        filtered = lab.copy()
        if exclude_ids:
            filtered[np.isin(filtered, np.array(exclude_ids, dtype=np.int32))] = 0

        # ---------- per-ROI table ----------
        roi_table: List[Dict[str, Any]] = []
        for i, p in enumerate(props):
            rid = p.label
            roi_table.append({
                "roi_id": int(rid),
                "area": float(areas[i]),
                "perimeter": float(perims[i]),
                "circularity": float(circularity[i]),
                "eccentricity": float(eccentricity[i]),
                "solidity": float(solidity[i]),
                "degree": int(degrees[i]),
                "neighbor_ids": neighbor_ids_per_roi.get(rid, []),
                "nearest_neighbor_dist": float(nn_dist[i]),
                "border_touch": bool(border_touch[i]),
                "area_z": float(area_z[i]),
                "boundary_gap": (float(gap[i]) if gap is not None else None),
                "excluded": bool(rid in exclude_ids),
                "reasons": reasons.get(rid, []),
            })

        # ---------- well summary ----------
        border_frac = float(np.mean(border_touch)) if max_id > 0 else 0.0
        gap_median = float(np.median(gap)) if gap is not None else np.nan
        excl_frac = float(len(exclude_ids)) / float(max_id)

        well = {
            "n_instances": int(max_id),
            "border_touch_frac": border_frac,
            "boundary_gap_median": gap_median,
            "focus_vlap": (float(vlap) if vlap is not None else np.nan),
            "snr": (float(snr) if snr is not None else np.nan),
            "saturation_frac": (float(sat_frac) if sat_frac is not None else np.nan),
            "excluded_frac": excl_frac,
            "excluded_ids": exclude_ids,
            "touching_pairs_violating": violating_edges,  # edges that caused exclusion
        }

        return {
            "instances_filtered": filtered,
            "exclude_ids": exclude_ids,
            "roi_table": roi_table,
            "well": well,
        }

    # ---------------- helpers ----------------

    def _adjacency(self, lab: np.ndarray, connectivity: int = 8) -> Tuple[Dict[int, set], List[Tuple[int,int]]]:
        """Pairs of labels that touch (8-connected by default)."""
        assert connectivity in (4, 8)
        shifts = [(1,0),(0,1),(-1,0),(0,-1)]
        if connectivity == 8:
            shifts += [(1,1),(1,-1),(-1,1),(-1,-1)]
        pairs = set()
        for dy, dx in shifts:
            B = np.roll(lab, shift=(dy, dx), axis=(0, 1))
            m = (lab != B) & (lab > 0) & (B > 0)
            if np.any(m):
                a = lab[m].astype(np.int32)
                b = B[m].astype(np.int32)
                for x, y in zip(a, b):
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    pairs.add((int(x), int(y)))
        neigh: Dict[int, set] = {}
        for a, b in pairs:
            neigh.setdefault(a, set()).add(b)
            neigh.setdefault(b, set()).add(a)
        return neigh, sorted(pairs)

    def _border_touching(self, lab: np.ndarray, max_id: int) -> np.ndarray:
        H, W = lab.shape
        border_mask = np.zeros_like(lab, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        ids_on_border = np.unique(lab[border_mask])
        touch = np.zeros(max_id, dtype=bool)
        ids_on_border = ids_on_border[ids_on_border > 0]
        touch[ids_on_border - 1] = True
        return touch

    def _nearest_neighbor_distances(self, centroids: np.ndarray) -> np.ndarray:
        if len(centroids) <= 1:
            return np.full((len(centroids),), np.inf, dtype=np.float32)
        tree = cKDTree(centroids[:, ::-1])  # (x, y)
        dists, _ = tree.query(centroids[:, ::-1], k=2)
        return dists[:, 1].astype(np.float32)

    def _boundary_gap(self, lab: np.ndarray, p_cell: np.ndarray, p_bound: np.ndarray) -> np.ndarray:
        max_id = int(lab.max())
        gaps = np.zeros(max_id, dtype=np.float32)
        for rid in range(1, max_id + 1):
            m = (lab == rid)
            if not m.any():
                gaps[rid - 1] = np.nan
                continue
            pc_in = float(p_cell[m].mean())
            ring = morphology.binary_dilation(m, morphology.disk(1)) & ~m
            pb_bd = float(p_bound[ring].mean()) if ring.any() else 0.0
            gaps[rid - 1] = pc_in - pb_bd
        return gaps

    def _intensity_focus_metrics(self, image: np.ndarray, fg_mask: np.ndarray) -> Tuple[float, float, float]:
        img = image

        # ---- 3D images: support both HWC and CHW ----
        if img.ndim == 3:
            # Heuristic: CHW if channel dim is first and last dim is not 1/3
            is_chw = (img.shape[0] in (1, 3)) and (img.shape[-1] not in (1, 3))

            if is_chw:
                # CHW: [C, H, W]
                C, H, W = img.shape

                if img.dtype == np.uint8:
                    if C == 1:
                        gray = img[0].astype(np.float32) / 255.0
                    else:
                        gray = img[:3].astype(np.float32).mean(axis=0) / 255.0
                    sat_hi = (img >= 255).any(axis=0)   # -> [H, W]
                    sat_lo = (img <= 0).any(axis=0)
                else:
                    if C == 1:
                        gray = img[0].astype(np.float32)
                    else:
                        gray = img[:3].astype(np.float32).mean(axis=0)
                    vmax = img.max()
                    vmin = img.min()
                    sat_hi = (img >= vmax).any(axis=0)
                    sat_lo = (img <= vmin).any(axis=0)

            else:
                # HWC: [H, W, C]
                H, W, C = img.shape

                if img.dtype == np.uint8:
                    gray = img.mean(axis=2).astype(np.float32) / 255.0
                    sat_hi = (img >= 255).any(axis=2)  # -> [H, W]
                    sat_lo = (img <= 0).any(axis=2)
                else:
                    gray = img.mean(axis=2).astype(np.float32)
                    vmax = img.max()
                    vmin = img.min()
                    sat_hi = (img >= vmax).any(axis=2)
                    sat_lo = (img <= vmin).any(axis=2)

        # ---- 2D images: grayscale ----
        else:
            gray = img.astype(np.float32)
            if img.dtype == np.uint8:
                sat_hi = (img >= 255)
                sat_lo = (img <= 0)
            else:
                vmax = img.max()
                vmin = img.min()
                sat_hi = (img >= vmax)
                sat_lo = (img <= vmin)

        # ---- saturation fraction ----
        sat_frac = float(np.mean(sat_hi | sat_lo))

        # ---- SNR between fg and bg ----
        bg = ~fg_mask
        if np.any(bg):
            mean_fg = float(gray[fg_mask].mean()) if np.any(fg_mask) else float(gray.mean())
            mean_bg = float(gray[bg].mean())
            std_bg = float(gray[bg].std() + 1e-6)
            snr = (mean_fg - mean_bg) / std_bg
        else:
            snr = 0.0

        # ---- focus measure: Laplacian variance ----
        lap = filters.laplace(gray)
        vlap = float(np.var(lap))

        return sat_frac, snr, vlap

    def _robust_z(self, x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        return 0.6745 * (x - med) / mad

