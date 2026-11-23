"""Export UNet-vs-three-expert-median CDC microscopy disagreements.

Based on the segmentation_annotation review exporter; uses the same UNet,
per-run PC/NC Gaussian calibration, and class-colored TIFF outlines.
Exports are review material, NOT evidence that the UNet is correct.
"""

from __future__ import annotations

import argparse
import copy
import warnings
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


SCORES = frozenset({1, 2, 4, 6, 8})
IDENTITY = ("Folder", "well", "image_name", "role")
SCORE_EXTRA_COLUMNS = (
    "annotator_1", "annotator_2", "annotator_3",
    "median_annotator_score", "unet_score", "human_pattern",
    "unet_matches_any_human", "unet_differs_from_all_humans",
    "unet_minus_median",
)
CASE_COLUMNS = (
    "center", *IDENTITY, "annotator_scores", "n_annotators",
    "n_distinct_scores", "majority_score", "disagreement_type",
)
ISSUE_COLUMNS = ("center", *IDENTITY, "annotator_scores", "reason")
CELL_COLUMNS = (
    "center", *IDENTITY, "roi_id", "predicted_label", "classifier_score",
    "area", "mean_r", "mean_g", "mean_b", "reviewed_label", "review_notes",
)
IMAGE_COLUMNS = (
    *CASE_COLUMNS, *SCORE_EXTRA_COLUMNS, "source_path", "annotated_path", "n_positive",
    "n_negative", "n_uncertain", "n_cells", "height", "width", "dtype",
    "export_dtype", "display_white_level",
)
DEFAULT_COLORS = {
    "pos": (255, 165, 0),       # orange, RGB
    "neg": (0, 220, 0),         # green, RGB
    "uncertain": (0, 130, 255), # blue, RGB
}


@dataclass
class AnnotationConfig:
    """Settings for rendering and production-cell classification."""

    outline_width: int = 2
    colors_rgb: Mapping[str, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COLORS)
    )
    classifier_margin: float = 1.0  # production classifier default
    qc: bool = False              # production run_image_analysis default
    unet_config: dict[str, Any] | None = None
    verify_tiff: bool = True
    overwrite: bool = False
    # review_8bit makes the microscopy signal visible in standard TIFF viewers.
    # native preserves the original uint8/uint16 data outside the outlines.
    render_mode: str = "review_8bit"
    display_percentile: float = 99.7
    display_gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.outline_width < 1:
            raise ValueError("outline_width must be >= 1")
        if self.render_mode not in {"review_8bit", "native"}:
            raise ValueError("render_mode must be 'review_8bit' or 'native'")
        if not np.isfinite(self.display_percentile) or not 0 < self.display_percentile <= 100:
            raise ValueError("display_percentile must be within (0, 100]")
        if not np.isfinite(self.display_gamma) or self.display_gamma <= 0:
            raise ValueError("display_gamma must be a positive finite number")
        if not np.isfinite(self.classifier_margin) or self.classifier_margin < 0:
            raise ValueError("classifier_margin must be a finite nonnegative number")
        if set(self.colors_rgb) != {"pos", "neg", "uncertain"}:
            raise ValueError("colors_rgb must define exactly pos, neg, uncertain")
        for label, rgb in self.colors_rgb.items():
            if len(rgb) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in rgb):
                raise ValueError(f"{label}: expected RGB integers in [0,255]")


def _component(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or any(x in text for x in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Unsafe {name}: {text!r}")
    return text


def _valid_score(raw: str) -> int:
    value = float(raw)
    if not np.isfinite(value) or int(value) != value or int(value) not in SCORES:
        raise ValueError(f"Invalid CDC score: {raw!r}")
    return int(value)


def _review_pra_layout(center: str) -> tuple[dict[str, str], list[str]]:
    """Use the canonical PRA acquisition order with the corrected PC layout.

    The existing PRA_GENERIC_LAYOUT incorrectly labels B10 as a second
    positive control. In the reviewed experiments A10 is the *only* PC; B10
    is treated as a sample. Do not mutate the global production constant.
    """
    from ...image_analysis.utils import PRA_GENERIC_LAYOUT, PRA_GENERIC_IMAGE_ORDER

    roles = dict(PRA_GENERIC_LAYOUT.wells)
    order = list(PRA_GENERIC_IMAGE_ORDER)
    if len(order) != len(set(order)) or set(order) != set(roles):
        raise ValueError("PRA_GENERIC_IMAGE_ORDER is inconsistent with PRA_GENERIC_LAYOUT")
    for well, expected_role in {"A1": "negative", "B1": "negative", "A10": "positive"}.items():
        if roles.get(well) != expected_role:
            raise ValueError(f"Unexpected canonical PRA role for {well}: {roles.get(well)!r}")
    if center == "Berlin":
        roles["B10"] = "sample"

    positives = {well for well, role in roles.items() if role == "positive"}
    expected = {"A10"} if center == "Berlin" else {"A10", "B10"}

    if positives != expected:
        raise ValueError(f"Expected positive controls {expected}, got {positives}")
    return roles, order


def _validate_acquisition_order(
    run_wells: Mapping[str, Mapping[str, str]], order: Sequence[str]
) -> None:
    """Verify the existing per-image well mapping for complete 60-image runs.

    This checks, but does not rewrite, the snake-shaped acquisition order.
    For incomplete runs it retains the explicit CSV well IDs.
    """
    def natural_key(name: str):
        return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                     for part in re.split(r"(\d+)", name))

    for folder, images in run_wells.items():
        if len(images) != len(order):
            continue
        observed = [images[name] for name in sorted(images, key=natural_key)]
        if observed != list(order):
            first = next(i for i, (a, b) in enumerate(zip(observed, order), 1) if a != b)
            raise ValueError(
                f"{folder}: image-to-well mapping disagrees with PRA_GENERIC_IMAGE_ORDER "
                f"at acquisition position {first}: CSV={observed[first-1]}, "
                f"layout={order[first-1]}"
            )


def select_unet_median_disagreements(
    csv_path: str | Path,
    *,
    exclude_frankfurt_duplicate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, dict[str, set[str]]]]]:
    """Select all three-expert-vs-UNet median disagreements from one long CSV.

    The identity is (site, Folder, well, image_name) only: nullable PRA and
    Specificity metadata must NEVER be grouping keys. Exclude the previously
    identified Frankfurt repeated run by default to match Figure 4. Bad or
    incomplete scores are logged as issues and never treated as agreements.

    Returns (cases, issues, full_run_controls). The source CSV is read only.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    key_cols = ["site", "Folder", "well", "image_name"]
    required = set(key_cols + ["Annotator", "score", "role"])
    raw = pd.read_csv(path, dtype={col: str for col in key_cols + ["Annotator", "role"]})
    if not required.issubset(raw.columns):
        raise ValueError(f"{path}: missing columns: {sorted(required - set(raw.columns))}")
    if raw.empty:
        raise ValueError(f"{path}: no scoring rows")
    for col in key_cols + ["Annotator"]:
        if raw[col].isna().any() or raw[col].str.strip().eq("").any():
            raise ValueError(f"{path}: empty image identifier or annotator in {col}")
    for col in ("site", "Folder", "well", "image_name"):
        raw[col] = raw[col].map(lambda v: _component(v, col))
    raw["Annotator"] = raw["Annotator"].str.strip().str.lower()
    raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
    if not raw["site"].isin(("Frankfurt", "Berlin")).all():
        raise ValueError(f"Unexpected sites: {sorted(raw['site'].unique())}")
    if exclude_frankfurt_duplicate:
        old_count = len(raw)
        raw = raw.loc[~(
            raw["site"].eq("Frankfurt") &
            raw["Folder"].eq("RUN_063F11D2D1DD")
        )].copy()
        print(f"Excluded duplicated Frankfurt RUN_063F11D2D1DD: {old_count - len(raw)} scoring rows")

    roles_by_site = {site: _review_pra_layout(site)[0] for site in ("Frankfurt", "Berlin")}
    # All controls from the complete readout sheet are required for the
    # per-run Gaussian classifier, not just the selected discordant images.
    controls: dict[str, dict[str, dict[str, set[str]]]] = {
        site: defaultdict(lambda: {"positive": set(), "negative": set()})
        for site in ("Frankfurt", "Berlin")
    }
    for (site, folder, well, image_name), group in raw.groupby(key_cols, sort=True):
        role = roles_by_site[site].get(well)
        if role is None:
            raise ValueError(f"Invalid well {site}/{folder}/{well}")
        if role in {"positive", "negative"}:
            controls[site][folder][role].add(image_name)

    cases, issues = [], []
    needed_annotators = {"1", "2", "3", "unet"}
    for (site, folder, well, image_name), group in raw.groupby(key_cols, sort=True):
        role = roles_by_site[site][well]
        base = {"center": site, "Folder": folder, "well": well,
                "image_name": image_name, "role": role}
        if not set(group["Annotator"]).issubset(needed_annotators):
            # Ignore other experimental scoring records if they exist.
            group = group[group["Annotator"].isin(needed_annotators)]
        observed: dict[str, int] = {}
        problem = None
        for annotator, records in group.groupby("Annotator", sort=False):
            values = records["score"]
            if values.isna().any() or not values.isin(SCORES).all():
                problem = "invalid_score"
                break
            unique_values = values.unique()
            if len(unique_values) != 1:
                problem = "conflicting_duplicate_scores"
                break
            observed[annotator] = int(unique_values[0])
        if problem is None and set(observed) != needed_annotators:
            problem = "missing_annotator_or_unet"
        if problem is not None:
            issues.append({**base, "reason": problem,
                           "annotator_scores": ";".join(
                               f"{k}:{v}" for k, v in sorted(observed.items()))})
            continue
        a, b, c = (observed[str(i)] for i in (1, 2, 3))
        median = sorted((a, b, c))[1]
        predicted = observed["unet"]
        if predicted == median:
            continue
        counts = Counter((a, b, c))
        pattern = "3:0" if len(counts) == 1 else "2:1" if len(counts) == 2 else "1:1:1"
        majority = next((k for k, n in counts.items() if n >= 2), None)
        cases.append({
            **base,
            "annotator_scores": f"1:{a};2:{b};3:{c};unet:{predicted}",
            "n_annotators": 3,
            "n_distinct_scores": len(counts),
            "majority_score": majority,
            "disagreement_type": "three_way" if len(counts) == 3 else
                                 "two_vs_one" if len(counts) == 2 else "unanimous_humans",
            "annotator_1": a, "annotator_2": b, "annotator_3": c,
            "median_annotator_score": median, "unet_score": predicted,
            "human_pattern": pattern,
            "unet_matches_any_human": predicted in (a, b, c),
            "unet_differs_from_all_humans": predicted not in (a, b, c),
            "unet_minus_median": predicted - median,
        })

    cases_df = pd.DataFrame(cases, columns=[*CASE_COLUMNS, *SCORE_EXTRA_COLUMNS])
    issue_df = pd.DataFrame(issues, columns=["center", *IDENTITY, "reason", "annotator_scores"])
    controls = {site: dict(per_folder) for site, per_folder in controls.items()}
    return cases_df, issue_df, controls


def prepare_review_image(
    image_rgb: np.ndarray,
    *,
    percentile: float = 99.7,
    gamma: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Make an RGB uint8 view of a dark uint8/uint16 microscopy image.

    Uses a SINGLE scale factor for all RGB channels, rather than separate
    channel normalization, to avoid changing the relative red/green signals.
    A global brightness level is estimated only from nonblack pixels so that
    large dark fields do not force the upper percentile to zero.

    This is for human display only. It does not alter raw source data,
    segmentation, extracted ROI values, or classifier predictions.
    ``gamma < 1`` makes dim cells easier to see.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("Expected uint8/uint16 image")
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must lie in (0, 100]")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be positive and finite")

    brightest = image_rgb.max(axis=2)
    nonzero = brightest[brightest > 0]
    white = float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0
    white = max(white, 1.0)
    normalized = np.clip(image_rgb.astype(np.float32) / white, 0.0, 1.0)
    if gamma != 1.0:
        np.power(normalized, gamma, out=normalized)
    # np.rint before cast avoids integer conversion bias.
    display = np.rint(normalized * 255.0).astype(np.uint8)
    return display, white


def draw_cell_outlines(
    image_rgb: np.ndarray,
    instance_labels: np.ndarray,
    rois: Sequence[Any],
    *,
    outline_width: int = 2,
    colors_rgb: Mapping[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw per-instance, class-colored borders over the *original* RGB pixels.

    Returns an RGB array with the same shape and dtype (uint8/uint16) as input.
    Non-border pixels are unchanged; the array passed in is not mutated.
    ``roi_id`` must match the actual instance label, not a list position.
    """
    if outline_width < 1:
        raise ValueError("outline_width must be >= 1")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {image_rgb.shape}")
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise TypeError("Input image must be uint8 or uint16 for lossless TIFF output")
    labels = np.asarray(instance_labels)
    if labels.ndim != 2 or labels.shape != image_rgb.shape[:2]:
        raise ValueError("Instance map must be HxW and have the same size as the image")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise TypeError("Instance labels must be nonnegative integers")
    palette = colors_rgb if colors_rgb is not None else DEFAULT_COLORS
    max_val = int(np.iinfo(image_rgb.dtype).max)
    output = image_rgb.copy()
    objects = ndi.find_objects(labels.astype(np.int32, copy=False))
    height, width = labels.shape
    pad = max(2, outline_width + 1)

    for roi in rois:
        rid = int(roi.get("roi_id") if isinstance(roi, dict) else roi.roi_id)
        cls = roi.get("label") if isinstance(roi, dict) else roi.label
        if cls not in palette:
            raise ValueError(f"Unknown ROI label {cls!r} for ROI {rid}")
        if not 1 <= rid <= len(objects) or objects[rid - 1] is None:
            raise ValueError(f"ROI ID {rid} has no pixels in instance label map")
        ys, xs = objects[rid - 1]
        y0, y1 = max(0, ys.start - pad), min(height, ys.stop + pad)
        x0, x1 = max(0, xs.start - pad), min(width, xs.stop + pad)
        roi_mask = (labels[y0:y1, x0:x1] == rid).astype(np.uint8)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(round(channel * max_val / 255.0)) for channel in palette[cls])
        # cv2 draws literal RGB component values into our RGB array. Conversion
        # to OpenCV BGR happens only at the TIFF-writing boundary.
        cv2.drawContours(output[y0:y1, x0:x1], contours, -1, color,
                         thickness=outline_width, lineType=cv2.LINE_8)
    return output


def save_annotated_tiff(path: str | Path, image_rgb: np.ndarray, *, verify: bool = True):
    """Write uint8/uint16 RGB TIFF without rescaling or lossy compression."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".partial.tif")
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    params = ([cv2.IMWRITE_TIFF_COMPRESSION, 1]
              if hasattr(cv2, "IMWRITE_TIFF_COMPRESSION") else [])
    try:
        if not cv2.imwrite(str(tmp), bgr, params):
            raise IOError(f"Cannot write TIFF: {tmp}")
        if verify:
            read_back = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)
            if read_back is None or read_back.dtype != bgr.dtype or not np.array_equal(read_back, bgr):
                raise IOError(f"TIFF pixel/depth verification failed: {tmp}")
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def _scaled_and_raw(path: Path):
    """Apply the exact scale=True image preprocessing used by the app pipeline."""
    from ...image_analysis.io.io import load_image, scale_image

    raw, report = load_image(path, scale=False, as_chw=False)
    if raw.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Expected uint8/uint16 RGB TIFF: {path}: {raw.dtype}")
    scaled, _ = scale_image(raw, report, fast=True)
    return raw, scaled


def _segment_extract(path: Path, segmenter: Any, qc: bool):
    """Reuse pipeline segmentation + optional QC + RGB ROI extraction."""
    from ...image_analysis.pipeline import _extract_roi_from_image
    from ...image_analysis.extractor import RGBExtractor
    from ...image_analysis.qc import QCMonitor

    raw, scaled = _scaled_and_raw(path)
    result, seg = _extract_roi_from_image(
        image=scaled,
        segmenter=segmenter,
        qc_monitor=QCMonitor() if qc else None,
        extractor=RGBExtractor(),
        well_id=path.stem,
        qc=qc,
    )
    labels = seg["instance_labels_for_rois"]
    if labels.shape != raw.shape[:2]:
        raise ValueError(f"Segmentation shape mismatch for {path}")
    return raw, labels, [roi.__dict__ for roi in result.rois]


def _path(roots: Mapping[str, str | Path], center: str, folder: str, name: str) -> Path:
    root = Path(roots[center]).expanduser().resolve()
    candidate = (root / _component(folder, "Folder") / _component(name, "image_name")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes source root: {candidate}")
    return candidate


def _calibrate_run(
    center: str, folder: str, control_map: Mapping[str, Any],
    roots: Mapping[str, str | Path], segmenter: Any, qc: bool,
) -> dict[str, Any]:
    """Fit the production 2D Gaussian using A10 PC and A1/B1 NC wells."""
    from ...image_analysis.calibrators import PCNCGaussian2DCalibrator

    pc_wells, nc_wells = [], []
    for role, destination in (("positive", pc_wells), ("negative", nc_wells)):
        names = sorted(control_map.get(role, []))
        if not names:
            raise ValueError(f"No {role} controls for {center}/{folder}")
        for name in names:
            path = _path(roots, center, folder, name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {role} control: {path}")
            _, _, rois = _segment_extract(path, segmenter, qc)
            if not rois:
                raise RuntimeError(f"No extracted cells in {role} control: {path}")
            destination.append(rois)
    return PCNCGaussian2DCalibrator().fit(pc_wells=pc_wells, nc_wells=nc_wells)


def _make_segmenter(unet_config: dict[str, Any] | None):
    """Load the same checkpoint/config as the production image-analysis pipeline."""
    import copy
    from ...image_analysis.config import INSTANCE_CONFIG, UNET_CONFIG
    from ...image_analysis.segmenter import SegmenterUNetInference

    config = copy.deepcopy(unet_config if unet_config is not None else UNET_CONFIG)
    config.setdefault("instance_cfg", INSTANCE_CONFIG.to_dict())
    return SegmenterUNetInference.from_config(config)


def _write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def annotate_disagreements(
    cases: pd.DataFrame,
    controls: Mapping[str, Mapping[str, Mapping[str, set[str]]]],
    roots: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    config: AnnotationConfig | None = None,
    limit_per_center: int | None = None,
    segmenter: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify disagreeing images, save outlined TIFFs and review CSVs.

    Parameters
    ----------
    cases
        Output of ``find_disagreements``. Includes all three-way splits.
    controls
        Original complete PC and NC well list from ``find_disagreements``.
    roots
        ``{'Berlin': Path(...), 'Frankfurt': Path(...)}``.
    output_dir
        Export root. Creates ``Berlin/`` and ``Frankfurt/`` folders.
    config
        Outline color/width, classifier margin, QC and model settings.
    limit_per_center
        Use 1-3 for first notebook test; None processes every disagreement.
    segmenter
        Optional preloaded segmenter (also useful for unit tests).

    Returns
    -------
    images, cells : pandas.DataFrame
        ``images`` describes every exported TIFF; ``cells`` is a row per
        classified ROI with blank ``reviewed_label``/``review_notes`` columns.
    """
    from ...image_analysis.classifiers import ROIClassifierGaussian2D3Way

    cfg = config or AnnotationConfig()
    if limit_per_center is not None and limit_per_center < 1:
        raise ValueError("limit_per_center must be >= 1 or None")
    out = Path(output_dir).expanduser().resolve()
    missing_cols = set(CASE_COLUMNS) - set(cases.columns)
    if missing_cols:
        raise ValueError(f"cases is missing columns: {sorted(missing_cols)}")
    selection = cases.sort_values(["center", "Folder", "well", "image_name"], kind="stable")
    if limit_per_center is not None:
        selection = selection.groupby("center", sort=False).head(limit_per_center)
    if selection.empty:
        raise ValueError("No discordant images selected")

    # Check all referenced inputs and required calibration controls BEFORE
    # initializing the model or creating partial outputs.
    destinations = []
    for row in selection.itertuples(index=False):
        path = _path(roots, row.center, row.Folder, row.image_name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected TIFF: {path}")
        ctrl = controls[row.center][row.Folder]
        for role in ("positive", "negative"):
            names = ctrl.get(role, set())
            if not names:
                raise ValueError(f"Missing {role} control metadata for {row.center}/{row.Folder}")
            for name in names:
                control_path = _path(roots, row.center, row.Folder, name)
                if not control_path.is_file():
                    raise FileNotFoundError(f"Missing calibration TIFF: {control_path}")
        name = f"{row.Folder}__{row.well}__{Path(row.image_name).stem}__annotated.tif"
        dst = out / row.center / name
        if dst.exists() and not cfg.overwrite:
            raise FileExistsError(f"Output exists (set overwrite=True or use a fresh folder): {dst}")
        destinations.append((row, path, dst))

    engine = segmenter if segmenter is not None else _make_segmenter(cfg.unet_config)
    calibrations: dict[tuple[str, str], dict[str, Any]] = {}
    image_rows, cell_rows = [], []
    for count, (row, source, destination) in enumerate(destinations, 1):
        run_key = (row.center, row.Folder)
        if run_key not in calibrations:
            print(f"Calibrating {row.center}/{row.Folder} with all PC/NC wells...", flush=True)
            calibrations[run_key] = _calibrate_run(
                row.center, row.Folder, controls[row.center][row.Folder],
                roots, engine, cfg.qc,
            )
        classifier = ROIClassifierGaussian2D3Way(
            calibrations[run_key], margin=cfg.classifier_margin
        )
        rgb_raw, labels, rois = _segment_extract(source, engine, cfg.qc)
        if not rois:
            raise RuntimeError(f"No cells extracted from selected image: {source}")
        classified = classifier(rois)
        if cfg.render_mode == "review_8bit":
            image_base, display_white = prepare_review_image(
                rgb_raw,
                percentile=cfg.display_percentile,
                gamma=cfg.display_gamma,
            )
        else:
            image_base, display_white = rgb_raw, None
        rendered = draw_cell_outlines(
            image_base, labels, classified,
            outline_width=cfg.outline_width,
            colors_rgb=cfg.colors_rgb,
        )
        save_annotated_tiff(destination, rendered, verify=cfg.verify_tiff)
        label_counts = Counter(r["label"] for r in classified)
        base = {k: getattr(row, k) for k in (*CASE_COLUMNS, *SCORE_EXTRA_COLUMNS)}
        image_rows.append({
            **base, "source_path": str(source), "annotated_path": str(destination),
            "n_positive": label_counts["pos"], "n_negative": label_counts["neg"],
            "n_uncertain": label_counts["uncertain"], "n_cells": len(classified),
            "height": rgb_raw.shape[0], "width": rgb_raw.shape[1],
            "dtype": str(rgb_raw.dtype), "export_dtype": str(rendered.dtype),
            "display_white_level": display_white,
        })
        for roi in classified:
            cell_rows.append({
                **{k: base[k] for k in ("center", *IDENTITY)},
                "roi_id": roi["roi_id"], "predicted_label": roi["label"],
                "classifier_score": roi.get("score"), "area": roi["area"],
                "mean_r": roi["mean_r"], "mean_g": roi["mean_g"],
                "mean_b": roi["mean_b"],
                "reviewed_label": "", "review_notes": "",
            })
        print(f"[{count}/{len(destinations)}] {row.center}/{row.Folder}/{row.image_name}: "
              f"{len(classified)} cells -> {destination.name}", flush=True)

    images = pd.DataFrame(image_rows, columns=IMAGE_COLUMNS)
    cells = pd.DataFrame(cell_rows, columns=CELL_COLUMNS)
    _write_table(out / "disagreement_image_manifest.csv", images)
    _write_table(out / "cell_review_template.csv", cells)
    return images, cells


def export_review_overlays_from_existing(
    image_manifest: str | Path,
    output_dir: str | Path,
    *,
    display_percentile: float = 99.7,
    display_gamma: float = 1.0,
    outline_colors_rgb: Mapping[str, tuple[int, int, int]] | None = None,
    verify_tiff: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Turn existing native outlined 16-bit TIFFs into viewable RGB TIFFs.

    This avoids re-running segmentation, control calibration, or classification.
    Reconstructs the outlines by finding the exact RGB drawing colors in the
    previous native TIFF. The source TIFF is used for the display base; it is
    never overwritten. Both image paths are obtained from the original manifest.

    Works with outputs made by ``draw_cell_outlines`` using LINE_8.
    If colors were customized in the original run, pass the same original
    palette through ``outline_colors_rgb``.
    """
    manifest = Path(image_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = pd.read_csv(manifest, dtype={"Folder": str, "well": str})
    required = {"center", "source_path", "annotated_path"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(rows.columns))}")
    palette = outline_colors_rgb if outline_colors_rgb is not None else DEFAULT_COLORS
    if set(palette) != {"pos", "neg", "uncertain"}:
        raise ValueError("outline_colors_rgb must contain pos, neg, uncertain")

    out = Path(output_dir).expanduser().resolve()
    targets = []
    for row in rows.itertuples(index=False):
        center = _component(row.center, "center")
        src, old = Path(row.source_path), Path(row.annotated_path)
        if not src.is_file() or not old.is_file():
            raise FileNotFoundError(f"Missing raw/annotated file: {src} / {old}")
        dst = out / center / old.name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Already exists: {dst}")
        targets.append((src, old, dst, center))

    results = []
    from ...image_analysis.io.io import load_image
    for i, (src, old, dst, center) in enumerate(targets, 1):
        raw, _ = load_image(src, scale=False, as_chw=False)
        old_bgr = cv2.imread(str(old), cv2.IMREAD_UNCHANGED)
        if old_bgr is None or old_bgr.ndim != 3 or old_bgr.shape[2] != 3:
            raise ValueError(f"Unable to decode native outlined RGB TIFF: {old}")
        annotated_rgb = cv2.cvtColor(old_bgr, cv2.COLOR_BGR2RGB)
        if raw.shape != annotated_rgb.shape or raw.dtype != annotated_rgb.dtype:
            raise ValueError(f"Old overlay does not match source TIFF: {old}")
        if raw.dtype not in (np.uint8, np.uint16):
            raise TypeError(f"Unsupported source type {raw.dtype} for {src}")

        display, white = prepare_review_image(
            raw, percentile=display_percentile, gamma=display_gamma
        )
        max_val = int(np.iinfo(raw.dtype).max)
        changed = np.any(annotated_rgb != raw, axis=2)
        counts = {}
        for label, color in palette.items():
            old_color = tuple(int(round(channel * max_val / 255.0)) for channel in color)
            mask = np.all(annotated_rgb == old_color, axis=2) & changed
            display[mask] = color
            counts[label] = int(mask.sum())
        if not any(counts.values()):
            raise RuntimeError(
                f"No colored outline pixels detected in {old}. "
                "Is it an original native TIFF with the specified color palette?"
            )
        save_annotated_tiff(dst, display, verify=verify_tiff)
        results.append({
            "center": center, "source_path": str(src),
            "old_annotated_path": str(old), "review_path": str(dst),
            "white_level": white, "positive_outline_pixels": counts["pos"],
            "negative_outline_pixels": counts["neg"],
            "uncertain_outline_pixels": counts["uncertain"],
        })
        print(f"[{i}/{len(targets)}] {center}: {dst.name}", flush=True)
    table = pd.DataFrame(results)
    _write_table(out / "review_overlay_manifest.csv", table)
    return table



def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line export of all UNet/median-human score disagreements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="./results/readout_scores_with_unet.csv",
                        help="Combined long-form human/UNet scores CSV")
    parser.add_argument("--frankfurt-images", default="./experiment_readout_images")
    parser.add_argument("--berlin-images", default="./berlin_validation")
    parser.add_argument("--output-dir", default="./unet_median_disagreement_review")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--unet-mode", default="small")
    parser.add_argument("--model-file", default=None,
                        help="Defaults to best_{unet_mode}_tiles_S512_seed187.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--limit-per-center", type=int, default=None,
                        help="Optional fast test, e.g. 2 images per center")
    parser.add_argument("--render-mode", choices=("review_8bit", "native"),
                        default="review_8bit")
    parser.add_argument("--include-duplicate-run", action="store_true",
                        help="Allow Frankfurt RUN_063F11D2D1DD (excluded in Figure 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List selected images and scores without loading UNet")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing image exports")
    args = parser.parse_args(argv)
    if args.limit_per_center is not None and args.limit_per_center < 1:
        parser.error("--limit-per-center must be >= 1")

    cases, issues, controls = select_unet_median_disagreements(
        args.scores,
        exclude_frankfurt_duplicate=not args.include_duplicate_run,
    )
    if cases.empty:
        print("WARNING: No complete valid UNet/median-human disagreements.")
        return 1
    print("Selected images by center:")
    print(cases.groupby("center").size().to_string())
    print(f"Total: {len(cases)} images; skipped incomplete/invalid: {len(issues)}")
    print("Of selected images, UNet differs from all three humans:",
          int(cases["unet_differs_from_all_humans"].sum()))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.is_file():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # This is the requested reviewer-ready score list; in dry-run it records
    # the *full* intended selection (before any per-site test limit).
    cases.to_csv(output_dir / "selected_unet_median_disagreements.csv", index=False)
    issues.to_csv(output_dir / "skipped_invalid_or_incomplete.csv", index=False)
    if args.dry_run:
        print("Dry run complete. No TIFFs were created.")
        return 0
    if args.limit_per_center is not None:
        cases = (cases.sort_values(["center", "Folder", "well", "image_name"])
                 .groupby("center", sort=False).head(args.limit_per_center).copy())

    from ...image_analysis.config import UNET_CONFIG
    cfg_dict = copy.deepcopy(UNET_CONFIG)
    cfg_dict.update({
        "unet_mode": args.unet_mode,
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "model_file": args.model_file or f"best_{args.unet_mode}_tiles_S512_seed187.pth",
    })
    if args.device:
        cfg_dict["device"] = args.device
        cfg_dict["use_amp"] = args.device == "cuda"
    model_path = Path(cfg_dict["model_dir"]) / cfg_dict["model_file"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing UNet checkpoint: {model_path}")
    config = AnnotationConfig(
        unet_config=cfg_dict,
        render_mode=args.render_mode,
        overwrite=args.overwrite,
    )
    roots = {"Frankfurt": args.frankfurt_images, "Berlin": args.berlin_images}
    image_manifest, _ = annotate_disagreements(
        cases=cases,
        controls=controls,
        roots=roots,
        output_dir=output_dir,
        config=config,
    )
    # The full image manifest contains the individual scores and paths.
    # Include blank review columns to document independent adjudication later.
    manifest = image_manifest.copy()
    manifest["reviewer_verdict"] = ""
    manifest["reviewer_notes"] = ""
    manifest.to_csv(output_dir / "unet_human_disagreement_scores.csv", index=False)
    print(f"Exported {len(manifest)} TIFFs and score manifest into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

