from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from alloviewer.image_analysis.pipeline import run_image_analysis
from alloviewer.image_analysis.calibrators import (
    PCNCMedianCalibrator,
    PCNCGaussianRGCalibrator,
    PCNCGaussian2DCalibrator
)
from alloviewer.image_analysis.classifiers import (
    ROIClassifier,
    ROIClassifierGaussian3Way,
    ROIClassifierGaussian2D3Way
)
from ...image_analysis.utils import (
    PRA_GENERIC_LAYOUT,
    PRA_GENERIC_IMAGE_ORDER,
    convert_frac_pos_to_score,
)


def _layout_dict(layout: Any) -> Dict[str, str]:
    if hasattr(layout, "wells"):
        return dict(layout.wells)
    return dict(layout)


def natural_image_sort_key(name: str):
    """
    Sort filenames by embedded numbers.
    Example:
        Bild_2.tif < Bild_10.tif
    """
    parts = re.split(r"(\d+)", str(name))
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def list_image_files(
    folder: Path,
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> list[Path]:
    allowed_suffixes = {s.lower() for s in allowed_suffixes}
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_suffixes
    ]
    return sorted(files, key=lambda p: natural_image_sort_key(p.name))


def frac_pos_raw_from_labels(labels: Iterable[str]) -> float:
    """
    Proper fraction positive in percent (0-100).

    Uses:
        100 * n_pos / (n_pos + n_neg)

    Notes
    -----
    - 'uncertain' labels are ignored
    - returns np.nan if there are no pos/neg labels
    """
    labels = list(labels)
    n_pos = sum(1 for x in labels if x == "pos")
    n_neg = sum(1 for x in labels if x == "neg")
    denom = n_pos + n_neg

    if denom == 0:
        return np.nan

    return 100.0 * (n_pos / denom)


# ---------------------------------------------------------------------
# image copying + well mapping
# ---------------------------------------------------------------------

def prepare_valid_folders_and_copy_images(
    folders: Iterable[str],
    *,
    src_root: str | Path = "./ext_images",
    dst_root: str | Path = "./experiment_readout_images",
    image_order: Sequence[str] = PRA_GENERIC_IMAGE_ORDER,
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    verbose: bool = True,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Keep only folders with exactly len(image_order) images.

    For each valid folder:
    - sort images in acquisition order
    - copy them to dst_root / folder if needed
    - return a mapping: folder -> well -> image_name
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    folder_to_image_map: dict[str, dict[str, str]] = {}
    valid_folders: list[str] = []

    for folder_name in pd.unique(pd.Series(list(folders)).dropna()):
        folder_name = str(folder_name)
        src_folder = src_root / folder_name

        if not src_folder.exists() or not src_folder.is_dir():
            if verbose:
                print(f"Skipping {folder_name}: source folder not found")
            continue

        src_images = list_image_files(src_folder, allowed_suffixes=allowed_suffixes)

        if len(src_images) != len(image_order):
            if verbose:
                print(
                    f"Skipping {folder_name}: found {len(src_images)} images "
                    f"instead of {len(image_order)}"
                )
            continue

        dst_folder = Path(dst_root) / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        dst_images = list_image_files(dst_folder, allowed_suffixes=allowed_suffixes)
        src_names = [p.name for p in src_images]
        dst_names = [p.name for p in dst_images]

        # keep existing copy if it already matches
        if dst_names != src_names:
            for p in dst_images:
                p.unlink()
            for img in src_images:
                shutil.copy2(img, dst_folder / img.name)

        folder_to_image_map[folder_name] = {
            well: img.name
            for well, img in zip(image_order, src_images)
        }
        valid_folders.append(folder_name)

    return folder_to_image_map, valid_folders


def reshape_scores_with_images(
    csv_path_or_df,
    *,
    sep: str = ";",
    src_root: str | Path = "./ext_images",
    dst_root: str | Path = "./experiment_readout_images",
    layout: Any = PRA_GENERIC_LAYOUT,
    image_order: Sequence[str] = PRA_GENERIC_IMAGE_ORDER,
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    drop_missing_scores: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convert a wide manual score sheet to long format and add:
        Folder | well | image_name | role | Annotator | score

    Also copies valid image folders to dst_root.
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df, sep=sep)

    if "Folder" not in df.columns:
        raise ValueError("Input must contain a 'Folder' column.")

    layout_map = _layout_dict(layout)

    well_pattern = re.compile(r"^[A-Z]\d+$")
    well_cols = [c for c in df.columns if well_pattern.match(str(c))]
    id_cols = [c for c in df.columns if c not in well_cols]

    folder_to_image_map, valid_folders = prepare_valid_folders_and_copy_images(
        folders=df["Folder"].dropna().unique(),
        src_root=src_root,
        dst_root=dst_root,
        image_order=image_order,
        allowed_suffixes=allowed_suffixes,
        verbose=verbose,
    )

    df = df[df["Folder"].astype(str).isin(valid_folders)].copy()

    out = df.melt(
        id_vars=id_cols,
        value_vars=well_cols,
        var_name="well",
        value_name="score",
    )

    out["image_name"] = out.apply(
        lambda row: folder_to_image_map[str(row["Folder"])][row["well"]],
        axis=1,
    )
    out["role"] = out["well"].map(layout_map)

    if drop_missing_scores:
        out = out.dropna(subset=["score"]).copy()

    if "adjusted_score" not in out.columns:
        out["adjusted_score"] = np.nan

    preferred = ["Folder", "well", "image_name", "role", "Annotator", "score", "adjusted_score"]
    existing_preferred = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    out = out.astype(object).replace({pd.NA: np.nan})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# imageJ scoring
# ---------------------------------------------------------------------

def score_imagej_rois(
    csv_path_or_df,
    *,
    sep: str = ",",
    folder_col: str = "Folder",
    image_col: str = "file_name",
    red_col: str = "mean_red",
    green_col: str = "mean_green",
    blue_col: Optional[str] = "mean_blue",
    x_col: Optional[str] = "X",
    y_col: Optional[str] = "Y",
    layout: Any = PRA_GENERIC_LAYOUT,
    image_order: Sequence[str] = PRA_GENERIC_IMAGE_ORDER,
    calibrator_cls: Optional[Type] = None,
    classifier_cls: Optional[Type] = None,
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    image_sort_key: Callable[[str], Any] = natural_image_sort_key,
    require_exactly_n_images: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score ImageJ ROI CSVs folder by folder.

    Output columns:
        folder, image_name, well_id, role,
        frac_pos, corrected_frac_pos,
        final_score, final_score_corrected,
        n_rois, pc_ref_raw, nc_ref_raw,
        calib_method, classifier_method
    """
    if calibrator_cls is None:
        raise ValueError("Please pass calibrator_cls, e.g. PCNCMedianCalibrator")
    if classifier_cls is None:
        raise ValueError("Please pass classifier_cls, e.g. ROIClassifierMedianRG")

    classifier_kwargs = classifier_kwargs or {}
    layout_map = _layout_dict(layout)

    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df, sep=sep)

    required_cols = [folder_col, image_col, red_col, green_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_rows: List[Dict[str, Any]] = []

    for folder_name, df_folder in df.groupby(folder_col, sort=False):
        image_names = sorted(df_folder[image_col].dropna().unique().tolist(), key=image_sort_key)

        if require_exactly_n_images and len(image_names) != len(image_order):
            if verbose:
                print(
                    f"Skipping folder {folder_name}: "
                    f"found {len(image_names)} images, expected {len(image_order)}"
                )
            continue

        if len(image_names) < len(image_order):
            if verbose:
                print(
                    f"Skipping folder {folder_name}: "
                    f"not enough images ({len(image_names)}) for full plate"
                )
            continue

        if len(image_names) > len(image_order):
            if verbose:
                print(
                    f"Folder {folder_name}: found {len(image_names)} images. "
                    f"Using first {len(image_order)} after sorting."
                )
            image_names = image_names[:len(image_order)]

        image_to_well = dict(zip(image_names, image_order))
        image_to_role = {img: layout_map[well] for img, well in image_to_well.items()}

        rois_per_image: Dict[str, List[Dict[str, Any]]] = {}
        for img_name, df_img in df_folder.groupby(image_col, sort=False):
            if img_name not in image_to_well:
                continue

            rois: List[Dict[str, Any]] = []
            for _, row in df_img.iterrows():
                roi = {
                    "mean_r": float(row[red_col]),
                    "mean_g": float(row[green_col]),
                }
                if blue_col is not None and blue_col in df_img.columns:
                    roi["mean_b"] = float(row[blue_col])
                if x_col is not None and x_col in df_img.columns:
                    roi["x"] = float(row[x_col])
                if y_col is not None and y_col in df_img.columns:
                    roi["y"] = float(row[y_col])
                rois.append(roi)

            rois_per_image[img_name] = rois

        pc_wells = [
            rois_per_image[img]
            for img in image_names
            if image_to_role[img] == "positive" and img in rois_per_image
        ]
        nc_wells = [
            rois_per_image[img]
            for img in image_names
            if image_to_role[img] == "negative" and img in rois_per_image
        ]

        if len(pc_wells) == 0 or len(nc_wells) == 0:
            if verbose:
                print(
                    f"Skipping folder {folder_name}: missing positive or negative control wells"
                )
            continue

        calibrator = calibrator_cls()
        calib = calibrator.fit(pc_wells=pc_wells, nc_wells=nc_wells)
        classifier = classifier_cls(calib=calib, **classifier_kwargs)

        per_image_rows: List[Dict[str, Any]] = []
        pc_raw_vals: List[float] = []
        nc_raw_vals: List[float] = []

        for img_name in image_names:
            rois = rois_per_image.get(img_name, [])
            well_id = image_to_well[img_name]
            role = image_to_role[img_name]

            classified_rois = classifier(rois)
            labels = [r["label"] for r in classified_rois]
            raw = frac_pos_raw_from_labels(labels)

            row_out = {
                "folder": folder_name,
                "image_name": img_name,
                "well_id": well_id,
                "role": role,
                "frac_pos": raw,
                "n_rois": len(rois),
                "calib_method": calib.get("method"),
                "classifier_method": getattr(classifier, "method", None),
            }
            per_image_rows.append(row_out)

            if role == "positive" and not pd.isna(raw):
                pc_raw_vals.append(raw)
            elif role == "negative" and not pd.isna(raw):
                nc_raw_vals.append(raw)

        pc_ref = float(np.mean(pc_raw_vals)) if len(pc_raw_vals) > 0 else np.nan
        nc_ref = float(np.mean(nc_raw_vals)) if len(nc_raw_vals) > 0 else np.nan

        for row_out in per_image_rows:
            raw = row_out["frac_pos"]

            if (
                pd.isna(raw)
                or pd.isna(pc_ref)
                or pd.isna(nc_ref)
                or pc_ref == nc_ref
            ):
                corr = np.nan
            else:
                corr = (raw - nc_ref) / (pc_ref - nc_ref) * 100.0
                corr = float(np.clip(corr, 0.0, 100.0))

            row_out["corrected_frac_pos"] = corr
            row_out["final_score"] = convert_frac_pos_to_score(raw)
            row_out["final_score_corrected"] = convert_frac_pos_to_score(corr)
            row_out["pc_ref_raw"] = pc_ref
            row_out["nc_ref_raw"] = nc_ref

        out_rows.extend(per_image_rows)

    out = pd.DataFrame(out_rows)

    wanted = [
        "folder",
        "image_name",
        "well_id",
        "role",
        "frac_pos",
        "corrected_frac_pos",
        "final_score",
        "final_score_corrected",
        "n_rois",
        "pc_ref_raw",
        "nc_ref_raw",
        "calib_method",
        "classifier_method",
    ]
    existing = [c for c in wanted if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining]

    out = out.astype(object).replace({pd.NA: np.nan})
    return out


def imagej_scores_to_annotator_rows(
    imagej_df: pd.DataFrame,
    *,
    folder_col: str = "folder",
    well_col: str = "well_id",
    image_col: str = "image_name",
    role_col: str = "role",
    raw_score_col: str = "final_score",
    adjusted_score_col: str = "final_score_corrected",
    annotator_name: str = "imageJ",
) -> pd.DataFrame:
    """
    Convert output of score_imagej_rois(...) into the shared long schema.
    """
    df = imagej_df.copy().rename(
        columns={
            folder_col: "Folder",
            well_col: "well",
            image_col: "image_name",
            role_col: "role",
            raw_score_col: "score",
            adjusted_score_col: "adjusted_score",
        }
    )

    required = ["Folder", "well", "image_name", "score", "adjusted_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ImageJ dataframe is missing required columns: {missing}")

    df["Annotator"] = annotator_name

    preferred = [
        "Folder",
        "well",
        "image_name",
        "role",
        "Annotator",
        "score",
        "adjusted_score",
        "frac_pos",
        "corrected_frac_pos",
        "n_rois",
    ]
    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    df = df.astype(object).replace({pd.NA: np.nan})
    return df.reset_index(drop=True)

def score_unet_folders(
    mapping_df: pd.DataFrame,
    *,
    image_base_path: str,
    unet_config: Any,
    layout: Any = PRA_GENERIC_LAYOUT,
    image_order: Sequence[str] = PRA_GENERIC_IMAGE_ORDER,
    annotator_name: str = "unet",
) -> pd.DataFrame:
    """
    Run UNET folder by folder and return rows in the shared long schema:
        Folder | well | image_name | role | Annotator |
        score | adjusted_score | frac_pos | corrected_frac_pos | n_rois
    """
    required = ["Folder", "well", "image_name", "role"]
    missing = [c for c in required if c not in mapping_df.columns]
    if missing:
        raise ValueError(f"mapping_df is missing required columns: {missing}")

    mapping_df = (
        mapping_df[["Folder", "well", "image_name", "role"]]
        .drop_duplicates()
        .copy()
    )

    dups = mapping_df.duplicated(subset=["Folder", "well"], keep=False)
    if dups.any():
        raise ValueError(
            "mapping_df must have unique Folder + well rows.\n"
            f"Examples:\n{mapping_df.loc[dups].head()}"
        )

    out_rows = []
    total_folders = mapping_df["Folder"].nunique()
    for i, (folder, df_folder) in enumerate(mapping_df.groupby("Folder", sort=False)):
        print(f"Calculating Folder {i}/{total_folders}...")
        image_storage_dir = Path(image_base_path) / str(folder)
        if not image_storage_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {image_storage_dir}")

        image_names = [
            p.name for p in list_image_files(
                image_storage_dir,
                allowed_suffixes=(".tif", ".tiff"),
            )
        ]

        if len(image_names) != len(image_order):
            raise ValueError(
                f"Invalid number of images in {image_storage_dir}: "
                f"{len(image_names)} instead of {len(image_order)}"
            )

        res = run_image_analysis(
            layout=layout,
            image_order=image_order,
            image_filenames=image_names,
            input_root=str(image_storage_dir),
            segmented_output_dir = "./tmp",
            segmented_url_prefix = "",
            unet_config=unet_config,
            qc=False,
            assay_type="crossmatch"
        )

        well_res = res["wells"]

        unet_rows = []
        for well, well_info in well_res.items():
            frac_pos = well_info["frac_pos"]
            corrected_frac_pos = well_info["frac_pos_corrected"]
            n_rois = well_info["n_rois"]

            unet_rows.append(
                {
                    "Folder": folder,
                    "well": well,
                    "score": convert_frac_pos_to_score(frac_pos),
                    "adjusted_score": convert_frac_pos_to_score(corrected_frac_pos),
                    "frac_pos": frac_pos,
                    "corrected_frac_pos": corrected_frac_pos,
                    "n_rois": n_rois,
                }
            )

        unet_df = pd.DataFrame(unet_rows).merge(
            df_folder[["well", "image_name", "role"]].drop_duplicates(),
            on="well",
            how="left",
            validate="one_to_one",
        )
        unet_df["Annotator"] = annotator_name
        out_rows.append(unet_df)

    out = pd.concat(out_rows, ignore_index=True)

    preferred = [
        "Folder",
        "well",
        "image_name",
        "role",
        "Annotator",
        "score",
        "adjusted_score",
        "frac_pos",
        "corrected_frac_pos",
        "n_rois",
    ]
    existing = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining]

    out = out.astype(object).replace({pd.NA: np.nan})
    return out.reset_index(drop=True)
# ---------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------

def concat_annotator_frames(
    frames: Sequence[pd.DataFrame],
    *,
    preferred_cols: Sequence[str] = (
        "Folder",
        "well",
        "image_name",
        "role",
        "Annotator",
        "score",
        "adjusted_score",
        "frac_pos",
        "corrected_frac_pos",
        "n_rois",
    ),
) -> pd.DataFrame:
    """
    Concatenate annotator tables in the shared long schema.
    Missing columns are filled with np.nan.
    """
    frames = [f.copy() for f in frames if f is not None and len(f) > 0]
    if len(frames) == 0:
        return pd.DataFrame(columns=list(preferred_cols))

    all_cols = []
    for f in frames:
        for c in f.columns:
            if c not in all_cols:
                all_cols.append(c)

    for f in frames:
        for c in all_cols:
            if c not in f.columns:
                f[c] = np.nan

    ordered_cols = [c for c in preferred_cols if c in all_cols]
    ordered_cols += [c for c in all_cols if c not in ordered_cols]

    out = pd.concat([f[ordered_cols] for f in frames], ignore_index=True, sort=False)
    out = out.astype(object).replace({pd.NA: np.nan})

    sort_cols = [c for c in ["Folder", "well", "image_name", "Annotator"] if c in out.columns]
    if len(sort_cols) > 0:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return out


def build_total_result_dataframe(
    *,
    score_sheet_file_path: str,
    imagej_csv_path: str,
    unet_config: Any,
    score_sheet_sep: str = ";",
    imagej_sep: str = ",",
    ext_image_root: str | Path = "./ext_images",
    experiment_image_root: str | Path = "./experiment_readout_images",
    layout: Any = PRA_GENERIC_LAYOUT,
    image_order: Sequence[str] = PRA_GENERIC_IMAGE_ORDER,
    imagej_calibrator_cls: Type = PCNCGaussian2DCalibrator,
    imagej_classifier_cls: Type = ROIClassifierGaussian2D3Way,
    imagej_classifier_kwargs: Optional[Dict[str, Any]] = None,
    imagej_annotator_name: str = "imageJ",
    unet_annotator_name: str = "unet",
    drop_missing_manual_scores: bool = False,
    output_csv_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline that returns one long dataframe containing:
        manual annotators + imageJ + unet

    Final schema:
        Folder | well | image_name | role | Annotator | score | adjusted_score
    """
    if output_csv_path is not None:
        save_path = os.path.dirname(output_csv_path)
        manual_df_path = os.path.join(save_path, "manual_df.csv")
        imagej_df_path = os.path.join(save_path, "imagej_df.csv")
        unet_df_path = os.path.join(save_path, "unet_df.csv")

    imagej_classifier_kwargs = imagej_classifier_kwargs or {}

    if os.path.isfile(manual_df_path):
        manual_df = pd.read_csv(manual_df_path, index_col = False)
    else:
        manual_df = reshape_scores_with_images(
            score_sheet_file_path,
            sep=score_sheet_sep,
            src_root=ext_image_root,
            dst_root=experiment_image_root,
            layout=layout,
            image_order=image_order,
            drop_missing_scores=drop_missing_manual_scores,
            verbose=verbose,
        )

        # we skip this folder as we do not know if thats a real folder
        manual_df = manual_df.loc[
            ~((manual_df["Folder"] == "20251021_25720338") & (manual_df["PRA"] == 3.0))
        ]
        manual_df = manual_df.loc[
            ~((manual_df["Folder"] == "RUN_F6681F983769") & (manual_df["PRA"] == 3.0))
        ]

        if output_csv_path is not None:
            manual_df.to_csv(manual_df_path, index = False)

    
    if os.path.isfile(imagej_df_path):
        imagej_df = pd.read_csv(imagej_df_path, index_col = False)
    else:
        imagej_scored = score_imagej_rois(
            imagej_csv_path,
            sep=imagej_sep,
            layout=layout,
            image_order=image_order,
            calibrator_cls=imagej_calibrator_cls,
            classifier_cls=imagej_classifier_cls,
            classifier_kwargs=imagej_classifier_kwargs,
            verbose=verbose,
        )

        imagej_df = imagej_scores_to_annotator_rows(
            imagej_scored,
            annotator_name=imagej_annotator_name,
        )

        if output_csv_path is not None:
            imagej_df.to_csv(imagej_df_path, index = False)

    if os.path.isfile(unet_df_path):
        unet_df = pd.read_csv(unet_df_path, index_col = False)
    else:
        mapping_df = concat_annotator_frames([manual_df, imagej_df])[
            ["Folder", "well", "image_name", "role"]
        ].drop_duplicates()

        unet_df = score_unet_folders(
            mapping_df,
            image_base_path=str(experiment_image_root),
            unet_config=unet_config,
            layout=layout,
            image_order=image_order,
            annotator_name=unet_annotator_name,
        )

        if output_csv_path is not None:
            unet_df.to_csv(unet_df_path, index = False)

    total_df = concat_annotator_frames([manual_df, imagej_df, unet_df])

    if output_csv_path is not None:
        total_df.to_csv(output_csv_path, index=False)

    return total_df


def run_external_experiments(
    score_sheet_file_path: str,
    imagej_csv_path: str,
    image_base_path: str,
    csv_output_file: str,
    unet_config: Any,
    *,
    score_sheet_sep: str = ";",
    imagej_sep: str = ",",
) -> pd.DataFrame:
    """
    Convenience wrapper around build_total_result_dataframe(...).
    """
    return build_total_result_dataframe(
        score_sheet_file_path=score_sheet_file_path,
        imagej_csv_path=imagej_csv_path,
        unet_config=unet_config,
        score_sheet_sep=score_sheet_sep,
        imagej_sep=imagej_sep,
        ext_image_root="./ext_images",
        experiment_image_root=image_base_path,
        output_csv_path=csv_output_file,
    )
