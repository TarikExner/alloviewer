import re
import os
import numpy as np
import pandas as pd
from alloviewer.main import run_image_analysis
from ...image_analysis.utils import (
    PRA_GENERIC_LAYOUT,
    PRA_GENERIC_IMAGE_ORDER,
    convert_frac_pos_to_score
)
from typing import Any, Iterable
import shutil
from pathlib import Path


def _natural_image_sort_key(name: str):
    """
    Sort filenames by embedded numbers, so:
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


def _list_image_files(
    folder: Path,
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> list[Path]:
    allowed_suffixes = {s.lower() for s in allowed_suffixes}
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_suffixes
    ]
    return sorted(files, key=lambda p: _natural_image_sort_key(p.name))


def _prepare_valid_folders_and_copy_images(
    folders: Iterable[str],
    src_root: str | Path = "./ext_images",
    dst_root: str | Path = "./experiment_readout_images",
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Only keep folders with exactly 60 images.

    For valid folders:
    - map fixed well order to image file names
    - copy images to dst_root / folder only if needed
    - if already copied correctly, keep them as they are
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    folder_to_image_map: dict[str, dict[str, str]] = {}
    valid_folders: list[str] = []

    for folder_name in pd.unique(pd.Series(list(folders)).dropna()):
        folder_name = str(folder_name)
        src_folder = src_root / folder_name

        if not src_folder.exists() or not src_folder.is_dir():
            print(f"Skipping {folder_name}: source folder not found")
            continue

        src_images = _list_image_files(src_folder, allowed_suffixes=allowed_suffixes)

        if len(src_images) != 60:
            print(f"Skipping {folder_name}: found {len(src_images)} images instead of 60")
            continue

        dst_folder = dst_root / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        dst_images = _list_image_files(dst_folder, allowed_suffixes=allowed_suffixes)

        # If destination already has the same 60 images, keep them as they are
        src_names = [p.name for p in src_images]
        dst_names = [p.name for p in dst_images]

        if dst_names != src_names:
            # refresh destination so it matches source exactly
            for p in dst_images:
                p.unlink()

            for img in src_images:
                shutil.copy2(img, dst_folder / img.name)

        folder_to_image_map[folder_name] = {
            well: img.name
            for well, img in zip(PRA_GENERIC_IMAGE_ORDER, src_images)
        }
        valid_folders.append(folder_name)

    return folder_to_image_map, valid_folders


def reshape_scores_with_images(
    csv_path_or_df,
    sep: str = ";",
    src_root: str | Path = "./ext_images",
    dst_root: str | Path = "./experiment_readout_images",
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    drop_missing_scores: bool = False,
) -> pd.DataFrame:
    """
    Convert a wide scoring sheet to long format and add image names and role.

    Output columns include:
    Folder | well | image_name | role | Annotator | score
    plus all remaining non-well columns.
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df, sep=sep)

    if "Folder" not in df.columns:
        raise ValueError("Input must contain a 'Folder' column.")

    well_pattern = re.compile(r"^[A-Z]\d+$")
    well_cols = [c for c in df.columns if well_pattern.match(str(c))]
    id_cols = [c for c in df.columns if c not in well_cols]

    folder_to_image_map, valid_folders = _prepare_valid_folders_and_copy_images(
        folders=df["Folder"].dropna().unique(),
        src_root=src_root,
        dst_root=dst_root,
        allowed_suffixes=allowed_suffixes,
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
    out["role"] = out["well"].map(
        PRA_GENERIC_LAYOUT.wells if hasattr(PRA_GENERIC_LAYOUT, "wells")
        else PRA_GENERIC_LAYOUT
    )

    if drop_missing_scores:
        out = out.dropna(subset=["score"]).copy()

    preferred = ["Folder", "well", "image_name", "role", "Annotator", "score"]
    existing_preferred = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    return out.reset_index(drop=True)


def run_external_experiments(
    score_sheet_file_path: str,
    image_base_path: str,
    csv_output_file: str,
    unet_config: Any,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Run analysis on copied external images and write AI scores back into the
    long-format score sheet.

    Adds:
    - role
    - AI_score_raw
    - AI_score_corr
    """
    score_sheet = reshape_scores_with_images(
        score_sheet_file_path,
        sep=sep,
        src_root="./ext_images",
        dst_root=image_base_path,
    )

    if "role" not in score_sheet.columns:
        score_sheet["role"] = score_sheet["well"].map(
            PRA_GENERIC_LAYOUT.wells if hasattr(PRA_GENERIC_LAYOUT, "wells") else PRA_GENERIC_LAYOUT
        )

    score_sheet["AI_score_raw"] = np.nan
    score_sheet["AI_score_corr"] = np.nan

    image_folders = score_sheet["Folder"].dropna().unique()

    for folder in image_folders:
        print(folder)

        image_storage_dir = Path(image_base_path) / str(folder)
        if not image_storage_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {image_storage_dir}")

        image_names = [
            p.name for p in _list_image_files(
                image_storage_dir,
                allowed_suffixes=(".tif", ".tiff"),
            )
        ]

        if len(image_names) != 60:
            raise ValueError(f"Invalid number of images in {image_storage_dir}: {len(image_names)}")

        res = run_image_analysis(
            layout=PRA_GENERIC_LAYOUT,
            image_order=PRA_GENERIC_IMAGE_ORDER,
            image_filenames=image_names,
            template_filename="template",
            data_dir=str(image_storage_dir),
            unet_config=unet_config,
            qc=False,
        )

        well_res = res["wells"]

        for well, well_info in well_res.items():
            frac_pos = well_info["frac_pos"]
            frac_pos_adj = well_info["frac_pos_corrected"]

            final_score = convert_frac_pos_to_score(frac_pos)
            final_score_adj = convert_frac_pos_to_score(frac_pos_adj)

            score_sheet.loc[
                (score_sheet["Folder"] == folder) &
                (score_sheet["well"] == well),
                "AI_score_raw"
            ] = final_score

            score_sheet.loc[
                (score_sheet["Folder"] == folder) &
                (score_sheet["well"] == well),
                "AI_score_corr"
            ] = final_score_adj

    score_sheet.to_csv(csv_output_file, index=False)
    return score_sheet
