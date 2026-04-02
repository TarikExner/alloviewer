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
from typing import Any


def reshape_scores(csv_path_or_df, sep=";") -> pd.DataFrame:
    """
    Convert a wide scoring sheet to long format with:
    folder | well | annotator | score
    while keeping all non-well columns.
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df, sep=sep)

    well_pattern = re.compile(r"^[A-Z]\d+$")
    well_cols = [c for c in df.columns if well_pattern.match(str(c))]

    id_cols = [c for c in df.columns if c not in well_cols]

    out = df.melt(
        id_vars=id_cols,
        value_vars=well_cols,
        var_name="well",
        value_name="score"
    )

    # Optional: reorder columns so the main ones come first
    preferred = ["Folder", "well", "Annotator", "score"]
    existing_preferred = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    assert isinstance(out, pd.DataFrame)

    return out


def run_external_experiments(score_sheet_file_path: str,
                             image_base_path: str,
                             csv_output_file: str,
                             unet_config: Any
                             ):

    score_sheet = reshape_scores(score_sheet_file_path)
    image_folders = score_sheet["Folder"].unique()
    score_sheet["AI_score_raw"] = [np.nan for _ in range(score_sheet.shape[0])]
    score_sheet["AI_score_corr"] = [np.nan for _ in range(score_sheet.shape[0])]

    for folder in image_folders:
        print(folder)
        image_storage_dir = os.path.join(image_base_path, folder)
        image_names = os.listdir(image_storage_dir)
        image_names = [file for file in image_names if file.endswith(".tif")]
        image_names.sort()
        if len(image_names) != 60:
            raise ValueError("Invalid length of images")
        res = run_image_analysis(
            layout=PRA_GENERIC_LAYOUT,
            image_order=PRA_GENERIC_IMAGE_ORDER,
            image_filenames=image_names,
            template_filename="template",
            data_dir=image_storage_dir,
            unet_config=unet_config,
            qc = False
        )
        well_res = res["wells"]
        for well in well_res:
            frac_pos = well_res[well]["frac_pos"]
            frac_pos_adj = well_res[well]["frac_pos_corrected"]
            final_score = convert_frac_pos_to_score(frac_pos)
            final_score_adj = convert_frac_pos_to_score(frac_pos_adj)
            score_sheet.loc[
                (score_sheet["Folder"] == folder) &
                (score_sheet["Well"] == well),
                "AI_score_raw"
            ] = final_score
            score_sheet.loc[
                (score_sheet["Folder"] == folder) &
                (score_sheet["Well"] == well),
                "AI_score_corr"
            ] = final_score_adj


    score_sheet.to_csv(csv_output_file, index = False)

    return score_sheet

