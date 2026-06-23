from __future__ import annotations

import copy
import glob
import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from alloviewer.image_analysis.segmenter import SegmenterConfig, SegmenterUNet

from .utils import (  # noqa: E402
    decode_json_maybe,
    load_human_roi_counts,
    segment_one_h5_entry
)

DEFAULT_UNET_MODE: Literal["small", "medium", "large"] = "small"
DEFAULT_OUTPUT_DIR = "./epoch_model_human_count_comparison"
DEFAULT_MODEL_DIR = "./models"
DEFAULT_H5_PATH = "./image_datasets/human_annotated_images.h5"
DEFAULT_HUMAN_CSV_DIR = "./human_annotations"

# Epoch filter defaults.
#
# Because later checkpoints can over-optimize to the simulator distribution,
# the default helper run only considers checkpoints up to epoch 40.
# Set DEFAULT_MAX_EPOCH = None, or pass max_epoch=None, to include all epochs.
DEFAULT_INCLUDE_EPOCHS: Optional[Sequence[int]] = None
DEFAULT_MIN_EPOCH: Optional[int] = None
DEFAULT_MAX_EPOCH: Optional[int] = 100

_EPOCH_RE = re.compile(r"(?:^|[_-])epoch[_-]?(\d+)(?:\D|$)", re.IGNORECASE)


def parse_epoch_from_model_file(path: str) -> Optional[int]:
    """Return epoch number parsed from a checkpoint filename."""
    name = os.path.basename(path)
    m = _EPOCH_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


def epoch_model_pattern(unet_mode: str) -> str:
    """Checkpoint pattern for one model size."""
    return f"{unet_mode}_*_epoch_*.pth"


def _sort_model_paths_by_epoch(paths: Sequence[str]) -> List[str]:
    """Sort checkpoint paths by parsed epoch, then filename."""
    return sorted(
        [str(p) for p in paths],
        key=lambda p: (
            parse_epoch_from_model_file(p)
            if parse_epoch_from_model_file(p) is not None
            else 10**12,
            os.path.basename(p),
        ),
    )


def filter_model_paths_by_epoch(
    model_paths: Sequence[str],
    *,
    include_epochs: Optional[Sequence[int]] = None,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[str]:
    """
    Filter checkpoint paths by parsed epoch.

    Parameters
    ----------
    model_paths:
        Checkpoint paths.
    include_epochs:
        Optional explicit set/list of epochs to keep, e.g. [5, 10, 15, 20, 25, 30, 35, 40].
        If provided, only these epochs are kept.
    min_epoch:
        Optional lower epoch bound, inclusive.
    max_epoch:
        Optional upper epoch bound, inclusive.

    Notes
    -----
    The filters are combined. For example:
      include_epochs=[10, 20, 30, 40], max_epoch=30
    keeps only epochs 10, 20, and 30.
    """
    include_set = None
    if include_epochs is not None:
        include_set = {int(e) for e in include_epochs}

    out: List[str] = []

    for p in model_paths:
        ep = parse_epoch_from_model_file(p)
        if ep is None:
            continue

        if include_set is not None and ep not in include_set:
            continue

        if min_epoch is not None and ep < int(min_epoch):
            continue

        if max_epoch is not None and ep > int(max_epoch):
            continue

        out.append(str(p))

    out = _sort_model_paths_by_epoch(out)

    if not out:
        raise FileNotFoundError(
            "No checkpoint files remained after epoch filtering. "
            f"include_epochs={include_epochs}, min_epoch={min_epoch}, max_epoch={max_epoch}"
        )

    return out


def find_epoch_model_files(
    model_dir: str,
    *,
    unet_mode: Literal["small", "medium", "large"] = DEFAULT_UNET_MODE,
    model_pattern: Optional[str] = None,
    include_epochs: Optional[Sequence[int]] = None,
    min_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
) -> List[str]:
    """
    Find saved epoch checkpoints for one UNet size.

    By default this searches only files such as:
      small_*_epoch_0050.pth

    It does not include best_*.pth checkpoints.

    Epoch filtering is applied after file discovery.
    """
    pattern = model_pattern or epoch_model_pattern(unet_mode)
    paths = glob.glob(os.path.join(model_dir, pattern))

    epoch_paths = [p for p in paths if parse_epoch_from_model_file(p) is not None]
    epoch_paths = _sort_model_paths_by_epoch(epoch_paths)

    if not epoch_paths:
        raise FileNotFoundError(
            f"No epoch model files matching pattern '{pattern}' were found in {model_dir}."
        )

    epoch_paths = filter_model_paths_by_epoch(
        epoch_paths,
        include_epochs=include_epochs,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    return epoch_paths


def _segmenter_cfg_for_model(
    *,
    model_path: str,
    unet_mode: Literal["small", "medium", "large"],
    base_cfg: Optional[SegmenterConfig],
) -> SegmenterConfig:
    """Copy/create a SegmenterConfig and point it to one checkpoint."""
    if base_cfg is None:
        cfg = SegmenterConfig(
            unet_mode=unet_mode,
            compute_instances=True,
            input_is_tiles=True,
        )
    else:
        cfg = copy.deepcopy(base_cfg)

    cfg.unet_mode = unet_mode
    cfg.compute_instances = True
    cfg.input_is_tiles = True
    cfg.model_dir = os.path.dirname(os.path.abspath(model_path))
    cfg.model_file = os.path.basename(model_path)
    return cfg

def run_one_model_on_h5(
    *,
    h5_path: str,
    model_path: str,
    unet_mode: Literal["small", "medium", "large"] = DEFAULT_UNET_MODE,
    segmenter_cfg: Optional[SegmenterConfig] = None,
    update_cell_mask: bool = False,
    desc: Optional[str] = None,
) -> pd.DataFrame:
    """Run one checkpoint on all entries in the human-count H5 file."""
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    epoch = parse_epoch_from_model_file(model_path)
    if epoch is None:
        raise ValueError(f"Could not parse epoch from model file: {model_path}")

    cfg = _segmenter_cfg_for_model(
        model_path=model_path,
        unet_mode=unet_mode,
        base_cfg=segmenter_cfg,
    )
    segmenter = SegmenterUNet(cfg)

    rows: List[Dict[str, Any]] = []

    with h5py.File(h5_path, "r") as f:
        if "imgs" not in f or "meta" not in f:
            raise KeyError("H5 file must contain datasets '/imgs' and '/meta'.")

        imgs_ds = f["imgs"]
        meta_ds = f["meta"]

        n_total = int(imgs_ds.shape[0])
        n_written = int(f.attrs.get("written", n_total))
        n_use = min(n_total, n_written)

        iterator = tqdm(
            range(n_use),
            desc=desc or f"Human-count set | {unet_mode} epoch {epoch}",
            dynamic_ncols=True,
        )

        for i in iterator:
            meta = decode_json_maybe(meta_ds[i])
            imgs_tiles = imgs_ds[i]
            row = segment_one_h5_entry(
                segmenter,
                imgs_tiles,
                meta,
                update_cell_mask=update_cell_mask,
            )
            row["image_index"] = int(i)
            row["unet_mode"] = unet_mode
            row["epoch"] = int(epoch)
            row["model_file"] = os.path.basename(model_path)
            row["model_path"] = os.path.abspath(model_path)
            rows.append(row)

    del segmenter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str = "pearson") -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")

    if method == "spearman":
        xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        return float(np.corrcoef(xr, yr)[0, 1])

    return float(np.corrcoef(x, y)[0, 1])


def summarize_count_errors(
    per_image_df: pd.DataFrame,
    *,
    selection_metric: str = "mae",
) -> pd.DataFrame:
    """Summarize count agreement for each checkpoint."""
    required = {
        "unet_mode",
        "epoch",
        "model_file",
        "model_path",
        "human_roi_count",
        "unet_roi_count",
    }
    missing = required - set(per_image_df.columns)
    if missing:
        raise ValueError(f"per_image_df is missing required columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    group_cols = ["unet_mode", "epoch", "model_file", "model_path"]

    for key, group in per_image_df.groupby(group_cols, dropna=False):
        unet_mode, epoch, model_file, model_path = key
        matched = group.dropna(subset=["human_roi_count", "unet_roi_count"]).copy()

        base = {
            "unet_mode": unet_mode,
            "epoch": epoch,
            "model_file": model_file,
            "model_path": model_path,
            "n_images_predicted": int(group["unet_roi_count"].notna().sum()),
            "n_images_with_human_count": int(group["human_roi_count"].notna().sum()),
            "n_matched": int(matched.shape[0]),
        }

        if matched.empty:
            rows.append(
                {
                    **base,
                    "mean_human_count": np.nan,
                    "mean_unet_count": np.nan,
                    "bias": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "median_abs_error": np.nan,
                    "mean_abs_relative_error": np.nan,
                    "pearson_r": np.nan,
                    "spearman_r": np.nan,
                }
            )
            continue

        human = matched["human_roi_count"].to_numpy(dtype=float)
        pred = matched["unet_roi_count"].to_numpy(dtype=float)
        err = pred - human
        abs_err = np.abs(err)

        denom = human.copy()
        denom[denom == 0] = np.nan
        abs_rel = abs_err / denom

        rows.append(
            {
                **base,
                "mean_human_count": float(np.mean(human)),
                "mean_unet_count": float(np.mean(pred)),
                "bias": float(np.mean(err)),
                "mae": float(np.mean(abs_err)),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "median_abs_error": float(np.median(abs_err)),
                "mean_abs_relative_error": float(np.nanmean(abs_rel)),
                "pearson_r": _safe_corr(human, pred, method="pearson"),
                "spearman_r": _safe_corr(human, pred, method="spearman"),
            }
        )

    summary = pd.DataFrame(rows)

    if selection_metric not in summary.columns:
        raise ValueError(
            f"selection_metric='{selection_metric}' is not present in summary columns: "
            f"{list(summary.columns)}"
        )

    lower_is_better = selection_metric not in {"pearson_r", "spearman_r"}
    valid = summary.dropna(subset=[selection_metric]).copy()
    summary["selected"] = False

    if not valid.empty:
        idx = valid[selection_metric].idxmin() if lower_is_better else valid[selection_metric].idxmax()
        summary.loc[idx, "selected"] = True

    return summary.sort_values(
        ["selected", "epoch"],
        ascending=[False, True],
    ).reset_index(drop=True)


def compare_human_counts_for_epoch_models(
    *,
    h5_path: str = DEFAULT_H5_PATH,
    human_csv_dir: str = DEFAULT_HUMAN_CSV_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    unet_mode: Literal["small", "medium", "large"] = DEFAULT_UNET_MODE,
    model_pattern: Optional[str] = None,
    model_paths: Optional[Sequence[str]] = None,
    include_epochs: Optional[Sequence[int]] = DEFAULT_INCLUDE_EPOCHS,
    min_epoch: Optional[int] = DEFAULT_MIN_EPOCH,
    max_epoch: Optional[int] = DEFAULT_MAX_EPOCH,
    segmenter_cfg: Optional[SegmenterConfig] = None,
    selection_metric: str = "mae",
    update_cell_mask: bool = False,
) -> Dict[str, Any]:
    """
    Run saved epoch checkpoints for one UNet size on the human-count image set.

    Writes:
      - per_image_counts_by_epoch.csv
      - summary_by_epoch.csv
      - selected_model.json

    The default selects the checkpoint with lowest MAE versus human counts.

    Epoch filtering
    ---------------
    include_epochs:
        Optional explicit epochs to include.
    min_epoch:
        Optional lower epoch bound, inclusive.
    max_epoch:
        Optional upper epoch bound, inclusive.

    Examples
    --------
    Include only epochs up to 40:
        compare_human_counts_for_epoch_models(max_epoch=40)

    Include specific checkpoints:
        compare_human_counts_for_epoch_models(include_epochs=[5, 10, 15, 20, 25, 30, 35, 40])

    Include all checkpoints:
        compare_human_counts_for_epoch_models(max_epoch=None)
    """
    os.makedirs(output_dir, exist_ok=True)

    if model_paths is None:
        model_paths = find_epoch_model_files(
            model_dir,
            unet_mode=unet_mode,
            model_pattern=model_pattern,
            include_epochs=include_epochs,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
        )
    else:
        model_paths = _sort_model_paths_by_epoch([str(p) for p in model_paths])
        model_paths = filter_model_paths_by_epoch(
            model_paths,
            include_epochs=include_epochs,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
        )

    human_df = load_human_roi_counts(human_csv_dir).copy()
    human_df["Folder"] = human_df["Folder"].astype(str)
    human_df["image_name"] = human_df["image_name"].astype(str)

    prediction_tables: List[pd.DataFrame] = []

    print(
        "Running human-count checkpoint comparison with "
        f"{len(model_paths)} checkpoint(s). "
        f"include_epochs={include_epochs}, min_epoch={min_epoch}, max_epoch={max_epoch}"
    )

    for model_path in model_paths:
        pred_df = run_one_model_on_h5(
            h5_path=h5_path,
            model_path=model_path,
            unet_mode=unet_mode,
            segmenter_cfg=segmenter_cfg,
            update_cell_mask=update_cell_mask,
        )
        prediction_tables.append(pred_df)

    unet_df = pd.concat(prediction_tables, ignore_index=True)
    unet_df["Folder"] = unet_df["Folder"].astype(str)
    unet_df["image_name"] = unet_df["image_name"].astype(str)

    per_image = unet_df.merge(
        human_df,
        on=["Folder", "image_name"],
        how="left",
        validate="many_to_one",
    )

    per_image["count_error"] = per_image["unet_roi_count"] - per_image["human_roi_count"]
    per_image["abs_count_error"] = per_image["count_error"].abs()
    per_image["relative_count_error"] = (
        per_image["count_error"] / per_image["human_roi_count"].replace(0, np.nan)
    )
    per_image["abs_relative_count_error"] = per_image["relative_count_error"].abs()

    summary = summarize_count_errors(per_image, selection_metric=selection_metric)

    per_image_csv = os.path.join(output_dir, "per_image_counts_by_epoch.csv")
    summary_csv = os.path.join(output_dir, "summary_by_epoch.csv")
    selected_json = os.path.join(output_dir, "selected_model.json")

    per_image.sort_values(["epoch", "Folder", "image_name"]).to_csv(per_image_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    selected = summary.loc[summary["selected"]].head(1)
    epoch_filter_info = {
        "include_epochs": None if include_epochs is None else [int(e) for e in include_epochs],
        "min_epoch": None if min_epoch is None else int(min_epoch),
        "max_epoch": None if max_epoch is None else int(max_epoch),
        "n_model_paths": int(len(model_paths)),
        "model_files": [os.path.basename(p) for p in model_paths],
    }

    if selected.empty:
        selected_info: Dict[str, Any] = {
            "selection_metric": selection_metric,
            "unet_mode": unet_mode,
            "epoch_filter": epoch_filter_info,
            "selected": None,
            "reason": "No model had a finite selection metric.",
        }
    else:
        row = selected.iloc[0].to_dict()
        selected_info = {
            "selection_metric": selection_metric,
            "unet_mode": unet_mode,
            "epoch_filter": epoch_filter_info,
            "selected_epoch": None if pd.isna(row.get("epoch")) else int(row["epoch"]),
            "selected_model_file": row.get("model_file"),
            "selected_model_path": row.get("model_path"),
            "selected_metric_value": (
                None if pd.isna(row.get(selection_metric)) else float(row[selection_metric])
            ),
            "row": {
                k: (None if pd.isna(v) else v)
                for k, v in row.items()
                if k != "selected"
            },
        }

    with open(selected_json, "w", encoding="utf-8") as f:
        json.dump(selected_info, f, indent=2)

    return {
        "per_image": per_image,
        "summary": summary,
        "selected": selected_info,
        "paths": {
            "per_image_csv": per_image_csv,
            "summary_csv": summary_csv,
            "selected_json": selected_json,
        },
    }


def run_epoch_model_human_count_comparison(
    unet_mode: Literal["small", "medium", "large"] = DEFAULT_UNET_MODE,
    include_epochs: Optional[Sequence[int]] = DEFAULT_INCLUDE_EPOCHS,
    min_epoch: Optional[int] = DEFAULT_MIN_EPOCH,
    max_epoch: Optional[int] = DEFAULT_MAX_EPOCH,
):
    """
    Convenience runner.

    Defaults to max_epoch=40 via DEFAULT_MAX_EPOCH.
    Pass max_epoch=None to include all epoch checkpoints.
    """
    cfg = SegmenterConfig(
        unet_mode=unet_mode,
        model_dir=DEFAULT_MODEL_DIR,
        model_file="",  # overwritten for each epoch checkpoint
        device="cuda",
        use_amp=True,
        compute_instances=True,
        input_is_tiles=True,
        normalize=True,
        instance_cfg={},
        thr_cell=0.1,
        thr_bound=0.1,
    )

    out = compare_human_counts_for_epoch_models(
        h5_path=DEFAULT_H5_PATH,
        human_csv_dir=DEFAULT_HUMAN_CSV_DIR,
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        unet_mode=unet_mode,
        model_pattern=None,  # defaults to f"{unet_mode}_*_epoch_*.pth"
        include_epochs=include_epochs,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        segmenter_cfg=cfg,
        selection_metric="mae",
        update_cell_mask=False,
    )

    print(out["summary"].head())
    print("\nSelected model:")
    print(json.dumps(out["selected"], indent=2))
    print("\nSaved:")
    for p in out["paths"].values():
        print(p)

    return out

