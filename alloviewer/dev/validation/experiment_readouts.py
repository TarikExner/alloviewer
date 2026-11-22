#!/usr/bin/env python3
"""Score Frankfurt and Berlin CDC-PRA readout images using AlloViewer UNet.

Input score sheets are *already* in long format and are never rebuilt, edited,
filtered, or overwritten. The original human rows are combined with new UNet
rows; `site` identifies Frankfurt or Berlin.

Frankfurt: PRA_GENERIC_LAYOUT (A1/B1 negative; A10/B10 positive).
Berlin:   same acquisition order, but A10 is the only positive control;
          B10 is a sample. Both sites use A1/B1 as negative controls.

The CSV's (Folder, well, image_name) mapping is authoritative. Images are NOT
assigned to wells by sorting filenames. In particular, this preserves Berlin
Bild_*.tif mappings and any acquisition numbering offsets.

Run from the directory containing ./experiment_readout_images, ./berlin_validation,
./models and ./results. See --help for options.
"""
from __future__ import annotations

import argparse
import copy
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED = {"Folder", "well", "image_name", "role", "Annotator", "score"}
IDENTITY = ["site", "Folder", "well", "image_name", "role"]
SITES = ("Frankfurt", "Berlin")


def site_layout(site: str):
    """Return an independent PlateLayout; never modify the package constant."""
    from ...image_analysis.structs import PlateLayout
    from ...image_analysis.utils import PRA_GENERIC_LAYOUT

    if site not in SITES:
        raise ValueError(f"Unknown site: {site!r}")
    roles = dict(PRA_GENERIC_LAYOUT.wells)
    if site == "Berlin":
        roles["B10"] = "sample"
    positives = {well for well, role in roles.items() if role == "positive"}
    negatives = {well for well, role in roles.items() if role == "negative"}
    expected_pos = {"A10", "B10"} if site == "Frankfurt" else {"A10"}
    if positives != expected_pos or negatives != {"A1", "B1"}:
        raise ValueError(f"Unexpected {site} control layout: PC={positives}, NC={negatives}")
    return PlateLayout(wells=roles)


def _read_score_sheet(path: str | Path, site: str) -> pd.DataFrame:
    """Read an existing long-form human score sheet without changing its scores."""
    source = Path(path).expanduser().resolve(strict=True)
    df = pd.read_csv(source, dtype={"Folder": str, "well": str, "image_name": str,
                                    "Annotator": str, "role": str})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if "site" in df.columns and not df["site"].dropna().eq(site).all():
        raise ValueError(f"{source}: existing site column conflicts with {site}")
    if df.empty:
        raise ValueError(f"{source}: empty score sheet")
    for col in ("Folder", "well", "image_name", "Annotator"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{source}: missing {col}")
    for col in ("Folder", "image_name"):
        bad = [v for v in df[col].unique() if v in {".", ".."} or Path(v).name != v]
        if bad:
            raise ValueError(f"{source}: unsafe {col} values: {bad[:3]}")

    layout = site_layout(site).wells
    unknown = sorted(set(df["well"]) - set(layout))
    if unknown:
        raise ValueError(f"{source}: wells not in layout: {unknown}")

    df.insert(0, "site", site) if "site" not in df.columns else None
    df["Annotator"] = df["Annotator"].astype(str)
    # Catch an accidentally supplied two-annotator CSV before running a model.
    n_annotators = df.groupby(["Folder", "well"])["Annotator"].nunique()
    incomplete = n_annotators[n_annotators != 3]
    if not incomplete.empty:
        raise ValueError(
            f"{source}: expected three annotators per well, but "
            f"{len(incomplete)} wells differ. Examples: {incomplete.head().to_dict()}"
        )
    # Existing Berlin B10 roles are still 'positive' in the curated human CSV.
    # Preserve them for audit, but use the corrected role for all calculations.
    df["role_from_score_sheet"] = df["role"]
    correct_role = df["well"].map(layout)
    unexpected = (df["role"] != correct_role) & ~(
        (site == "Berlin") & (df["well"] == "B10") & (df["role"] == "positive")
    )
    if unexpected.any():
        raise ValueError(f"{source}: unexpected role mismatch, e.g.\n"
                         f"{df.loc[unexpected, ['Folder','well','role']].head()}")
    df["role"] = correct_role

    # A curated sheet may legitimately contain repeated scoring records with
    # distinct metadata. Preserve all of them, but make the issue visible.
    duplicated = df.duplicated(["Folder", "well", "image_name", "Annotator"], keep=False)
    if duplicated.any():
        warnings.warn(
            f"{site}: {int(duplicated.sum())} human rows belong to repeated "
            "(Folder, well, image_name, Annotator) keys; none are removed. "
            f"Example:\n{df.loc[duplicated, ['Folder', 'well', 'Annotator']].head()}",
            stacklevel=2,
        )
    print(f"{site}: loaded {len(df)} existing human scores from {source}", flush=True)
    return df


def load_existing_score_sheets(
    frankfurt_csv: str | Path = "./experiment_readout_images/expert_scoring_long.csv",
    berlin_csv: str | Path = "./berlin_validation/Berlin_expert_scoring_long.csv",
) -> pd.DataFrame:
    """Only read and concatenate the curated human score sheets."""
    frankfurt = _read_score_sheet(frankfurt_csv, "Frankfurt")
    berlin = _read_score_sheet(berlin_csv, "Berlin")
    return pd.concat([frankfurt, berlin], ignore_index=True, sort=False)


def image_mapping_from_scores(human: pd.DataFrame) -> pd.DataFrame:
    """Validate the existing well-to-image mapping and return one row per well."""
    from ...image_analysis.utils import PRA_GENERIC_IMAGE_ORDER

    mapping = human[IDENTITY].drop_duplicates().copy()
    expected = set(PRA_GENERIC_IMAGE_ORDER)
    if len(expected) != len(PRA_GENERIC_IMAGE_ORDER):
        raise ValueError("PRA_GENERIC_IMAGE_ORDER contains repeated wells")
    if PRA_GENERIC_IMAGE_ORDER[-1] != "A10":
        raise ValueError("Expected last acquired well to be A10; verify Berlin ordering")

    for (site, folder), group in mapping.groupby(["site", "Folder"], sort=False):
        wells = set(group["well"])
        if wells != expected:
            raise ValueError(f"{site}/{folder}: missing/extra wells: "
                             f"missing={sorted(expected-wells)}, extra={sorted(wells-expected)}")
        if len(group) != len(expected) or group["image_name"].nunique() != len(expected):
            raise ValueError(f"{site}/{folder}: ambiguous well-to-image mapping")
        required_roles = site_layout(site).wells
        if any(row.role != required_roles[row.well] for row in group.itertuples()):
            raise ValueError(f"{site}/{folder}: wrong control role in mapping")
    return mapping


def _score_or_nan(value: Any) -> float:
    """Do not convert missing fractions into score 8."""
    from ...image_analysis.utils import convert_frac_pos_to_score

    if value is None or not np.isfinite(float(value)):
        return np.nan
    return float(convert_frac_pos_to_score(float(value)))


def score_unet_folders(
    mapping_df: pd.DataFrame,
    *,
    image_roots: Mapping[str, str | Path],
    unet_config: Mapping[str, Any] | None = None,
    preview_root: str | Path | None = None,
    qc: bool = False,
) -> pd.DataFrame:
    """Run the standard UNet image pipeline once per assay using site-specific controls."""
    from ...image_analysis.pipeline import run_image_analysis
    from ...image_analysis.utils import PRA_GENERIC_IMAGE_ORDER

    if not set(SITES).issubset(image_roots):
        raise ValueError(f"image_roots must contain both {SITES}")
    mapping_df = image_mapping_from_scores(mapping_df)
    out: list[dict[str, Any]] = []
    groups = list(mapping_df.groupby(["site", "Folder"], sort=True))

    for i, ((site, folder), group) in enumerate(groups, 1):
        root = Path(image_roots[site]).expanduser().resolve()
        image_dir = (root / folder).resolve()
        if not image_dir.is_relative_to(root) or not image_dir.is_dir():
            raise FileNotFoundError(f"Missing {site} image folder: {image_dir}")
        by_well = group.set_index("well")
        image_names = [str(by_well.loc[well, "image_name"]) for well in PRA_GENERIC_IMAGE_ORDER]
        missing = [name for name in image_names if not (image_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{site}/{folder}: {len(missing)} TIFFs missing, e.g. {missing[:5]}")

        print(f"[{i}/{len(groups)}] {site}/{folder}: "
              f"PC={[k for k,v in site_layout(site).wells.items() if v=='positive']}",
              flush=True)

        def process(preview_dir: Path) -> dict[str, Any]:
            return run_image_analysis(
                layout=site_layout(site),
                image_order=list(PRA_GENERIC_IMAGE_ORDER),
                image_filenames=image_names,
                input_root=image_dir,
                segmented_output_dir=preview_dir,
                segmented_url_prefix="",
                unet_config=copy.deepcopy(dict(unet_config)) if unet_config is not None else None,
                qc=qc,
                # Per-well scores do not require an HLA layout. The old module
                # used crossmatch mode for this same reason.
                assay_type="crossmatch",
                column_modes={col: "T" for col in range(1, 11)},
            )

        if preview_root is None:
            with tempfile.TemporaryDirectory(prefix="readout_unet_") as tmp:
                result = process(Path(tmp))
        else:
            preview = Path(preview_root) / site / folder
            preview.mkdir(parents=True, exist_ok=True)
            result = process(preview)

        wells = result["wells"]
        if set(wells) != set(PRA_GENERIC_IMAGE_ORDER):
            raise ValueError(f"{site}/{folder}: pipeline returned unexpected wells")
        for well in PRA_GENERIC_IMAGE_ORDER:
            info = wells[well]
            raw = info.get("frac_pos")
            corrected = info.get("frac_pos_corrected")
            out.append({
                "site": site,
                "Folder": folder,
                "well": well,
                "image_name": by_well.loc[well, "image_name"],
                "role": by_well.loc[well, "role"],
                "role_from_score_sheet": by_well.loc[well, "role_from_score_sheet"]
                if "role_from_score_sheet" in by_well else by_well.loc[well, "role"],
                "Annotator": "unet",
                "score": _score_or_nan(raw),
                "adjusted_score": _score_or_nan(corrected),
                "frac_pos": raw,
                "corrected_frac_pos": corrected,
                "n_rois": info.get("n_rois"),
            })

    return pd.DataFrame(out)


def _propagate_unambiguous_metadata(human: pd.DataFrame, unet: pd.DataFrame) -> pd.DataFrame:
    """Copy human metadata to UNet rows only where the value is unambiguous."""
    reserved = set(unet.columns) | {"site", "Folder", "well", "image_name",
                                    "role", "Annotator", "score", "adjusted_score"}
    extras = [col for col in human.columns if col not in reserved]
    if not extras:
        return unet
    keys = ["site", "Folder", "well", "image_name"]
    # Repeated human scores with different PRA values cannot give a unique
    # value for that field on the UNet row. Leave that field empty.
    meta = human.groupby(keys, dropna=False)[extras].agg(
        lambda series: series.dropna().iloc[0] if series.dropna().nunique() == 1 else np.nan
    ).reset_index()
    return unet.merge(meta, on=keys, how="left", validate="one_to_one")


def build_total_result_dataframe(
    *,
    unet_config: Mapping[str, Any] | None = None,
    frankfurt_scores: str | Path = "./experiment_readout_images/expert_scoring_long.csv",
    berlin_scores: str | Path = "./berlin_validation/Berlin_expert_scoring_long.csv",
    frankfurt_images: str | Path = "./experiment_readout_images",
    berlin_images: str | Path = "./berlin_validation",
    output_csv_path: str | Path | None = "./results/readout_scores_with_unet.csv",
    unet_csv_path: str | Path | None = "./results/readout_unet_only.csv",
    preview_root: str | Path | None = None,
    limit_folders_per_site: int | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Append UNet readouts to curated human scores; never rewrite source sheets."""
    human = load_existing_score_sheets(frankfurt_scores, berlin_scores)
    mapping = image_mapping_from_scores(human)
    if limit_folders_per_site is not None:
        if limit_folders_per_site < 1:
            raise ValueError("limit_folders_per_site must be >= 1")
        selected = (mapping[["site", "Folder"]].drop_duplicates()
                    .sort_values(["site", "Folder"])
                    .groupby("site", sort=False).head(limit_folders_per_site))
        mapping = mapping.merge(selected, on=["site", "Folder"], validate="many_to_one")
        human = human.merge(selected, on=["site", "Folder"], validate="many_to_one")

    if dry_run:
        print("Dry run: source sheets were read; no models executed or files written.")
        print(mapping.groupby(["site", "Folder"], sort=True).size().to_string())
        return human

    unet = score_unet_folders(
        mapping,
        image_roots={"Frankfurt": frankfurt_images, "Berlin": berlin_images},
        unet_config=unet_config,
        preview_root=preview_root,
    )
    unet = _propagate_unambiguous_metadata(human, unet)
    # Every human row stays intact; one new UNet row per unique site/folder/well.
    combined = pd.concat([human, unet], ignore_index=True, sort=False)
    priority = ["site", "Folder", "well", "image_name", "role", "Annotator",
                "score", "adjusted_score", "frac_pos", "corrected_frac_pos", "n_rois"]
    combined = combined[[*priority, *(c for c in combined if c not in priority)]]
    if unet_csv_path is not None:
        path = Path(unet_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        unet.to_csv(path, index=False)
        print(f"UNet-only readouts: {path} ({len(unet)} rows)", flush=True)
    if output_csv_path is not None:
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
        print(f"Combined existing human + UNet scores: {path} ({len(combined)} rows)", flush=True)
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frankfurt-scores", default="./experiment_readout_images/expert_scoring_long.csv")
    parser.add_argument("--berlin-scores", default="./berlin_validation/Berlin_expert_scoring_long.csv")
    parser.add_argument("--frankfurt-images", default="./experiment_readout_images")
    parser.add_argument("--berlin-images", default="./berlin_validation")
    parser.add_argument("--out", default="./results/readout_scores_with_unet.csv")
    parser.add_argument("--unet-out", default="./results/readout_unet_only.csv")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--unet-mode", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--model-file", default=None)
    parser.add_argument("--preview-root", default=None)
    parser.add_argument("--limit-folders-per-site", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = None
    if not args.dry_run:
        from ...image_analysis.config import UNET_CONFIG
        config = copy.deepcopy(UNET_CONFIG)
        config.update({
            "unet_mode": args.unet_mode,
            "model_dir": args.model_dir,
            "model_file": args.model_file or f"best_{args.unet_mode}_tiles_S512_seed187.pth",
            "compute_instances": True,
        })

    build_total_result_dataframe(
        unet_config=config,
        frankfurt_scores=args.frankfurt_scores,
        berlin_scores=args.berlin_scores,
        frankfurt_images=args.frankfurt_images,
        berlin_images=args.berlin_images,
        output_csv_path=args.out,
        unet_csv_path=args.unet_out,
        preview_root=args.preview_root,
        limit_folders_per_site=args.limit_folders_per_site,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

