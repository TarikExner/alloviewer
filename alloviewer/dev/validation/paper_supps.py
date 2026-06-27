from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


def _copy_attrs(src_obj, dst_obj) -> None:
    """Copy HDF5 attributes."""
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def _safe_chunks(
    src_chunks: tuple[int, ...] | None,
    dst_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    """
    Reuse source chunks when possible, but make sure no chunk dimension
    exceeds the destination shape.
    """
    if src_chunks is None:
        return None

    if len(src_chunks) != len(dst_shape):
        return None

    chunks = []
    for c, s in zip(src_chunks, dst_shape):
        if s <= 0:
            chunks.append(1)
        else:
            chunks.append(max(1, min(int(c), int(s))))

    return tuple(chunks)


def _dataset_creation_kwargs(
    src: h5py.Dataset,
    dst_shape: tuple[int, ...],
) -> dict:
    """
    Preserve relevant HDF5 storage options.
    """
    kwargs = {}

    chunks = _safe_chunks(src.chunks, dst_shape)
    if chunks is not None:
        kwargs["chunks"] = chunks

    if src.compression is not None:
        kwargs["compression"] = src.compression

    if src.compression_opts is not None:
        kwargs["compression_opts"] = src.compression_opts

    if src.shuffle:
        kwargs["shuffle"] = True

    if src.fletcher32:
        kwargs["fletcher32"] = True

    if src.scaleoffset is not None:
        kwargs["scaleoffset"] = src.scaleoffset

    return kwargs


def _copy_dataset_subset(
    src: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    *,
    n_source_samples: int,
    n_keep: int,
    block_size: int,
) -> None:
    """
    Copy a dataset.

    If dataset.shape[0] == n_source_samples, copy only first n_keep entries.
    Otherwise copy the whole dataset unchanged.
    """
    if src.shape == ():
        dst = dst_parent.create_dataset(name, data=src[()])
        _copy_attrs(src, dst)
        return

    subset_first_axis = (
        len(src.shape) >= 1
        and int(src.shape[0]) == int(n_source_samples)
    )

    if subset_first_axis:
        dst_shape = (int(n_keep), *tuple(src.shape[1:]))
    else:
        dst_shape = tuple(src.shape)

    kwargs = _dataset_creation_kwargs(src, dst_shape)

    dst = dst_parent.create_dataset(
        name,
        shape=dst_shape,
        dtype=src.dtype,
        **kwargs,
    )
    _copy_attrs(src, dst)

    if subset_first_axis:
        for start in range(0, n_keep, block_size):
            end = min(n_keep, start + block_size)
            dst[start:end, ...] = src[start:end, ...]
    else:
        # Whole-dataset copy. Usually small metadata/helper arrays.
        dst[...] = src[...]


def _copy_group_subset(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    n_source_samples: int,
    n_keep: int,
    block_size: int,
) -> None:
    """
    Recursively copy groups and datasets, subsetting datasets whose first axis
    matches the sample axis.
    """
    _copy_attrs(src_group, dst_group)

    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            _copy_dataset_subset(
                item,
                dst_group,
                name,
                n_source_samples=n_source_samples,
                n_keep=n_keep,
                block_size=block_size,
            )

        elif isinstance(item, h5py.Group):
            new_group = dst_group.create_group(name)
            _copy_group_subset(
                item,
                new_group,
                n_source_samples=n_source_samples,
                n_keep=n_keep,
                block_size=block_size,
            )

        else:
            raise TypeError(
                f"Unsupported HDF5 object at '{item.name}': {type(item)}"
            )


def subset_h5_for_zenodo(
    h5_paths: Sequence[str | os.PathLike],
    out_dir: str | os.PathLike,
    *,
    n_images: int = 1000,
    image_key: str = "imgs",
    block_size: int = 16,
    overwrite: bool = False,
    tmp_suffix: str = ".tmp",
) -> list[Path]:
    """
    Create reduced HDF5 files for Zenodo upload.

    For each input H5 file, this writes a file with the same basename into
    out_dir, containing the first n_images samples.

    This is IO-only:
      - no dataset class instantiation
      - no transforms
      - no normalization
      - no model inference

    Parameters
    ----------
    h5_paths:
        List of input HDF5 paths.

    out_dir:
        Output folder. Files are saved with the same filename as the input.

    n_images:
        Number of samples/images to keep from axis 0.

    image_key:
        Dataset used to infer the sample count. Usually "imgs".

    block_size:
        Number of samples copied per write block. Lower if RAM is tight,
        higher if storage is fast.

    overwrite:
        Whether to overwrite existing output files.

    Returns
    -------
    list[pathlib.Path]
        Paths to written subset files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_images = int(n_images)
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    written_paths: list[Path] = []

    for src_path_raw in h5_paths:
        src_path = Path(src_path_raw)
        if not src_path.exists():
            raise FileNotFoundError(f"Input H5 does not exist: {src_path}")

        dst_path = out_dir / src_path.name
        tmp_path = dst_path.with_name(dst_path.name + tmp_suffix)

        if src_path.resolve() == dst_path.resolve():
            raise ValueError(
                f"Refusing to overwrite source in-place: {src_path}"
            )

        if dst_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {dst_path}. "
                "Use overwrite=True to replace it."
            )

        if tmp_path.exists():
            tmp_path.unlink()

        with h5py.File(src_path, "r", libver="latest", swmr=True) as src:
            if image_key not in src:
                raise KeyError(
                    f"Cannot infer sample axis: missing dataset '{image_key}' "
                    f"in {src_path}"
                )

            n_source_samples = int(src[image_key].shape[0])
            n_keep = min(n_images, n_source_samples)

            with h5py.File(tmp_path, "w", libver="latest") as dst:
                _copy_group_subset(
                    src,
                    dst,
                    n_source_samples=n_source_samples,
                    n_keep=n_keep,
                    block_size=block_size,
                )

                # Update root-level bookkeeping attrs if present.
                dst.attrs["subset_source_file"] = str(src_path)
                dst.attrs["subset_n_source_samples"] = int(n_source_samples)
                dst.attrs["subset_n_kept_samples"] = int(n_keep)

                if "written" in dst.attrs:
                    dst.attrs.modify("written", int(n_keep))
                else:
                    dst.attrs["written"] = int(n_keep)

                dst.flush()

        if dst_path.exists() and overwrite:
            dst_path.unlink()

        tmp_path.replace(dst_path)
        written_paths.append(dst_path)

        print(
            f"[zenodo subset] {src_path.name}: "
            f"kept {n_keep}/{n_source_samples} samples -> {dst_path}"
        )

    return written_paths
