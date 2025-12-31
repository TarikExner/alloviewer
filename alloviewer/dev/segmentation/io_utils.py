import signal
import h5py
import os
import numpy as np
import cv2
import tifffile as tiff
from typing import Tuple, Optional, Sequence, Any, Dict

def scale_to_10bit(
    img: np.ndarray,                # (C, S, S) float32
    mode: str = "clip01",           # 'clip01' | 'minmax' | 'percentile'
    percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Scale float image to uint16 using 10-bit range (0..1023)."""
    x = np.nan_to_num(img, copy=True)
    C = x.shape[0]
    out = np.empty_like(x, dtype=np.uint16)

    if mode == "clip01":
        x = np.clip(x, 0.0, 1.0)
        return np.rint(x * 1023.0).astype(np.uint16)

    for c in range(C):
        xc = x[c]
        if mode == "minmax":
            lo, hi = float(np.min(xc)), float(np.max(xc))
        elif mode == "percentile":
            lo, hi = np.percentile(xc, [percentiles[0], percentiles[1]])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            out[c] = 0
            continue

        yc = (xc - lo) / (hi - lo)
        yc = np.clip(yc, 0.0, 1.0)
        out[c] = np.rint(yc * 1023.0).astype(np.uint16)

    return out


def imwrite_tiff_cv2(path: str, arr: np.ndarray) -> None:
    """
    Write a uint16 TIFF with OpenCV.

    arr may be:
      - (H, W) uint16
      - (H, W, 3) uint16
    """
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    ok = cv2.imwrite(path, arr, params)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {path}")

def imwrite_tiff_tifffile(    path: str,
    arr: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    force_rgb: bool = True,
) -> None:
    """
    Write a uint16 TIFF with compression disabled and metadata for ImageJ/FIJI.

    Accepts:
      - (H, W) uint16                -> grayscale
      - (H, W, 3) uint16             -> RGB (channel-last)
      - (C, H, W) uint16             -> stack of C planes
        * if C==3 and force_rgb=True -> RGB (true-color)
        * else                       -> ImageJ stack (C,Y,X)

    Notes:
      - Values are assumed in 0..1023 (10-bit) but stored as uint16 unchanged.
      - Sets photometric tag and axes metadata so ImageJ shows correct colors.
      - Compression is disabled (compression=None).
    """
    if arr.dtype != np.uint16:
        raise TypeError("arr must be uint16")

    # Grayscale 2D
    if arr.ndim == 2:
        tiff.imwrite(
            path,
            arr,
            compression=None,
            photometric="minisblack",
            metadata={"axes": "YX"},
            software="export_h5_to_tiff",
        )
        return

    # Channel-last RGB already
    if arr.ndim == 3 and arr.shape[-1] == 3 and (force_rgb or arr.shape[0] != 3):
        tiff.imwrite(
            path,
            arr,  # (H,W,3) RGB
            compression=None,
            photometric="rgb",
            planarconfig="contig",
            metadata={"axes": "YXS"},
            software="export_h5_to_tiff",
        )
        return

    # Channel-first -> RGB if desired
    if arr.ndim == 3 and arr.shape[0] == 3 and force_rgb:
        rgb = np.moveaxis(arr, 0, -1)  # (H,W,3)
        tiff.imwrite(
            path,
            rgb,
            compression=None,
            photometric="rgb",
            planarconfig="contig",
            metadata={"axes": "YXS"},
            software="export_h5_to_tiff",
        )
        return

    # Channel-first stack (ImageJ-friendly)
    if arr.ndim == 3 and arr.shape[0] >= 1:
        md = {"axes": "CYX"}
        if channel_names is not None:
            md["channel_names"] = list(channel_names)
        tiff.imwrite(
            path,
            arr,  # (C,H,W)
            imagej=True,
            compression=None,
            photometric="minisblack",
            metadata=md,
            software="export_h5_to_tiff",
        )
        return

    raise ValueError(f"Unsupported shape: {arr.shape} (dtype={arr.dtype})")

def setup_stop_flag():
    """
    Install SIGINT/SIGTERM handlers and return a shared stop dict.
    """
    stop = {"flag": False}

    def _handle_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return stop

def jsonify(obj: Any) -> Any:
    """
    Make nested structures JSON-serializable (handle numpy scalars/arrays).
    """
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    return obj


def flush_safe_file(f: h5py.File):
    """
    Flush HDF5 file and underlying file descriptor (best effort).
    """
    f.flush()
    try:
        f.id.flush()
    except Exception:
        pass
    try:
        fd = f.id.get_vfd_handle()
        if fd is not None:
            os.fsync(fd)
    except Exception:
        pass


def init_or_validate_varT_in_file(
    f: h5py.File,
    *,
    new_file: bool,
    length: int,
    T0: int,
    C_img: int,
    C_tgt: int,
    H: int,
    W: int,
    compression: Optional[str],
    chunk_N: int,
    extra_attrs: Dict[str, Any],
    check_attrs: Dict[str, Any],
):
    """
    Shared HDF5 init/validation for variable-T datasets:

      imgs: [N, T, C_img, S, S]
      tgts: [N, T, C_tgt, S, S]
      inst: [N, T, S, S]
      meta: [N] vlen JSON

    Returns:
      d_imgs, d_tgts, d_inst, d_meta, written
    """
    vlen_str = h5py.string_dtype(encoding="utf-8")

    if new_file:
        attrs = {
            "version": 1,
            "length": int(length),
            "T": int(T0),
            "H": int(H),
            "W": int(W),
            "C_img": int(C_img),
            "C_tgt": int(C_tgt),
            "written": 0,
        }
        attrs.update(extra_attrs)
        f.attrs.update(attrs)

        f.create_dataset(
            "imgs",
            shape=(length, T0, C_img, H, W),
            maxshape=(length, None, C_img, H, W),
            dtype=np.float32,
            chunks=(int(chunk_N), T0, C_img, H, W),
            compression=compression,
        )
        f.create_dataset(
            "tgts",
            shape=(length, T0, C_tgt, H, W),
            maxshape=(length, None, C_tgt, H, W),
            dtype=np.float32,
            chunks=(int(chunk_N), T0, C_tgt, H, W),
            compression=compression,
        )
        f.create_dataset(
            "inst",
            shape=(length, T0, H, W),
            maxshape=(length, None, H, W),
            dtype=np.int32,
            chunks=(int(chunk_N), T0, H, W),
            compression=compression,
        )
        f.create_dataset(
            "meta",
            shape=(length,),
            dtype=vlen_str,
            chunks=(min(1024, length),),
        )
    else:
        # basic checks (length + any extra ones)
        assert int(f.attrs["length"]) == int(length), "length mismatch"
        for k, v in check_attrs.items():
            assert k in f.attrs, f"missing attr '{k}' in existing HDF5"
            assert f.attrs[k] == v, f"attr mismatch for '{k}'"

        file_T = int(f.attrs.get("T", T0))
        if T0 > file_T:
            # grow tile dim immediately to fit at least T0
            f["imgs"].resize((length, T0, C_img, H, W))
            f["tgts"].resize((length, T0, C_tgt, H, W))
            f["inst"].resize((length, T0, H, W))
            f.attrs.modify("T", int(T0))
        else:
            T0 = file_T

        if "written" not in f.attrs:
            f.attrs["written"] = 0

    d_imgs = f["imgs"]
    d_tgts = f["tgts"]
    d_inst = f["inst"]
    d_meta = f["meta"]
    written = int(f.attrs["written"])
    return d_imgs, d_tgts, d_inst, d_meta, written

