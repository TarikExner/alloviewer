import os
import json
import signal
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import torch

from .config import default_camera, default_scene

# --- config fallbacks ---
from . import SimCellsDataset

# --------------------- collate ---------------------

def collate_no_meta(batch):
    imgs, tgts, exs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)   # works for [3,H,W] AND for [1,3,H,W]
    tgts = torch.stack(tgts, dim=0)
    inst = torch.stack([e["instance_labels"] for e in exs], dim=0)
    metas = [e["meta"] for e in exs]
    return imgs, tgts, {"instance_labels": inst, "meta": metas}

def create_dataset_h5(
    out_path: str,
    length: int,
    mode: str = "crop_well_resize",
    target: int = 512,
    rng_seed: int = 187,
    gen_batch_size: int = 128,
    num_workers_gen: int = 16,
    compression: Optional[str] = "lzf",
    flush_every: int = 8,
    resume: bool = True,
    camera_cfg=None,
    scene_cfg=None,
):
    """
    Pre-allocate a single HDF5 and append EXACT tensors from SimCellsDataset.__getitem__:

      /imgs: float32 [N, 1, 3, S, S]
      /tgts: float32 [N, 1, C, S, S]
      /inst: int32   [N, 1, S, S]
      /meta: vlen JSON (one per sample)
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if camera_cfg is None:
        camera_cfg = default_camera()
    if scene_cfg is None:
        scene_cfg = default_scene()

    # --- graceful stop on SIGINT/SIGTERM ---
    stop = {"flag": False}
    def _handle_signal(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- dataset & dataloader (generation on CPU) ---
    ds = SimCellsDataset(
        length=length, mode=mode, target=target,
        rng_seed=rng_seed, camera_cfg=camera_cfg, scene_cfg=scene_cfg,
    )
    dl = DataLoader(
        ds,
        batch_size=gen_batch_size,
        shuffle=False,
        num_workers=int(num_workers_gen),
        pin_memory=False,
        persistent_workers=(num_workers_gen > 0),
        prefetch_factor=4,
        drop_last=False,
        collate_fn=collate_no_meta,   # must stack -> gives [B,1,3,S,S]
    )

    # --- peek first batch ---
    it = iter(dl)
    try:
        first_imgs, first_tgts, first_extras = next(it)
    except StopIteration:
        raise RuntimeError("Empty dataset (length=0).")

    # shapes NOW:
    # first_imgs: [B0, 1, 3, S, S]
    # first_tgts: [B0, 1, C_tgt, S, S]
    B0, T0, C_img, S, _ = first_imgs.shape
    _, _, C_tgt, _, _ = first_tgts.shape

    vlen_str = h5py.special_dtype(vlen=str)
    new_file = (not os.path.exists(out_path))
    with h5py.File(out_path, "a", libver="latest") as f:
        if new_file:
            f.attrs.update({
                "version": 1,
                "length": int(length),
                "mode": mode,
                "target": int(target),
                "rng_seed": int(rng_seed),
                "T": int(T0),          # 1 for now
                "C_img": int(C_img),
                "C_tgt": int(C_tgt),
                "written": 0,
            })
            # create 5D datasets
            f.create_dataset(
                "imgs",
                shape=(length, T0, C_img, S, S),
                dtype=np.float32,
                chunks=(gen_batch_size, T0, C_img, S, S),
                compression=compression,
            )
            f.create_dataset(
                "tgts",
                shape=(length, T0, C_tgt, S, S),
                dtype=np.float32,
                chunks=(gen_batch_size, T0, C_tgt, S, S),
                compression=compression,
            )
            f.create_dataset(
                "inst",
                shape=(length, T0, S, S),
                dtype=np.int32,
                chunks=(gen_batch_size, T0, S, S),
                compression=compression,
            )
            f.create_dataset(
                "meta",
                shape=(length,),
                dtype=vlen_str,
                chunks=(min(1024, length),),
            )
        else:
            # sanity checks against existing file
            assert int(f.attrs["length"]) == int(length), "length mismatch"
            assert f.attrs["mode"] == mode, "mode mismatch"
            assert int(f.attrs["target"]) == int(target), "target mismatch"
            assert int(f.attrs.get("T", 1)) == int(T0), "tile-dim mismatch"
            assert f["imgs"].shape[1:] == (T0, C_img, S, S), "image shape mismatch"
            assert f["tgts"].shape[1:] == (T0, C_tgt, S, S), "target shape mismatch"
            assert f["inst"].shape[1:] == (T0, S, S), "inst shape mismatch"
            if "written" not in f.attrs:
                f.attrs["written"] = 0

        d_imgs, d_tgts, d_inst, d_meta = f["imgs"], f["tgts"], f["inst"], f["meta"]

        # --- resume offset ---
        written = int(f.attrs["written"])
        if written >= length:
            print(f"[export] already complete: {written}/{length}")
            return out_path

        pbar = tqdm(total=length, initial=written, desc="export h5", dynamic_ncols=True)
        last_flush = 0

        def _flush_safe():
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

        def _jsonify(obj):
            if isinstance(obj, (np.generic,)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            return obj

        def _write_slice(imgs, tgts, inst, metas):
            nonlocal written, last_flush
            n = imgs.shape[0]   # B
            if n <= 0:
                return 0
            end = min(length, written + n)
            take = end - written
            if take <= 0:
                return 0

            d_imgs[written:end] = imgs[:take].detach().cpu().numpy().astype(np.float32)
            d_tgts[written:end] = tgts[:take].detach().cpu().numpy().astype(np.float32)
            d_inst[written:end] = inst[:take].detach().cpu().numpy().astype(np.int32)
            d_meta[written:end] = [
                json.dumps(_jsonify(m), separators=(",", ":"))
                for m in metas[:take]
            ]
            written = end
            f.attrs.modify("written", int(written))
            last_flush += 1
            if (last_flush % int(max(1, flush_every))) == 0:
                _flush_safe()
            pbar.update(take)
            return take

        # --- fast-forward if resuming ---
        to_skip = written
        if resume and to_skip > 0:
            skip_batches = to_skip // gen_batch_size
            for _ in range(skip_batches):
                try:
                    next(it)
                except StopIteration:
                    break
            skip_left = to_skip % gen_batch_size
            if skip_left > 0:
                try:
                    b_img, b_tgt, b_ex = next(it)
                    if b_img.shape[0] > skip_left:
                        _write_slice(
                            b_img[skip_left:], b_tgt[skip_left:],
                            b_ex["instance_labels"][skip_left:],
                            b_ex["meta"][skip_left:],
                        )
                except StopIteration:
                    pass
        else:
            # write first batch we already have
            _write_slice(
                first_imgs, first_tgts,
                first_extras["instance_labels"],
                first_extras["meta"],
            )

        # --- main loop ---
        for imgs, tgts, extras in it:
            if stop["flag"]:
                break
            _write_slice(imgs, tgts, extras["instance_labels"], extras["meta"])
            if written >= length:
                break

        _flush_safe()
        pbar.close()
        print(f"done: {written}/{length} → {out_path}")
        return out_path

