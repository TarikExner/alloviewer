import numpy as np
import torch
from torch.utils.data import Dataset

from . import simulate_image

def rand_choice(rng, val_or_range):
    if isinstance(val_or_range, (list, tuple)) and len(val_or_range) == 2:
        lo, hi = val_or_range
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(lo, hi + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return val_or_range

class SimCellsDataset(Dataset):
    """
    Returns:
      img_t:  float32 [3,H,W] in [0,1]
      tgt_t:  float32 [2,H,W]  (0=cell, 1=boundary)
      extras: dict with:
              - "instance_labels": int32 [H,W]
              - "meta": meta from sim
    """
    def __init__(
        self,
        length=10000,
        tile_size=512,
        n_cells=(80, 220),
        cell_diameter=(10, 28),
        frac_positive=(0.2, 0.8),
        blur_sigma=(0.6, 1.8),
        background_level=(0.0, 0.04),
        color_jitter=(0.05, 0.12),
        photon_level=(1500, 4000),
        boundary_width=2,
        aug_flip=True,
        aug_rot90=True,
        aug_gamma=(0.90, 1.12),
        rng_seed=123,
        sim_fn=None
    ):
        if sim_fn is None:
            self.sim_fn = simulate_image
        else:
            self.sim_fn = sim_fn
        self.length = int(length)
        self.tile_size = int(tile_size)
        self.n_cells = n_cells
        self.cell_diameter = cell_diameter
        self.frac_positive = frac_positive
        self.blur_sigma = blur_sigma
        self.background_level = background_level
        self.color_jitter = color_jitter
        self.photon_level = photon_level
        self.boundary_width = int(boundary_width)
        self.aug_flip = aug_flip
        self.aug_rot90 = aug_rot90
        self.aug_gamma = aug_gamma
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return self.length

    def _apply_aug(self, img, cell, bound, inst):
        if self.aug_flip and self.rng.random() < 0.5:
            img   = np.flip(img, axis=1)
            cell  = np.flip(cell, axis=1)
            bound = np.flip(bound, axis=1)
            inst  = np.flip(inst, axis=1)
        if self.aug_flip and self.rng.random() < 0.5:
            img   = np.flip(img, axis=0)
            cell  = np.flip(cell, axis=0)
            bound = np.flip(bound, axis=0)
            inst  = np.flip(inst, axis=0)

        if self.aug_rot90:
            k = int(self.rng.integers(0, 4))
            if k:
                img   = np.rot90(img,   k, axes=(0, 1))
                cell  = np.rot90(cell,  k, axes=(0, 1))
                bound = np.rot90(bound, k, axes=(0, 1))
                inst  = np.rot90(inst,  k, axes=(0, 1))

        if isinstance(self.aug_gamma, (list, tuple)) and len(self.aug_gamma) == 2:
            g = float(self.rng.uniform(self.aug_gamma[0], self.aug_gamma[1]))
            img = np.clip(img, 1e-4, 1.0) ** g
            img = np.clip(img, 0.0, 1.0)

        return img, cell, bound, inst

    def __getitem__(self, idx):
        N   = self.tile_size
        nC  = rand_choice(self.rng, self.n_cells)
        dia = rand_choice(self.rng, self.cell_diameter)
        fp  = rand_choice(self.rng, self.frac_positive)
        blr = rand_choice(self.rng, self.blur_sigma)
        bg  = rand_choice(self.rng, self.background_level)
        cj  = rand_choice(self.rng, self.color_jitter)
        ph  = rand_choice(self.rng, self.photon_level)
        seed = int(self.rng.integers(0, 2**31 - 1))

        assert self.sim_fn is not None

        img, meta, targets = self.sim_fn(
            N=N,
            n_cells=nC,
            cell_diameter=dia,
            frac_positive=fp,
            background_level=bg,
            color_jitter=cj,
            blur_sigma=blr,
            photon_level=ph,
            seed=seed,
            return_targets=True,
            boundary_width=self.boundary_width,
        )
        cell  = targets["cell_mask"].astype(np.float32)     # H,W
        bound = targets["boundary"].astype(np.float32)      # H,W
        inst  = targets["instance_labels"].astype(np.int32) # H,W

        # aug (kept in sync)
        img, cell, bound, inst = self._apply_aug(img, cell, bound, inst)

        # pack target [2,H,W]
        tgt = np.stack([cell, bound], axis=0).astype(np.float32)

        # to tensors
        img_c = np.ascontiguousarray(np.transpose(img, (2, 0, 1)), dtype=np.float32)
        tgt_c = np.ascontiguousarray(tgt, dtype=np.float32)
        inst_c = np.ascontiguousarray(inst, dtype=np.int32)

        img_t = torch.from_numpy(img_c)
        tgt_t = torch.from_numpy(tgt_c)
        extras = {"instance_labels": torch.from_numpy(inst_c), "meta": meta}
        return img_t, tgt_t, extras

