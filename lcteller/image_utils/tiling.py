# tiling.py

from __future__ import annotations
from typing import Callable, Optional, Tuple, Iterable
import numpy as np
import torch
import torch.nn.functional as F


def iter_sliding_windows(H: int, W: int, tile: int, overlap: int):
    stride = tile - overlap
    assert stride > 0, "overlap must be smaller than tile"

    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)

    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)

    for y0 in ys:
        y1 = y0 + tile
        for x0 in xs:
            x1 = x0 + tile
            yield (y0, y1, x0, x1)


def sliding_window_apply_torch(
    image: torch.Tensor,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    tile: int = 512,
    overlap: int = 128,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    out_channels: Optional[int] = None,
) -> torch.Tensor:
    """
    Core version. Works on torch only.

    image: [C, H, W]
    apply_fn: [1, C, tile, tile] -> [1, Cout, tile, tile]
    returns: [Cout, H, W]
    """
    assert image.dim() == 3
    C, H, W = image.shape
    device = image.device
    dtype = image.dtype

    acc = None
    acc_w = None

    for (y0, y1, x0, x1) in iter_sliding_windows(H, W, tile, overlap):
        crop = image[:, y0:y1, x0:x1]
        th, tw = crop.shape[1], crop.shape[2]

        py = tile - th
        px = tile - tw
        if py > 0 or px > 0:
            crop = F.pad(crop, (0, px, 0, py),
                         mode=pad_mode if pad_mode != "constant" else "constant",
                         value=pad_value if pad_mode == "constant" else 0.0)

        inp = crop.unsqueeze(0)  # [1, C, tile, tile]
        out = apply_fn(inp)      # [1, Cout, tile, tile]

        if py > 0 or px > 0:
            out = out[..., :th, :tw]

        if acc is None:
            Cout = out.shape[1] if out_channels is None else out_channels
            acc = torch.zeros((1, Cout, H, W), device=device, dtype=out.dtype)
            acc_w = torch.zeros((1, 1, H, W), device=device, dtype=out.dtype)

        acc[..., y0:y1, x0:x1] += out
        acc_w[..., y0:y1, x0:x1] += 1.0

    acc_w = torch.clamp(acc_w, min=1e-6)
    out_full = acc / acc_w
    return out_full.squeeze(0)


def sliding_window_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile: int = 512,
    overlap: int = 128,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    amp: bool = False,
) -> torch.Tensor:
    def _apply(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if amp:
                with torch.autocast(device_type=image.device.type, dtype=torch.float16):
                    return model(x)
            else:
                return model(x)

    return sliding_window_apply_torch(
        image=image,
        apply_fn=_apply,
        tile=tile,
        overlap=overlap,
        pad_mode=pad_mode,
        pad_value=pad_value,
    )


def sliding_window_apply_numpy(
    image: np.ndarray,
    apply_fn: Callable[[np.ndarray], np.ndarray],
    tile: int = 512,
    overlap: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Numpy front-end.
    image: [C, H, W] numpy
    apply_fn: expects numpy [1, C, tile, tile] -> numpy [1, Cout, tile, tile]
              (we wrap it under the hood)
    """
    # to torch
    t_img = torch.from_numpy(image).to(device=device, dtype=torch.float32)

    def _apply_torch(x: torch.Tensor) -> torch.Tensor:
        # x: torch [1, C, tile, tile] -> make numpy, call user's fn, back to torch
        x_np = x.detach().cpu().numpy()
        y_np = apply_fn(x_np)
        y_t = torch.from_numpy(y_np).to(device=x.device, dtype=torch.float32)
        return y_t

    out_t = sliding_window_apply_torch(
        image=t_img,
        apply_fn=_apply_torch,
        tile=tile,
        overlap=overlap,
    )
    return out_t.detach().cpu().numpy()

