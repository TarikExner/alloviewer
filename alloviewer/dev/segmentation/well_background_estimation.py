import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path
from alloviewer.image_analysis.io import load_image
from IPython.display import clear_output
import os

BCG_IMAGES_FOLDERS = [
    "./ext_images/20251106_25065441_iPhone_XR_JPEG",
    "./ext_images/20251106_25722169_iPhone_XR_JPEG",
    "./ext_images/20251106_25722269_iPhone_XR_JPEG",
    "./ext_images/20251107_25065521_GooglePixel",
    "./ext_images/20251107_25722332_GooglePixel",
    "./ext_images/20251014_25719960",
    "./ext_images/20251014_25720084",
    "./ext_images/20251107_25065521",
    "./ext_images/20251107_25722332",
    "./ext_images/20251210_25066239_XM",
    "./ext_images/20251210_25066239_XM_+DTT",
    "./ext_images/20260120_26700788_LCT_RV",
    "./ext_images/20260120_26700790_LCT_RV",
    "./ext_images/20260121_26700790_LCT_RV_DTT",
]

def show_rectangle(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    ax=None,
    color: str = "red",
    linewidth: float = 2.0,
    clip_to_image: bool = True,
):
    """
    Show an image with a rectangle overlay.

    Parameters
    ----------
    img : np.ndarray
        Image in [H,W], [H,W,C], or [C,H,W].
    x : int
        Left coordinate of the rectangle.
    y : int
        Upper coordinate of the rectangle.
    width : int
        Rectangle width in pixels.
    height : int
        Rectangle height in pixels.
    ax : matplotlib.axes.Axes
        Matplotlib axis receiving the plot or formatting changes. Default is ``None``.
    color : str
        Color used to render the corresponding graphical element. Default is ``'red'``.
    linewidth : float
        Line width used to render the curve or outline. Default is ``2.0``.
    clip_to_image : bool
        If True, clamp the rectangle so it stays inside the image. Default is ``True``.

    Returns
    -------
    object
        Computed value of ``ax``.

    Raises
    ------
    ValueError
        Unsupported image shape: {arr.shape} width and height must be > 0
    """
    arr = np.asarray(img)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        disp = arr
        H, W = disp.shape
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            disp = arr[..., 0]
        else:
            disp = arr[..., :3]
        H, W = disp.shape[:2]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")

    if clip_to_image:
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        width = min(width, W - x)
        height = min(height, H - y)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(disp)

    rect = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)

    ax.set_title(f"x={x}, y={y}, width={width}, height={height}")
    #ax.axis("off")
    return ax



def extract_background_tiles(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    tile_size: int = 512,
    step: int = 128,
    interpolation: int = cv2.INTER_CUBIC,
    random_rotate_90: bool = True,
    rng: np.random.Generator | None = None,
    save_path: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Crop an image region into channel-first background tiles with optional rotation and saving.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.
    x : int
        Horizontal column coordinate in pixels.
    y : int
        Vertical row coordinate in pixels.
    width : int
        Width of the selected image region, in pixels.
    height : int
        Height of the selected image region, in pixels.
    tile_size : int
        Square tile side length in pixels. Default is ``512``.
    step : int
        Stride between adjacent tile origins, in pixels. Default is ``128``.
    interpolation : int
        OpenCV interpolation flag used for resizing. Default is ``cv2.INTER_CUBIC``.
    random_rotate_90 : bool
        Whether to randomly rotate each output tile by a multiple of 90 degrees. Default is
        ``True``.
    rng : np.random.Generator | None
        NumPy random-number generator used for sampling. Default is ``None``.
    save_path : str | Path | None
        Optional path at which to save the extracted tile array. Default is ``None``.
    verbose : bool
        Whether to emit progress information. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Computed value of ``out``.

    Raises
    ------
    ValueError
        Unsupported image shape: {arr.shape} width and height must be > 0
    """
    arr = np.asarray(img)

    # Normalize to CHW
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        pass
    elif arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    C, H, W = arr.shape

    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")

    # clip ROI to image bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    width = min(width, W - x)
    height = min(height, H - y)

    roi = arr[:, y:y+height, x:x+width]
    _, h, w = roi.shape

    def resize_tile(tile: np.ndarray) -> np.ndarray:
        """
        Resize each channel to the configured square tile size.

        Parameters
        ----------
        tile : numpy.ndarray
            Channel-first background tile to resize.

        Returns
        -------
        numpy.ndarray
            Computed value of ``out``.
        """
        out = np.empty((C, tile_size, tile_size), dtype=arr.dtype)
        for c in range(C):
            out[c] = cv2.resize(
                tile[c],
                (tile_size, tile_size),
                interpolation=interpolation,
            )
        return out

    def axis_positions(dim: int) -> list[int]:
        """
        Choose starting offsets for the sliding background-tile window.

        Parameters
        ----------
        dim : int
            Length of the spatial axis along which tile positions are generated.

        Returns
        -------
        list[int]
            Starting offsets for the sliding background-tile window.
        """
        extra = dim - tile_size
        if extra < step:
            return [0]
        return list(range(0, extra + 1, step))

    xs = axis_positions(w)
    ys = axis_positions(h)

    if verbose:
        print(f"Input image shape (CHW): {(C, H, W)}")
        print(f"Requested ROI: x={x}, y={y}, width={width}, height={height}")
        print(f"Effective ROI shape: (C={C}, H={h}, W={w})")
        print(f"x positions ({len(xs)}): {xs[:15]}{' ...' if len(xs) > 15 else ''}")
        print(f"y positions ({len(ys)}): {ys[:15]}{' ...' if len(ys) > 15 else ''}")
        print(f"Expected tile count: {len(xs) * len(ys)}")

    if rng is None:
        rng = np.random.default_rng()

    tiles = []
    for yy in ys:
        for xx in xs:
            tile = roi[:, yy:min(yy + tile_size, h), xx:min(xx + tile_size, w)]

            if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
                tile = resize_tile(tile)

            if random_rotate_90:
                k = int(rng.integers(0, 4))
                tile = np.rot90(tile, k=k, axes=(1, 2)).copy()

            tiles.append(tile)

    out = np.stack(tiles, axis=0)

    if verbose:
        print(f"Actual output shape: {out.shape}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, out)

    return out


def collect_image_paths(root_dirs, suffixes=(".tif", ".tiff", ".png", ".jpg", ".jpeg")):
    """
    Collect and sort paths with supported image suffixes below the given roots.

    Parameters
    ----------
    root_dirs : sequence of path-like
        Root folders recursively searched for images.
    suffixes : tuple
        Image filename suffixes included in the search. Default is ``('.tif', '.tiff',
        '.png', '.jpg', '.jpeg')``.

    Returns
    -------
    object
        And sort paths with supported image suffixes below the given roots.
    """
    paths = []
    for root in root_dirs:
        root = Path(root)
        for p in root.rglob("*"):
            if p.suffix.lower() in suffixes:
                paths.append(p)
    return sorted(paths)

def make_output_name(img_path: str | Path) -> str:
    """
    Build a NumPy output filename from the source folder and image stem.

    Parameters
    ----------
    img_path : str | Path
        Source image path.

    Returns
    -------
    str
        A NumPy output filename from the source folder and image stem.
    """
    p = Path(img_path)
    folder = p.parent.name
    stem = p.stem
    return f"{folder}_{stem}.npy"


def review_and_save_one_image(
    img_path,
    *,
    data_dir=None,
    out_dir="./test_images",
    x=0,
    y=0,
    width=520,
    height=1620,
    tile_size=512,
    step=20,
):
    """
    Review one image in a notebook:
    - show rectangle
    - type new coords if needed
    - accept/save, skip, or quit

    Parameters
    ----------
    img_path : str or os.PathLike
        Source image path.
    data_dir : str or os.PathLike
        Root directory containing the source image or FCS dataset. Default is ``None``.
    out_dir : str
        Directory in which generated outputs are stored. Default is ``'./test_images'``.
    x : int
        Horizontal column coordinate in pixels. Default is ``0``.
    y : int
        Vertical row coordinate in pixels. Default is ``0``.
    width : int
        Width of the selected image region, in pixels. Default is ``520``.
    height : int
        Height of the selected image region, in pixels. Default is ``1620``.
    tile_size : int
        Square tile side length in pixels. Default is ``512``.
    step : int
        Stride between adjacent tile origins, in pixels. Default is ``20``.

    Returns
    -------
    status : str
        "saved", "skipped", or "quit"
    rect : tuple[int, int, int, int]
        final rectangle
    """
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img, report = load_image(
        os.path.basename(img_path),
        base_dir = os.path.dirname(img_path),
        scale=True,
        as_chw=True,
    )

    while True:
        plt.figure(figsize=(8, 8))
        show_rectangle(img, x=x, y=y, width=width, height=height)
        plt.show()

        print(f"\nImage: {img_path}")
        print(f"Current rectangle: x={x}, y={y}, width={width}, height={height}")
        print("Commands:")
        print("  a                -> accept and save")
        print("  s                -> skip")
        print("  q                -> quit")
        print("  x y width height -> update rectangle")

        cmd = input("Enter command: ").strip()

        if cmd.lower() == "a":
            tiles = extract_background_tiles(
                img=img,
                x=x,
                y=y,
                width=width,
                height=height,
                tile_size=tile_size,
                step=step,
            )

            out_name = make_output_name(img_path)
            out_path = out_dir / out_name
            np.save(out_path, tiles)

            print(f"Saved: {out_path}")
            print(f"Tiles shape: {tiles.shape}")
            return "saved", (x, y, width, height)

        elif cmd.lower() == "s":
            return "skipped", (x, y, width, height)

        elif cmd.lower() == "q":
            return "quit", (x, y, width, height)

        else:
            parts = cmd.split()
            if len(parts) != 4:
                print("Invalid input. Use: a, s, q, or: x y width height")
                continue

            try:
                x, y, width, height = map(int, parts)
            except ValueError:
                print("Could not parse integers.")
                continue


def review_dataset_and_save(
    image_paths,
    *,
    data_dir=None,
    out_dir="background_tiles",
    x=0,
    y=0,
    width=520,
    height=1620,
    tile_size=512,
    step=128,
    reuse_last_rectangle=True,
    skip_existing=True,
):
    """
    Loop through images for manual review.

    Parameters
    ----------
    image_paths : sequence of path-like
        Paths to the images included in the operation.
    data_dir : str or os.PathLike
        Root directory containing the source image or FCS dataset. Default is ``None``.
    out_dir : str
        Directory in which generated outputs are stored. Default is ``'background_tiles'``.
    x : int
        Horizontal column coordinate in pixels. Default is ``0``.
    y : int
        Vertical row coordinate in pixels. Default is ``0``.
    width : int
        Width of the selected image region, in pixels. Default is ``520``.
    height : int
        Height of the selected image region, in pixels. Default is ``1620``.
    tile_size : int
        Square tile side length in pixels. Default is ``512``.
    step : int
        Stride between adjacent tile origins, in pixels. Default is ``128``.
    reuse_last_rectangle : bool
        Whether the previous image's rectangle is reused as the next selection. Default is
        ``True``.
    skip_existing : bool
        Whether to skip inputs that already have the required output files. Default is
        ``True``.

    Returns
    -------
    object
        Computed value of ``results``.

    Notes
    -----
    Images whose output file already exists are skipped automatically.
    They still count toward the displayed progress counter.
    """
    image_paths = list(image_paths)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    cur_x, cur_y, cur_w, cur_h = x, y, width, height

    for i, img_path in enumerate(image_paths, start=1):
        img_path = Path(img_path)

        out_name = make_output_name(img_path)
        out_path = out_dir / out_name

        print(f"\n===== [{i}/{len(image_paths)}] {img_path} =====")

        # Skip images that have already been processed.
        if skip_existing and out_path.exists():
            print(f"Already processed, skipping: {out_path}")

            results.append({
                "image_path": str(img_path),
                "output_path": str(out_path),
                "status": "already_exists",
                "x": None,
                "y": None,
                "width": None,
                "height": None,
            })

            clear_output(wait=True)
            continue

        status, rect = review_and_save_one_image(
            img_path,
            data_dir=data_dir,
            out_dir=out_dir,
            x=cur_x,
            y=cur_y,
            width=cur_w,
            height=cur_h,
            tile_size=tile_size,
            step=step,
        )

        results.append({
            "image_path": str(img_path),
            "output_path": str(out_path),
            "status": status,
            "x": rect[0],
            "y": rect[1],
            "width": rect[2],
            "height": rect[3],
        })

        if status == "quit":
            break

        if reuse_last_rectangle and status == "saved":
            cur_x, cur_y, cur_w, cur_h = rect

        clear_output(wait=True)

    return results

def collect_ext_image_files(
    ext_root="./ext_images",
    exclude_roots=("./human_annotations", "./experiment_readout_images"),
    suffixes=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
):
    """
    Collect all image files from subfolders in ext_root, excluding any subfolder
    whose name also exists in one of the exclude_roots.

    Parameters
    ----------
    ext_root : str
        Root of the external-image collection. Default is ``'./ext_images'``.
    exclude_roots : tuple
        Directory roots omitted from recursive image collection. Default is
        ``('./human_annotations', './experiment_readout_images')``.
    suffixes : tuple
        Image filename suffixes included in the search. Default is ``('.tif', '.tiff',
        '.png', '.jpg', '.jpeg')``.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.
    """
    ext_root = Path(ext_root)
    suffixes = tuple(s.lower() for s in suffixes)

    excluded_names = {
        p.name
        for ex_root in exclude_roots
        for p in Path(ex_root).iterdir()
        if p.is_dir()
    }

    files = [
        f
        for subfolder in ext_root.iterdir()
        if subfolder.is_dir() and subfolder.name not in excluded_names
        for f in subfolder.iterdir()
        if f.is_file() and f.suffix.lower() in suffixes
    ]

    return sorted(files)


