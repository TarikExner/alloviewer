
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display
from scipy import ndimage as ndi
from skimage import filters, morphology

from alloviewer.image_analysis.io import load_image


LABEL_IGNORE = np.uint8(0)
LABEL_BACKGROUND = np.uint8(1)
LABEL_FOREGROUND = np.uint8(2)

LABEL_NAMES = {
    int(LABEL_IGNORE): "ignore",
    int(LABEL_BACKGROUND): "background",
    int(LABEL_FOREGROUND): "foreground",
}


def collect_annotation_paths(
    root_dirs: Sequence[str | Path],
    suffixes: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> list[Path]:
    """Collect image files below the requested reference-image folders."""
    suffixes = tuple(s.lower() for s in suffixes)
    paths: list[Path] = []

    for root in dict.fromkeys(Path(p) for p in root_dirs):
        if not root.exists():
            continue

        paths.extend(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes
        )

    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("No annotation images were found.")

    return paths


def infer_device_label(path: str | Path) -> str:
    """Infer the camera/device group from a file path."""
    text = str(path).lower()

    if "mono_rgb" in text or "mono_real" in text:
        return "monochrome_real"
    if "iphone" in text:
        return "iphone"
    if "googlepixel" in text or "pixel" in text:
        return "googlepixel"
    return "microscope"


def make_annotation_stem(path: str | Path) -> str:
    """Create a stable output name from parent-folder and file stem."""
    path = Path(path)
    return f"{path.parent.name}_{path.stem}"


def _signal_from_rgb(image: np.ndarray, mode: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)

    if mode == "max":
        signal = image.max(axis=-1)
    elif mode == "luma":
        signal = (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        )
    elif mode == "green":
        signal = image[..., 1]
    elif mode == "red_green_max":
        signal = np.maximum(image[..., 0], image[..., 1])
    else:
        raise ValueError(
            "signal_mode must be one of: "
            "'max', 'luma', 'green', 'red_green_max'"
        )

    return np.asarray(signal, dtype=np.float32)



def _validated_local_block_size(value: int) -> int:
    """Return an odd local-threshold block size of at least 3."""
    block_size = max(3, int(value))
    if block_size % 2 == 0:
        block_size += 1
    return block_size


def propose_region_mask(
    image: np.ndarray,
    *,
    signal_mode: str = "red_green_max",
    polarity: str = "bright",
    background_sigma: float = 20.0,
    threshold_mode: str = "otsu",
    foreground_quantile: float = 0.92,
    local_block_size: int = 51,
    local_offset: float = 0.0,
    use_extreme_rescue: bool = True,
    extreme_quantile: float = 0.985,
    min_object_size: int = 20,
    max_object_size: int = 0,
    close_radius: int = 1,
    fill_holes: bool = True,
    max_hole_area: int = 500,
) -> np.ndarray:
    """
    Create an editable three-class proposal.

    Labels
    ------
    0: ignore
    1: background
    2: foreground

    Threshold modes
    ---------------
    otsu:
        One global Otsu threshold on the background-corrected signal.

    quantile:
        One global quantile threshold on the background-corrected signal.

    local:
        A spatially varying Gaussian local threshold on the
        background-corrected signal. ``local_block_size`` must be larger than
        the objects of interest. Even values are increased by one.

    Notes
    -----
    ``use_extreme_rescue`` adds very bright pixels for bright-cell mode, very
    dark pixels for dark-cell mode, or both tails for two-sided mode. This can
    recover cell centers that disappear after background subtraction.

    Hole filling is area-limited. This avoids filling large gaps inside cell
    clusters while still filling small missed cell centers.
    """
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    signal = _signal_from_rgb(image, signal_mode)

    sigma = max(0.0, float(background_sigma))
    if sigma > 0:
        background = cv2.GaussianBlur(
            signal,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    else:
        background = np.zeros_like(signal)

    if polarity == "bright":
        corrected = signal - background
    elif polarity == "dark":
        corrected = background - signal
    elif polarity == "both":
        corrected = np.abs(signal - background)
    else:
        raise ValueError("polarity must be 'bright', 'dark', or 'both'")

    corrected = np.asarray(corrected, dtype=np.float32)
    finite = np.isfinite(corrected) & np.isfinite(signal)

    if not finite.any():
        return np.full(signal.shape, LABEL_IGNORE, dtype=np.uint8)

    values = corrected[finite]
    threshold_mode = str(threshold_mode).lower()

    if threshold_mode == "otsu":
        if np.allclose(values.min(), values.max()):
            threshold = float(values.max())
        else:
            threshold = float(filters.threshold_otsu(values))

        foreground = finite & (corrected >= threshold)

    elif threshold_mode == "quantile":
        q = float(np.clip(foreground_quantile, 0.0, 1.0))
        threshold = float(np.quantile(values, q))
        foreground = finite & (corrected >= threshold)

    elif threshold_mode == "local":
        block_size = _validated_local_block_size(local_block_size)

        local_threshold = filters.threshold_local(
            corrected,
            block_size=block_size,
            method="gaussian",
            offset=float(local_offset),
            mode="reflect",
        )

        # Requiring a positive background-corrected response prevents broad,
        # nearly flat background regions from passing a negative local threshold.
        foreground = (
            finite
            & (corrected > 0.0)
            & (corrected >= local_threshold)
        )

    else:
        raise ValueError(
            "threshold_mode must be one of: 'otsu', 'quantile', 'local'"
        )

    if use_extreme_rescue:
        q_extreme = float(np.clip(extreme_quantile, 0.500001, 0.999999))
        signal_values = signal[finite]

        if polarity == "bright":
            high = float(np.quantile(signal_values, q_extreme))
            rescue = finite & (signal >= high)

        elif polarity == "dark":
            low = float(np.quantile(signal_values, 1.0 - q_extreme))
            rescue = finite & (signal <= low)

        else:
            low = float(np.quantile(signal_values, 1.0 - q_extreme))
            high = float(np.quantile(signal_values, q_extreme))
            rescue = finite & ((signal <= low) | (signal >= high))

        foreground |= rescue

    if close_radius > 0:
        foreground = morphology.binary_closing(
            foreground,
            morphology.disk(int(close_radius)),
        )

    if fill_holes and max_hole_area > 0:
        foreground = morphology.remove_small_holes(
            foreground,
            area_threshold=int(max_hole_area),
        )

    if min_object_size > 0:
        foreground = morphology.remove_small_objects(
            foreground,
            min_size=int(min_object_size),
        )

    # Remove connected components larger than the selected maximum area.
    # max_object_size=0 disables this filter.
    if max_object_size > 0 and foreground.any():
        component_labels = measure.label(
            foreground,
            connectivity=2,
        )
        component_areas = np.bincount(component_labels.ravel())
        keep_component = component_areas <= int(max_object_size)
        keep_component[0] = False
        foreground = keep_component[component_labels]

    labels = np.full(signal.shape, LABEL_BACKGROUND, dtype=np.uint8)
    labels[~finite] = LABEL_IGNORE
    labels[foreground] = LABEL_FOREGROUND
    return labels


def derive_local_background(
    labels: np.ndarray,
    *,
    inner_radius: int = 3,
    outer_radius: int = 15,
) -> np.ndarray:
    """
    Derive a local-background ring from reviewed labels.

    Only pixels explicitly marked as background are retained.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    foreground = labels == LABEL_FOREGROUND
    confirmed_background = labels == LABEL_BACKGROUND

    inner_radius = max(0, int(inner_radius))
    outer_radius = max(inner_radius + 1, int(outer_radius))

    inner = ndi.binary_dilation(foreground, iterations=inner_radius)
    outer = ndi.binary_dilation(foreground, iterations=outer_radius)

    return (outer & ~inner & confirmed_background).astype(bool)


def _resize_for_display(
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float, float]:
    """Return a display copy and original/display coordinate scale factors."""
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))

    if scale == 1.0:
        shown = image
    else:
        shown = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    sy = height / shown.shape[0]
    sx = width / shown.shape[1]
    return shown, sy, sx


def _labels_to_rgba(labels: np.ndarray, opacity: float) -> np.ndarray:
    """Convert labels to an RGBA overlay."""
    labels = np.asarray(labels, dtype=np.uint8)
    rgba = np.zeros(labels.shape + (4,), dtype=np.float32)

    # ignore: blue
    rgba[labels == LABEL_IGNORE, :3] = (0.1, 0.35, 1.0)
    # background: black with low alpha
    rgba[labels == LABEL_BACKGROUND, :3] = (0.0, 0.0, 0.0)
    # foreground: red
    rgba[labels == LABEL_FOREGROUND, :3] = (1.0, 0.1, 0.1)

    alpha = float(np.clip(opacity, 0.0, 1.0))
    rgba[labels == LABEL_IGNORE, 3] = alpha
    rgba[labels == LABEL_BACKGROUND, 3] = 0.08 * alpha
    rgba[labels == LABEL_FOREGROUND, 3] = alpha
    return rgba


class NotebookMaskReviewer:
    """
    Interactive Jupyter reviewer for foreground/background masks.

    Requirements
    ------------
    Run ``%matplotlib widget`` before creating the reviewer. This requires
    ``ipympl`` and ``ipywidgets``.

    Saved files
    -----------
    <stem>_regions.npy
        uint8 label image with values 0=ignore, 1=background, 2=foreground.

    <stem>_regions.json
        Source path, device, class counts, proposal settings, and loader report.

    <stem>_preview.png
        Optional visual check of the saved labels.
    """

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        out_dir: str | Path = "./region_annotations",
        start_index: int = 0,
        max_display_side: int = 1200,
        canvas_width_px: int = 1000,
        save_preview: bool = True,
        load_existing: bool = True,
    ):
        if not image_paths:
            raise ValueError("image_paths is empty")

        self.image_paths = [Path(p) for p in image_paths]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.index = int(np.clip(start_index, 0, len(self.image_paths) - 1))
        self.max_display_side = int(max_display_side)
        self.save_preview = bool(save_preview)
        self.load_existing = bool(load_existing)

        self.image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.display_labels: Optional[np.ndarray] = None
        self.load_report = None
        self.scale_y = 1.0
        self.scale_x = 1.0

        self.max_display_side = int(max_display_side)
        self.canvas_width_px = int(canvas_width_px)

        self._painting = False
        self._last_point: Optional[tuple[int, int]] = None

        self._build_widgets()
        self._build_figure()
        self._connect_events()
        self._load_current()

    @classmethod
    def from_folders(
        cls,
        folders: Sequence[str | Path],
        **kwargs,
    ) -> "NotebookMaskReviewer":
        paths = collect_annotation_paths(folders)
        return cls(paths, **kwargs)

    def _build_widgets(self) -> None:
        self.signal_mode = widgets.Dropdown(
            options=["red_green_max", "max", "green", "luma"],
            value="red_green_max",
            description="Signal",
        )
        self.polarity = widgets.Dropdown(
            options=["bright", "dark", "both"],
            value="bright",
            description="Cells",
        )
        self.threshold_mode = widgets.Dropdown(
            options=[
                ("Otsu", "otsu"),
                ("Quantile", "quantile"),
                ("Local", "local"),
            ],
            value="otsu",
            description="Threshold",
        )
        self.foreground_quantile = widgets.FloatSlider(
            value=0.92,
            min=0.70,
            max=0.995,
            step=0.005,
            description="Quantile",
            readout_format=".3f",
            continuous_update=False,
        )
        self.local_block_size = widgets.IntSlider(
            value=51,
            min=9,
            max=301,
            step=2,
            description="Local block",
            continuous_update=False,
        )
        self.local_offset = widgets.FloatSlider(
            value=0.0,
            min=-0.10,
            max=0.10,
            step=0.002,
            description="Local offset",
            readout_format=".3f",
            continuous_update=False,
        )
        self.use_extreme_rescue = widgets.Checkbox(
            value=True,
            description="Rescue extreme pixels",
            indent=False,
        )
        self.extreme_quantile = widgets.FloatSlider(
            value=0.985,
            min=0.90,
            max=0.999,
            step=0.001,
            description="Extreme q",
            readout_format=".3f",
            continuous_update=False,
        )
        self.background_sigma = widgets.FloatSlider(
            value=20.0,
            min=0.0,
            max=100.0,
            step=1.0,
            description="BG sigma",
            continuous_update=False,
        )
        self.min_object_size = widgets.IntSlider(
            value=20,
            min=0,
            max=1000,
            step=5,
            description="Min area",
            continuous_update=False,
        )
        self.max_object_size = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100_000_000,
            step=100,
            description="Max area",
            tooltip=(
                "Maximum connected foreground component area in original "
                "image pixels. Use 0 to disable."
            ),
            layout=widgets.Layout(width="260px"),
        )
        self.close_radius = widgets.IntSlider(
            value=1,
            min=0,
            max=10,
            step=1,
            description="Close",
            continuous_update=False,
        )
        self.fill_holes = widgets.Checkbox(
            value=True,
            description="Fill small holes",
            indent=False,
        )
        self.max_hole_area = widgets.IntSlider(
            value=500,
            min=0,
            max=5000,
            step=25,
            description="Max hole",
            continuous_update=False,
        )
        self.opacity = widgets.FloatSlider(
            value=0.45,
            min=0.0,
            max=0.9,
            step=0.05,
            description="Opacity",
            continuous_update=False,
        )
        self.brush_mode = widgets.ToggleButtons(
            options=[
                ("Foreground", int(LABEL_FOREGROUND)),
                ("Background", int(LABEL_BACKGROUND)),
                ("Ignore", int(LABEL_IGNORE)),
            ],
            value=int(LABEL_FOREGROUND),
            description="Brush",
        )
        self.brush_radius = widgets.IntSlider(
            value=12,
            min=1,
            max=200,
            step=1,
            description="Radius",
            continuous_update=False,
        )

        self.propose_button = widgets.Button(description="Build proposal")
        self.reset_background_button = widgets.Button(description="All background")
        self.save_button = widgets.Button(description="Save")
        self.save_next_button = widgets.Button(description="Save + next")
        self.previous_button = widgets.Button(description="Previous")
        self.next_button = widgets.Button(description="Skip / next")
        self.reload_button = widgets.Button(description="Reload saved")

        self.status = widgets.HTML()
        self.progress = widgets.HTML()
        self.app = None

        self.propose_button.on_click(self._on_propose)
        self.reset_background_button.on_click(self._on_reset_background)
        self.save_button.on_click(self._on_save)
        self.save_next_button.on_click(self._on_save_next)
        self.previous_button.on_click(self._on_previous)
        self.next_button.on_click(self._on_next)
        self.reload_button.on_click(self._on_reload)
        self.opacity.observe(self._on_opacity_change, names="value")
        self.threshold_mode.observe(
            self._on_threshold_mode_change,
            names="value",
        )
        self.use_extreme_rescue.observe(
            self._on_extreme_rescue_change,
            names="value",
        )
        self.fill_holes.observe(
            self._on_fill_holes_change,
            names="value",
        )

        self._update_proposal_control_state()

        proposal_box = widgets.VBox([
            widgets.HTML("<b>Automatic proposal</b>"),
            widgets.HBox([
                self.signal_mode,
                self.polarity,
                self.threshold_mode,
            ]),
            self.foreground_quantile,
            self.local_block_size,
            self.local_offset,
            self.background_sigma,
            self.use_extreme_rescue,
            self.extreme_quantile,
            self.min_object_size,
            self.max_object_size,
            widgets.HTML(
                "<small>Max area is connected-component area; 0 disables it.</small>"
            ),
            self.close_radius,
            self.fill_holes,
            self.max_hole_area,
            widgets.HBox([
                self.propose_button,
                self.reset_background_button,
            ]),
        ])

        paint_box = widgets.VBox([
            widgets.HTML("<b>Manual correction</b>"),
            self.brush_mode,
            self.brush_radius,
            self.opacity,
            widgets.HTML(
                "Drag on the image to paint. "
                "Labels: 0 ignore, 1 background, 2 foreground."
            ),
        ])

        action_box = widgets.VBox([
            widgets.HTML("<b>Files</b>"),
            widgets.HBox([
                self.previous_button,
                self.next_button,
                self.reload_button,
            ]),
            widgets.HBox([self.save_button, self.save_next_button]),
            self.progress,
            self.status,
        ])

        self.controls = widgets.HBox(
            [proposal_box, paint_box, action_box],
            layout=widgets.Layout(align_items="flex-start"),
        )

    def _build_figure(self) -> None:
        dpi = 100
    
        with plt.ioff():
            self.fig, self.ax = plt.subplots(
                figsize=(self.canvas_width_px / dpi, 8),
                dpi=dpi,
            )
    
        self.ax.set_axis_off()
        self.ax.set_aspect("equal", adjustable="box")
    
        self.image_artist = self.ax.imshow(
            np.zeros((10, 10, 3), dtype=np.float32),
            interpolation="nearest",
        )
    
        self.overlay_artist = self.ax.imshow(
            np.zeros((10, 10, 4), dtype=np.float32),
            interpolation="nearest",
        )
    
        self.title_artist = self.ax.set_title("")
    
        self.fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.94,
        )
    
        canvas = self.fig.canvas
    
        if hasattr(canvas, "layout"):
            canvas.layout.width = f"{self.canvas_width_px}px"
            canvas.layout.height = "800px"
    
        if hasattr(canvas, "toolbar_position"):
            canvas.toolbar_position = "bottom"
    
        if hasattr(canvas, "resizable"):
            canvas.resizable = False
    
        if hasattr(canvas, "header_visible"):
            canvas.header_visible = False
    
        if hasattr(canvas, "footer_visible"):
            canvas.footer_visible = True

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def show(self) -> "NotebookMaskReviewer":
        # ipympl's Canvas is already an ipywidget. Putting it inside an Output
        # widget can leave a blank canvas in some Notebook 7/JupyterLab setups.
        # Attach it directly to the widget tree instead.
        self.app = widgets.VBox(
            [self.controls, self.fig.canvas],
            layout=widgets.Layout(width="100%"),
        )
        display(self.app)

        self.fig.canvas.draw()
        if hasattr(self.fig.canvas, "flush_events"):
            self.fig.canvas.flush_events()

        return self

    @property
    def current_path(self) -> Path:
        return self.image_paths[self.index]

    def _paths_for_current(self) -> tuple[Path, Path, Path]:
        stem = make_annotation_stem(self.current_path)
        device = infer_device_label(self.current_path)
        folder = self.out_dir / device
        folder.mkdir(parents=True, exist_ok=True)

        return (
            folder / f"{stem}_regions.npy",
            folder / f"{stem}_regions.json",
            folder / f"{stem}_preview.png",
        )

    def _load_current(self) -> None:
        path = self.current_path
        image, report = load_image(
            path.name,
            base_dir=path.parent,
            as_chw=False,
            scale=True,
            fast_scale=True,
        )

        image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")

        self.image = image
        self.load_report = report
        self.display_image, self.scale_y, self.scale_x = _resize_for_display(
            image,
            self.max_display_side,
        )

        mask_path, _, _ = self._paths_for_current()
        if self.load_existing and mask_path.exists():
            labels = np.load(mask_path)
            if labels.shape != image.shape[:2]:
                raise ValueError(
                    f"Saved mask shape {labels.shape} does not match "
                    f"image shape {image.shape[:2]} for {path}"
                )
            self.labels = labels.astype(np.uint8, copy=False)
            loaded_text = "Loaded existing annotation."
        else:
            self.labels = propose_region_mask(image)
            loaded_text = "Created a new automatic proposal."

        self._last_point = None
        self._refresh_display()
        self._set_status(loaded_text)

    def _refresh_display(self) -> None:
        if self.image is None or self.labels is None or self.display_image is None:
            return

        shown_h, shown_w = self.display_image.shape[:2]
        self.display_labels = cv2.resize(
            self.labels,
            (shown_w, shown_h),
            interpolation=cv2.INTER_NEAREST,
        )

        self.image_artist.set_data(
            np.ascontiguousarray(self.display_image, dtype=np.float32)
        )
        self.overlay_artist.set_data(
            np.ascontiguousarray(
                _labels_to_rgba(self.display_labels, self.opacity.value),
                dtype=np.float32,
            )
        )
        # set_data() does not update the spatial extent inherited
        # from the original 10 × 10 placeholder image.
        extent = (
            -0.5,
            shown_w - 0.5,
            shown_h - 0.5,
            -0.5,
        )
        
        self.image_artist.set_extent(extent)
        self.overlay_artist.set_extent(extent)
        
        path = self.current_path
        self.title_artist.set_text(
            f"[{self.index + 1}/{len(self.image_paths)}] "
            f"{path.parent.name}/{path.name}"
        )
        self.progress.value = (
            f"<b>{self.index + 1}/{len(self.image_paths)}</b><br>"
            f"Device: {infer_device_label(path)}"
        )

        self.ax.set_xlim(-0.5, shown_w - 0.5)
        self.ax.set_ylim(shown_h - 0.5, -0.5)
        self.ax.set_aspect("equal", adjustable="box")
        
        # Size the actual Matplotlib figure according to the image aspect ratio.
        aspect = shown_h / shown_w
        
        plot_height_px = int(round(self.canvas_width_px * aspect))
        canvas_height_px = max(400, plot_height_px + 70)
        
        dpi = self.fig.get_dpi()
        
        self.fig.set_size_inches(
            self.canvas_width_px / dpi,
            canvas_height_px / dpi,
            forward=True,
        )
        
        if hasattr(self.fig.canvas, "layout"):
            self.fig.canvas.layout.width = f"{self.canvas_width_px}px"
            self.fig.canvas.layout.height = f"{canvas_height_px}px"
        
        self.fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.94,
        )
        
        self.fig.canvas.draw()

    def force_refresh(self) -> None:
        """Force the current image and overlay to be sent to the browser canvas."""
        self._refresh_display()
        self.fig.canvas.draw()
        if hasattr(self.fig.canvas, "flush_events"):
            self.fig.canvas.flush_events()

    def diagnostics(self) -> dict:
        """Return basic state used to diagnose a blank canvas."""
        import matplotlib

        result = {
            "backend": matplotlib.get_backend(),
            "current_path": str(self.current_path),
            "canvas_type": type(self.fig.canvas).__name__,
        }

        if self.image is not None:
            result["image_shape"] = tuple(self.image.shape)
            result["image_dtype"] = str(self.image.dtype)
            result["image_min"] = float(np.nanmin(self.image))
            result["image_max"] = float(np.nanmax(self.image))

        if self.display_image is not None:
            result["display_shape"] = tuple(self.display_image.shape)
            result["display_min"] = float(np.nanmin(self.display_image))
            result["display_max"] = float(np.nanmax(self.display_image))

        if self.labels is not None:
            values, counts = np.unique(self.labels, return_counts=True)
            result["label_counts"] = {
                int(v): int(c) for v, c in zip(values, counts)
            }

        return result

    def _set_status(self, text: str) -> None:
        if self.labels is None:
            counts = ""
        else:
            unique, n = np.unique(self.labels, return_counts=True)
            count_map = dict(zip(unique.tolist(), n.tolist()))
            counts = (
                f" | ignore={count_map.get(0, 0):,}"
                f", background={count_map.get(1, 0):,}"
                f", foreground={count_map.get(2, 0):,}"
            )
        self.status.value = f"{text}{counts}"

    def _update_proposal_control_state(self) -> None:
        """Enable only controls relevant to the selected proposal mode."""
        mode = self.threshold_mode.value

        self.foreground_quantile.disabled = mode != "quantile"
        self.local_block_size.disabled = mode != "local"
        self.local_offset.disabled = mode != "local"

        self.extreme_quantile.disabled = not self.use_extreme_rescue.value
        self.max_hole_area.disabled = not self.fill_holes.value

    def _on_threshold_mode_change(self, _change) -> None:
        self._update_proposal_control_state()

    def _on_extreme_rescue_change(self, _change) -> None:
        self._update_proposal_control_state()

    def _on_fill_holes_change(self, _change) -> None:
        self._update_proposal_control_state()

    def _proposal_settings(self) -> dict:
        return {
            "signal_mode": self.signal_mode.value,
            "polarity": self.polarity.value,
            "background_sigma": float(self.background_sigma.value),
            "threshold_mode": self.threshold_mode.value,
            "foreground_quantile": float(self.foreground_quantile.value),
            "local_block_size": int(self.local_block_size.value),
            "local_offset": float(self.local_offset.value),
            "use_extreme_rescue": bool(self.use_extreme_rescue.value),
            "extreme_quantile": float(self.extreme_quantile.value),
            "min_object_size": int(self.min_object_size.value),
            "max_object_size": int(self.max_object_size.value),
            "close_radius": int(self.close_radius.value),
            "fill_holes": bool(self.fill_holes.value),
            "max_hole_area": int(self.max_hole_area.value),
        }

    def _on_propose(self, _button) -> None:
        if self.image is None:
            return
        self.labels = propose_region_mask(
            self.image,
            **self._proposal_settings(),
        )
        self._refresh_display()
        self._set_status("Rebuilt automatic proposal.")

    def _on_reset_background(self, _button) -> None:
        if self.image is None:
            return
        self.labels = np.full(
            self.image.shape[:2],
            LABEL_BACKGROUND,
            dtype=np.uint8,
        )
        self._refresh_display()
        self._set_status("Set the full image to background.")

    def _on_opacity_change(self, _change) -> None:
        self._refresh_display()

    def _display_to_original(self, x: float, y: float) -> tuple[int, int]:
        if self.image is None:
            return 0, 0

        height, width = self.image.shape[:2]
        xx = int(round(float(x) * self.scale_x))
        yy = int(round(float(y) * self.scale_y))
        return (
            int(np.clip(xx, 0, width - 1)),
            int(np.clip(yy, 0, height - 1)),
        )

    def _paint_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        if self.labels is None:
            return

        label = int(self.brush_mode.value)
        radius = int(self.brush_radius.value)

        cv2.line(
            self.labels,
            start,
            end,
            color=label,
            thickness=max(1, radius * 2),
            lineType=cv2.LINE_8,
        )
        cv2.circle(
            self.labels,
            end,
            radius=radius,
            color=label,
            thickness=-1,
            lineType=cv2.LINE_8,
        )

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._painting = True
        point = self._display_to_original(event.xdata, event.ydata)
        self._last_point = point
        self._paint_segment(point, point)
        self._refresh_display()

    def _on_motion(self, event) -> None:
        if (
            not self._painting
            or self._last_point is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        point = self._display_to_original(event.xdata, event.ydata)
        self._paint_segment(self._last_point, point)
        self._last_point = point
        self._refresh_display()

    def _on_release(self, _event) -> None:
        if self._painting:
            self._painting = False
            self._last_point = None
            self._set_status("Mask edited.")

    def save_current(self) -> tuple[Path, Path]:
        if self.image is None or self.labels is None:
            raise RuntimeError("No image is loaded.")

        mask_path, metadata_path, preview_path = self._paths_for_current()
        np.save(mask_path, self.labels.astype(np.uint8))

        unique, counts = np.unique(self.labels, return_counts=True)
        count_map = {
            LABEL_NAMES.get(int(label), str(int(label))): int(count)
            for label, count in zip(unique, counts)
        }

        report_dict = {}
        if self.load_report is not None:
            for name in (
                "path", "shape", "dtype", "mode", "pages", "used_backend",
                "bit_depth", "white_level", "shifted", "warnings"
            ):
                if hasattr(self.load_report, name):
                    value = getattr(self.load_report, name)
                    if isinstance(value, tuple):
                        value = list(value)
                    report_dict[name] = value

        metadata = {
            "image_path": str(self.current_path),
            "device": infer_device_label(self.current_path),
            "image_shape": list(self.image.shape),
            "mask_path": str(mask_path),
            "labels": LABEL_NAMES,
            "class_counts": count_map,
            "proposal_settings": self._proposal_settings(),
            "loader_report": report_dict,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        if self.save_preview:
            shown_h, shown_w = self.display_image.shape[:2]
            preview_labels = cv2.resize(
                self.labels,
                (shown_w, shown_h),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = _labels_to_rgba(preview_labels, 0.45)

            base = np.clip(self.display_image * 255.0, 0, 255).astype(np.uint8)
            alpha = overlay[..., 3:4]
            rgb = overlay[..., :3]
            preview = (
                (1.0 - alpha) * base.astype(np.float32)
                + alpha * (rgb * 255.0)
            )
            preview = np.clip(preview, 0, 255).astype(np.uint8)
            cv2.imwrite(
                str(preview_path),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
            )

        self._set_status(f"Saved {mask_path.name}.")
        return mask_path, metadata_path

    def _on_save(self, _button) -> None:
        self.save_current()

    def _on_save_next(self, _button) -> None:
        self.save_current()
        self.next_image()

    def next_image(self) -> None:
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self._load_current()
        else:
            self._set_status("Already at the final image.")

    def previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_current()
        else:
            self._set_status("Already at the first image.")

    def _on_next(self, _button) -> None:
        self.next_image()

    def _on_previous(self, _button) -> None:
        self.previous_image()

    def _on_reload(self, _button) -> None:
        mask_path, _, _ = self._paths_for_current()
        if not mask_path.exists():
            self._set_status("No saved mask exists for this image.")
            return

        labels = np.load(mask_path)
        if self.image is None or labels.shape != self.image.shape[:2]:
            raise ValueError("Saved mask shape does not match the loaded image.")

        self.labels = labels.astype(np.uint8, copy=False)
        self._refresh_display()
        self._set_status("Reloaded saved annotation.")


def review_ext_image_masks(
    ext_image_folders,
    *,
    out_dir="./region_annotations",
    start_index=0,
    max_display_side=1200,
    canvas_width_px=1000,
    save_preview=True,
    load_existing=True,
) -> NotebookMaskReviewer:
    """
    Create and show the notebook reviewer for EXT_IMAGES_FOLDERS.

    Run ``%matplotlib widget`` in a notebook cell before calling this function.
    """
    reviewer = NotebookMaskReviewer.from_folders(
        ext_image_folders,
        out_dir=out_dir,
        start_index=start_index,
        max_display_side=max_display_side,
        save_preview=save_preview,
        load_existing=load_existing,
        canvas_width_px=canvas_width_px
    )
    return reviewer.show()


# -----------------------------------------------------------------------------
# Extended reviewer: well exclusion, local-background ring, and persistent
# settings across images. Later definitions intentionally replace the base
# names above while keeping the module usable as one standalone file.
# -----------------------------------------------------------------------------

from skimage import measure

LABEL_OUTSIDE_WELL = np.uint8(3)
LABEL_NAMES[int(LABEL_OUTSIDE_WELL)] = "outside_well"
PHONE_DEVICES = {"iphone", "googlepixel"}


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected True component."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    labelled = measure.label(mask, connectivity=2)
    counts = np.bincount(labelled.ravel())
    if counts.size <= 1:
        return np.zeros_like(mask, dtype=bool)

    counts[0] = 0
    return labelled == int(np.argmax(counts))


def propose_well_mask(
    image: np.ndarray,
    *,
    signal_mode: str = "luma",
    blur_sigma: float = 35.0,
    threshold_offset: float = 0.0,
    close_radius: int = 20,
    shrink_margin: int = 8,
    use_ellipse: bool = False,
    downsample_max_side: int = 700,
) -> np.ndarray:
    """Estimate the bright physical well and reject dark surroundings."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    height, width = image.shape[:2]
    signal = _signal_from_rgb(image, signal_mode)

    scale = min(
        1.0,
        float(max(32, int(downsample_max_side))) / float(max(height, width)),
    )

    if scale < 1.0:
        small_w = max(8, int(round(width * scale)))
        small_h = max(8, int(round(height * scale)))
        signal_small = cv2.resize(
            signal,
            (small_w, small_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        signal_small = signal.copy()

    sigma_small = max(0.0, float(blur_sigma) * scale)
    if sigma_small > 0:
        smooth = cv2.GaussianBlur(
            signal_small,
            ksize=(0, 0),
            sigmaX=sigma_small,
            sigmaY=sigma_small,
            borderType=cv2.BORDER_REFLECT,
        )
    else:
        smooth = signal_small

    finite = np.isfinite(smooth)
    if not finite.any():
        return np.ones((height, width), dtype=np.uint8)

    values = smooth[finite]
    if np.allclose(values.min(), values.max()):
        return np.ones((height, width), dtype=np.uint8)

    threshold = float(filters.threshold_otsu(values))
    threshold = float(np.clip(threshold + float(threshold_offset), 0.0, 1.0))
    candidate = finite & (smooth >= threshold)

    close_small = max(0, int(round(int(close_radius) * scale)))
    if close_small > 0:
        candidate = morphology.binary_closing(
            candidate,
            morphology.disk(close_small),
        )

    candidate = ndi.binary_fill_holes(candidate)
    candidate = _largest_component(candidate)

    if not candidate.any():
        return np.ones((height, width), dtype=np.uint8)

    if use_ellipse:
        candidate_u8 = candidate.astype(np.uint8)
        contours, _ = cv2.findContours(
            candidate_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if contour.shape[0] >= 5:
                fitted = np.zeros_like(candidate_u8)
                cv2.ellipse(
                    fitted,
                    cv2.fitEllipse(contour),
                    color=1,
                    thickness=-1,
                )
                candidate = fitted.astype(bool)

    full = cv2.resize(
        candidate.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    margin = max(0, int(shrink_margin))
    if margin > 0 and full.any():
        full = ndi.distance_transform_edt(full) > float(margin)

    full = _largest_component(full)
    if not full.any():
        return np.ones((height, width), dtype=np.uint8)

    return full.astype(np.uint8)


def derive_local_background(
    labels: np.ndarray,
    well_mask: Optional[np.ndarray] = None,
    *,
    inner_radius: int = 2,
    outer_radius: int = 6,
) -> np.ndarray:
    """Derive a near-cell ring limited to confirmed valid background."""
    labels = np.asarray(labels, dtype=np.uint8)
    foreground = labels == LABEL_FOREGROUND
    background = labels == LABEL_BACKGROUND

    if well_mask is not None:
        inside = np.asarray(well_mask, dtype=bool)
        foreground &= inside
        background &= inside

    inner_radius = max(0, int(inner_radius))
    outer_radius = max(inner_radius + 1, int(outer_radius))

    if inner_radius == 0:
        inner = foreground.copy()
    else:
        inner = ndi.binary_dilation(foreground, iterations=inner_radius)

    outer = ndi.binary_dilation(foreground, iterations=outer_radius)
    return (outer & ~inner & background).astype(bool)


def _well_ring_overlay(
    labels: np.ndarray,
    well_mask: np.ndarray,
    *,
    region_opacity: float,
    outside_opacity: float,
    ring: Optional[np.ndarray] = None,
    ring_opacity: float = 0.55,
) -> np.ndarray:
    """Create red foreground, blue ring, yellow ignore, purple outside overlay."""
    labels = np.asarray(labels, dtype=np.uint8)
    inside = np.asarray(well_mask, dtype=bool)
    rgba = np.zeros(labels.shape + (4,), dtype=np.float32)

    ignore = (labels == LABEL_IGNORE) & inside
    background = (labels == LABEL_BACKGROUND) & inside
    foreground = (labels == LABEL_FOREGROUND) & inside
    outside = ~inside

    alpha = float(np.clip(region_opacity, 0.0, 1.0))

    rgba[background, :3] = (0.0, 0.0, 0.0)
    rgba[background, 3] = 0.05 * alpha

    rgba[ignore, :3] = (1.0, 0.72, 0.05)
    rgba[ignore, 3] = alpha

    rgba[foreground, :3] = (1.0, 0.08, 0.08)
    rgba[foreground, 3] = alpha

    if ring is not None:
        local_ring = np.asarray(ring, dtype=bool) & background
        rgba[local_ring, :3] = (0.05, 0.35, 1.0)
        rgba[local_ring, 3] = float(np.clip(ring_opacity, 0.0, 1.0))

    rgba[outside, :3] = (0.58, 0.12, 0.72)
    rgba[outside, 3] = float(np.clip(outside_opacity, 0.0, 1.0))
    return rgba


_BaseNotebookMaskReviewer = NotebookMaskReviewer


class NotebookMaskReviewer(_BaseNotebookMaskReviewer):
    """Reviewer extended with an editable well mask and live local ring."""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        out_dir: str | Path = "./region_annotations",
        start_index: int = 0,
        max_display_side: int = 1200,
        canvas_width_px: int = 1000,
        save_preview: bool = True,
        load_existing: bool = True,
        auto_well_for_phones: bool = True,
    ):
        self.auto_well_for_phones = bool(auto_well_for_phones)
        self.well_mask: Optional[np.ndarray] = None
        self.display_well_mask: Optional[np.ndarray] = None
        self.display_ring: Optional[np.ndarray] = None

        super().__init__(
            image_paths,
            out_dir=out_dir,
            start_index=start_index,
            max_display_side=max_display_side,
            canvas_width_px=canvas_width_px,
            save_preview=save_preview,
            load_existing=load_existing,
        )

    def _build_widgets(self) -> None:
        super()._build_widgets()

        # Replace the old numeric brush selector with semantic edit modes.
        self.brush_mode.options = [
            ("Foreground", "foreground"),
            ("Background", "background"),
            ("Ignore", "ignore"),
            ("Well include", "well_include"),
            ("Well exclude", "well_exclude"),
        ]
        self.brush_mode.value = "foreground"

        self.well_signal_mode = widgets.Dropdown(
            options=["luma", "red_green_max", "max", "green"],
            value="luma",
            description="Well signal",
        )
        self.well_blur_sigma = widgets.FloatSlider(
            value=35.0,
            min=0.0,
            max=150.0,
            step=1.0,
            description="Well blur",
            continuous_update=False,
        )
        self.well_threshold_offset = widgets.FloatSlider(
            value=0.0,
            min=-0.20,
            max=0.20,
            step=0.005,
            description="Well offset",
            readout_format=".3f",
            continuous_update=False,
        )
        self.well_close_radius = widgets.IntSlider(
            value=20,
            min=0,
            max=100,
            step=1,
            description="Well close",
            continuous_update=False,
        )
        self.well_shrink_margin = widgets.IntSlider(
            value=8,
            min=0,
            max=100,
            step=1,
            description="Well shrink",
            continuous_update=False,
        )
        self.well_use_ellipse = widgets.Checkbox(
            value=False,
            description="Fit ellipse",
            indent=False,
        )
        self.build_well_button = widgets.Button(description="Build well mask")
        self.all_inside_button = widgets.Button(description="All inside well")

        self.show_ring = widgets.Checkbox(
            value=True,
            description="Show local ring",
            indent=False,
        )
        self.ring_inner_radius = widgets.IntSlider(
            value=2,
            min=0,
            max=30,
            step=1,
            description="Ring inner",
            continuous_update=False,
        )
        self.ring_outer_radius = widgets.IntSlider(
            value=6,
            min=1,
            max=60,
            step=1,
            description="Ring outer",
            continuous_update=False,
        )
        self.ring_opacity = widgets.FloatSlider(
            value=0.55,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Ring opacity",
            continuous_update=False,
        )
        self.outside_opacity = widgets.FloatSlider(
            value=0.45,
            min=0.0,
            max=0.9,
            step=0.05,
            description="Outside opacity",
            continuous_update=False,
        )

        self.build_well_button.on_click(self._on_build_well)
        self.all_inside_button.on_click(self._on_all_inside)

        for control in (
            self.show_ring,
            self.ring_inner_radius,
            self.ring_outer_radius,
            self.ring_opacity,
            self.outside_opacity,
        ):
            control.observe(self._on_ring_display_change, names="value")

        proposal_box = widgets.VBox([
            widgets.HTML("<b>Cell proposal</b>"),
            widgets.HBox([
                self.signal_mode,
                self.polarity,
                self.threshold_mode,
            ]),
            self.foreground_quantile,
            self.local_block_size,
            self.local_offset,
            self.background_sigma,
            self.use_extreme_rescue,
            self.extreme_quantile,
            self.min_object_size,
            self.max_object_size,
            widgets.HTML(
                "<small>Max area is connected-component area; 0 disables it.</small>"
            ),
            self.close_radius,
            self.fill_holes,
            self.max_hole_area,
            widgets.HBox([
                self.propose_button,
                self.reset_background_button,
            ]),
        ])

        well_box = widgets.VBox([
            widgets.HTML("<b>Well mask</b>"),
            self.well_signal_mode,
            self.well_blur_sigma,
            self.well_threshold_offset,
            self.well_close_radius,
            self.well_shrink_margin,
            self.well_use_ellipse,
            widgets.HBox([
                self.build_well_button,
                self.all_inside_button,
            ]),
        ])

        edit_box = widgets.VBox([
            widgets.HTML("<b>Manual correction</b>"),
            self.brush_mode,
            self.brush_radius,
            self.opacity,
            widgets.HTML(
                "Region brushes edit classes 0–2. Well brushes edit the "
                "purple outside-well mask."
            ),
        ])

        ring_box = widgets.VBox([
            widgets.HTML("<b>Local background display</b>"),
            self.show_ring,
            self.ring_inner_radius,
            self.ring_outer_radius,
            self.ring_opacity,
            self.outside_opacity,
        ])

        action_box = widgets.VBox([
            widgets.HTML("<b>Files</b>"),
            widgets.HBox([
                self.previous_button,
                self.next_button,
                self.reload_button,
            ]),
            widgets.HBox([
                self.save_button,
                self.save_next_button,
            ]),
            self.progress,
            self.status,
        ])

        self.controls = widgets.VBox([
            widgets.HBox(
                [proposal_box, well_box],
                layout=widgets.Layout(align_items="flex-start"),
            ),
            widgets.HBox(
                [edit_box, ring_box, action_box],
                layout=widgets.Layout(align_items="flex-start"),
            ),
        ])

    def _paths_for_current(self) -> tuple[Path, Path, Path, Path]:
        """
        Build output paths that mirror the immediate source folder.

        Example
        -------
        Source:
            .../20251106_25065441_iPhone_XR_JPEG/IMG_1234.jpeg

        Outputs:
            region_annotations/
                20251106_25065441_iPhone_XR_JPEG/
                    IMG_1234_regions.npy
                    IMG_1234_well_mask.npy
                    IMG_1234_regions.json
                    IMG_1234_preview.png
        """
        source_folder_name = self.current_path.parent.name
        image_stem = self.current_path.stem

        folder = self.out_dir / source_folder_name
        folder.mkdir(parents=True, exist_ok=True)

        return (
            folder / f"{image_stem}_regions.npy",
            folder / f"{image_stem}_well_mask.npy",
            folder / f"{image_stem}_regions.json",
            folder / f"{image_stem}_preview.png",
        )

    def _well_settings(self) -> dict:
        return {
            "signal_mode": self.well_signal_mode.value,
            "blur_sigma": float(self.well_blur_sigma.value),
            "threshold_offset": float(self.well_threshold_offset.value),
            "close_radius": int(self.well_close_radius.value),
            "shrink_margin": int(self.well_shrink_margin.value),
            "use_ellipse": bool(self.well_use_ellipse.value),
        }

    def _ring_settings(self) -> dict:
        inner = max(0, int(self.ring_inner_radius.value))
        outer = max(inner + 1, int(self.ring_outer_radius.value))
        return {
            "show": bool(self.show_ring.value),
            "inner_radius": inner,
            "outer_radius": outer,
            "opacity": float(self.ring_opacity.value),
        }

    def combined_labels(self) -> np.ndarray:
        """Return 0/1/2 labels with class 3 outside the valid well."""
        if self.labels is None or self.well_mask is None:
            raise RuntimeError("No image is loaded.")

        combined = self.labels.astype(np.uint8, copy=True)
        combined[self.well_mask == 0] = LABEL_OUTSIDE_WELL
        return combined

    def _load_current(self) -> None:
        """Load an image while retaining every current widget setting."""
        path = self.current_path
        image, report = load_image(
            path.name,
            base_dir=path.parent,
            as_chw=False,
            scale=True,
            fast_scale=True,
        )

        image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")

        self.image = image
        self.load_report = report
        self.display_image, self.scale_y, self.scale_x = _resize_for_display(
            image,
            self.max_display_side,
        )

        regions_path, well_path, _, _ = self._paths_for_current()

        if self.load_existing and regions_path.exists():
            loaded = np.load(regions_path).astype(np.uint8, copy=False)
            if loaded.shape != image.shape[:2]:
                raise ValueError(
                    f"Saved mask shape {loaded.shape} does not match "
                    f"image shape {image.shape[:2]} for {path}"
                )

            if well_path.exists():
                well = np.load(well_path).astype(bool)
                if well.shape != image.shape[:2]:
                    raise ValueError(
                        f"Saved well shape {well.shape} does not match "
                        f"image shape {image.shape[:2]} for {path}"
                    )
            else:
                well = loaded != LABEL_OUTSIDE_WELL

            self.well_mask = well.astype(np.uint8)
            self.labels = loaded.copy()
            self.labels[self.labels == LABEL_OUTSIDE_WELL] = LABEL_BACKGROUND
            loaded_text = "Loaded existing annotation."

        else:
            # This is the requested persistence behavior: the new image uses
            # the controls exactly as they currently stand.
            self.labels = propose_region_mask(
                image,
                **self._proposal_settings(),
            )

            device = infer_device_label(path)
            if self.auto_well_for_phones and device in PHONE_DEVICES:
                self.well_mask = propose_well_mask(
                    image,
                    **self._well_settings(),
                )
                loaded_text = (
                    "Created cell and phone-well proposals with current settings."
                )
            else:
                self.well_mask = np.ones(image.shape[:2], dtype=np.uint8)
                loaded_text = (
                    "Created cell proposal with current settings; full image "
                    "marked inside the well."
                )

        self._last_point = None
        self._refresh_display()
        self._set_status(loaded_text)

    def _display_ring_mask(self) -> Optional[np.ndarray]:
        if (
            not self.show_ring.value
            or self.display_labels is None
            or self.display_well_mask is None
        ):
            return None

        scale_mean = max(
            1e-6,
            0.5 * (float(self.scale_x) + float(self.scale_y)),
        )
        inner_original = max(0, int(self.ring_inner_radius.value))
        outer_original = max(
            inner_original + 1,
            int(self.ring_outer_radius.value),
        )
        inner_display = max(0, int(round(inner_original / scale_mean)))
        outer_display = max(
            inner_display + 1,
            int(round(outer_original / scale_mean)),
        )

        return derive_local_background(
            self.display_labels,
            self.display_well_mask,
            inner_radius=inner_display,
            outer_radius=outer_display,
        )

    def _refresh_display(self) -> None:
        if (
            self.image is None
            or self.labels is None
            or self.well_mask is None
            or self.display_image is None
        ):
            return

        shown_h, shown_w = self.display_image.shape[:2]
        self.display_labels = cv2.resize(
            self.labels,
            (shown_w, shown_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        self.display_well_mask = cv2.resize(
            self.well_mask,
            (shown_w, shown_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        self.display_ring = self._display_ring_mask()

        overlay = _well_ring_overlay(
            self.display_labels,
            self.display_well_mask,
            region_opacity=float(self.opacity.value),
            outside_opacity=float(self.outside_opacity.value),
            ring=self.display_ring,
            ring_opacity=float(self.ring_opacity.value),
        )

        self.image_artist.set_data(
            np.ascontiguousarray(self.display_image, dtype=np.float32)
        )
        self.overlay_artist.set_data(
            np.ascontiguousarray(overlay, dtype=np.float32)
        )

        extent = (-0.5, shown_w - 0.5, shown_h - 0.5, -0.5)
        self.image_artist.set_extent(extent)
        self.overlay_artist.set_extent(extent)

        path = self.current_path
        self.title_artist.set_text(
            f"[{self.index + 1}/{len(self.image_paths)}] "
            f"{path.parent.name}/{path.name}"
        )
        self.progress.value = (
            f"<b>{self.index + 1}/{len(self.image_paths)}</b><br>"
            f"Device: {infer_device_label(path)}"
        )

        self.ax.set_xlim(-0.5, shown_w - 0.5)
        self.ax.set_ylim(shown_h - 0.5, -0.5)
        self.ax.set_aspect("equal", adjustable="box")

        aspect = shown_h / shown_w
        canvas_height_px = max(
            400,
            int(round(self.canvas_width_px * aspect)) + 70,
        )
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(
            self.canvas_width_px / dpi,
            canvas_height_px / dpi,
            forward=True,
        )

        if hasattr(self.fig.canvas, "layout"):
            self.fig.canvas.layout.width = f"{self.canvas_width_px}px"
            self.fig.canvas.layout.height = f"{canvas_height_px}px"

        self.fig.subplots_adjust(
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.94,
        )
        self.fig.canvas.draw()

    def _set_status(self, text: str) -> None:
        if self.labels is None or self.well_mask is None:
            counts = ""
        else:
            combined = self.combined_labels()
            values, numbers = np.unique(combined, return_counts=True)
            count_map = {
                int(value): int(number)
                for value, number in zip(values, numbers)
            }
            counts = (
                f" | ignore={count_map.get(0, 0):,}"
                f", background={count_map.get(1, 0):,}"
                f", foreground={count_map.get(2, 0):,}"
                f", outside={count_map.get(3, 0):,}"
            )
        self.status.value = f"{text}{counts}"

    def _on_ring_display_change(self, _change) -> None:
        self._refresh_display()

    def _on_build_well(self, _button) -> None:
        if self.image is None:
            return
        self.well_mask = propose_well_mask(
            self.image,
            **self._well_settings(),
        )
        self._refresh_display()
        self._set_status("Rebuilt well mask with current settings.")

    def _on_all_inside(self, _button) -> None:
        if self.image is None:
            return
        self.well_mask = np.ones(self.image.shape[:2], dtype=np.uint8)
        self._refresh_display()
        self._set_status("Marked the full image as inside the well.")

    def _paint_segment(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> None:
        if self.labels is None or self.well_mask is None:
            return

        mode = str(self.brush_mode.value)
        radius = int(self.brush_radius.value)
        thickness = max(1, radius * 2)

        if mode in {"foreground", "background", "ignore"}:
            value = {
                "foreground": int(LABEL_FOREGROUND),
                "background": int(LABEL_BACKGROUND),
                "ignore": int(LABEL_IGNORE),
            }[mode]
            target = self.labels
        elif mode in {"well_include", "well_exclude"}:
            value = 1 if mode == "well_include" else 0
            target = self.well_mask
        else:
            raise ValueError(f"Unknown brush mode: {mode}")

        cv2.line(
            target,
            start,
            end,
            color=value,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
        cv2.circle(
            target,
            end,
            radius=radius,
            color=value,
            thickness=-1,
            lineType=cv2.LINE_8,
        )

    def save_current(self) -> tuple[Path, Path, Path]:
        if self.image is None or self.labels is None or self.well_mask is None:
            raise RuntimeError("No image is loaded.")

        regions_path, well_path, metadata_path, preview_path = (
            self._paths_for_current()
        )
        combined = self.combined_labels()

        np.save(regions_path, combined.astype(np.uint8))
        np.save(well_path, self.well_mask.astype(bool))

        values, counts = np.unique(combined, return_counts=True)
        count_map = {
            LABEL_NAMES.get(int(value), str(int(value))): int(count)
            for value, count in zip(values, counts)
        }

        ring_settings = self._ring_settings()
        ring = derive_local_background(
            self.labels,
            self.well_mask,
            inner_radius=ring_settings["inner_radius"],
            outer_radius=ring_settings["outer_radius"],
        )

        report_dict = {}
        if self.load_report is not None:
            for name in (
                "path", "shape", "dtype", "mode", "pages", "used_backend",
                "bit_depth", "white_level", "shifted", "warnings",
            ):
                if hasattr(self.load_report, name):
                    value = getattr(self.load_report, name)
                    if isinstance(value, tuple):
                        value = list(value)
                    report_dict[name] = value

        metadata = {
            "image_path": str(self.current_path),
            "device": infer_device_label(self.current_path),
            "image_shape": list(self.image.shape),
            "regions_path": str(regions_path),
            "well_mask_path": str(well_path),
            "labels": LABEL_NAMES,
            "class_counts": count_map,
            "local_ring_pixels": int(ring.sum()),
            "proposal_settings": self._proposal_settings(),
            "well_settings": self._well_settings(),
            "ring_settings": ring_settings,
            "loader_report": report_dict,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        if self.save_preview:
            overlay = _well_ring_overlay(
                self.display_labels,
                self.display_well_mask,
                region_opacity=float(self.opacity.value),
                outside_opacity=float(self.outside_opacity.value),
                ring=self.display_ring,
                ring_opacity=float(self.ring_opacity.value),
            )
            base = np.clip(
                self.display_image * 255.0,
                0,
                255,
            ).astype(np.uint8)
            alpha = overlay[..., 3:4]
            preview = (
                (1.0 - alpha) * base.astype(np.float32)
                + alpha * (overlay[..., :3] * 255.0)
            )
            preview = np.clip(preview, 0, 255).astype(np.uint8)
            cv2.imwrite(
                str(preview_path),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
            )

        self._set_status(f"Saved {regions_path.name}.")
        return regions_path, well_path, metadata_path

    def _on_reload(self, _button) -> None:
        regions_path, well_path, _, _ = self._paths_for_current()
        if not regions_path.exists():
            self._set_status("No saved annotation exists for this image.")
            return

        loaded = np.load(regions_path).astype(np.uint8, copy=False)
        if self.image is None or loaded.shape != self.image.shape[:2]:
            raise ValueError("Saved region mask shape does not match the image.")

        if well_path.exists():
            well = np.load(well_path).astype(bool)
            if well.shape != self.image.shape[:2]:
                raise ValueError("Saved well mask shape does not match the image.")
        else:
            well = loaded != LABEL_OUTSIDE_WELL

        self.well_mask = well.astype(np.uint8)
        self.labels = loaded.copy()
        self.labels[self.labels == LABEL_OUTSIDE_WELL] = LABEL_BACKGROUND
        self._refresh_display()
        self._set_status("Reloaded saved annotation.")

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        result["well_settings"] = self._well_settings()
        result["ring_settings"] = self._ring_settings()
        if self.well_mask is not None:
            result["well_pixels"] = int(np.asarray(self.well_mask, bool).sum())
            result["combined_label_counts"] = {
                int(value): int(count)
                for value, count in zip(
                    *np.unique(self.combined_labels(), return_counts=True)
                )
            }
        return result


def annotation_output_paths(
    image_path: str | Path,
    out_dir: str | Path = "./region_annotations",
) -> tuple[Path, Path, Path, Path]:
    """
    Return the expected annotation output paths for one source image.

    The layout mirrors the source image's immediate parent folder.
    """
    image_path = Path(image_path)
    folder = Path(out_dir) / image_path.parent.name
    stem = image_path.stem

    return (
        folder / f"{stem}_regions.npy",
        folder / f"{stem}_well_mask.npy",
        folder / f"{stem}_regions.json",
        folder / f"{stem}_preview.png",
    )


def annotation_is_complete(
    image_path: str | Path,
    out_dir: str | Path = "./region_annotations",
) -> bool:
    """
    Return True when the required annotation files exist.

    The preview is intentionally optional because it is only a visual aid.
    """
    regions_path, well_path, metadata_path, _ = annotation_output_paths(
        image_path,
        out_dir=out_dir,
    )

    return (
        regions_path.is_file()
        and well_path.is_file()
        and metadata_path.is_file()
    )


def collect_unannotated_paths(
    root_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path = "./region_annotations",
    suffixes: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> tuple[list[Path], int]:
    """
    Collect source images and remove images with complete saved annotations.

    Returns
    -------
    remaining_paths:
        Images that still need annotation.
    skipped_count:
        Number of already completed images.
    """
    all_paths = collect_annotation_paths(
        root_dirs,
        suffixes=suffixes,
    )

    remaining_paths = [
        path
        for path in all_paths
        if not annotation_is_complete(path, out_dir=out_dir)
    ]

    return remaining_paths, len(all_paths) - len(remaining_paths)


def review_ext_image_masks(
    ext_image_folders: Sequence[str | Path],
    *,
    out_dir: str | Path = "./region_annotations",
    start_index: int = 0,
    max_display_side: int = 1200,
    canvas_width_px: int = 1000,
    save_preview: bool = True,
    load_existing: bool = True,
    auto_well_for_phones: bool = True,
    skip_existing: bool = False,
) -> NotebookMaskReviewer:
    """
    Create the reviewer; current settings persist across image changes.

    Parameters
    ----------
    skip_existing:
        If True, images with existing region mask, well mask, and metadata
        files are omitted before the reviewer starts.
    """
    out_dir = Path(out_dir)

    if skip_existing:
        image_paths, skipped_count = collect_unannotated_paths(
            ext_image_folders,
            out_dir=out_dir,
        )

        if not image_paths:
            raise RuntimeError(
                f"All {skipped_count} available images already have complete "
                f"annotations in {out_dir}."
            )

        print(
            f"Skipping {skipped_count} completed images; "
            f"{len(image_paths)} images remain."
        )
    else:
        image_paths = collect_annotation_paths(ext_image_folders)

    reviewer = NotebookMaskReviewer(
        image_paths,
        out_dir=out_dir,
        start_index=start_index,
        max_display_side=max_display_side,
        canvas_width_px=canvas_width_px,
        save_preview=save_preview,
        load_existing=load_existing,
        auto_well_for_phones=auto_well_for_phones,
    )

    return reviewer.show()


### USAGE IN JUPYTER_NOTEBOOK
"""
%matplotlib widget
reviewer = review_ext_image_masks(
    images,
    out_dir="./region_annotations",
    start_index=0,
    max_display_side=1200,
    canvas_width_px=1000,
    auto_well_for_phones=True,
    skip_existing=True,
)
"""

