from matplotlib.axes import Axes
import numpy as np

def unify_axis_limits(ax: Axes):
    """
    Set both axes to the common minimum and maximum of their current limits.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    """
    y_lim_min, y_lim_max = ax.get_ylim()
    x_lim_min, x_lim_max = ax.get_xlim()

    _min = min(y_lim_min, x_lim_min)
    _max = max(y_lim_max, x_lim_max)

    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)

    return

def adjust_fontsize_ticklabels(ax: Axes,
                               fontsize: int):
    """
    Set the font size of both axes' tick labels.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    fontsize : int
        Text font size in points.
    """
    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

def figure_label(ax: Axes, label, x: float = 0.0, y: float = 1.0):
    """
    labels individual subfigures. Requires subgrid to not use figure axis coordinates.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    label : object
        Display label or identifier for the corresponding item.
    x : float
        Horizontal label position in the axes data coordinates. Default is ``0.0``.
    y : float
        Vertical label position in the axes data coordinates. Default is ``1.0``.
    """
    ax.text(x, y, label, fontsize=12)
    return

def prep_image_axis(ax: Axes):
    """
    Hide all decorations on an image axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    """
    ax.axis("off")
    return

def remove_ticks_and_labels(ax):
    """
    Clear axis titles and tick labels and disable tick marks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis receiving the plot or formatting changes.
    """
    ax.set_xlabel("")
    ax.set_ylabel("")  #
    ax.set_xticklabels([])
    ax.set_yticklabels([])  #
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def remove_axis_labels(ax: Axes):
    """
    Hide tick marks and tick-label text on the supplied axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    """
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def imshow_no_axes(ax: Axes, img_rgb_uint8: np.ndarray) -> None:
    """
    Display an RGB image without ticks or visible spines.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis receiving the plot or formatting changes.
    img_rgb_uint8 : numpy.ndarray
        RGB image with uint8 channels in the range 0 to 255.
    """
    ax.imshow(img_rgb_uint8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

