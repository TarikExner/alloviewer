AXIS_LABEL_SIZE = 6
TITLE_SIZE = 8
UMAP_LABEL_SIZE = 6

DPI = 300

FIGURE_WIDTH_FULL = 6.75
FIGURE_WIDTH_HALF = FIGURE_WIDTH_FULL / 2

FIGURE_HEIGHT_FULL = 9.375
FIGURE_HEIGHT_HALF = FIGURE_HEIGHT_FULL / 2

SUPERVISED_SCORE = "f1_score"
UNSUPERVISED_SCORE = "jaccard_score"

CONF_MATRIX_COLORS = ["#31688E", "#35B779", "#FDE725", "#440154"]

CONF_MATRIX_LABEL_DICT = {
    "fp": "false pos.",
    "fn": "false neg.",
    "tp": "true pos.",
    "tn": "true neg.",
}

EXPERIMENT_LEGEND_CMAP = "tab20"

TWO_COL_LEGEND = {
    "ncol": 2,  # Two columns
    "columnspacing": 0.1,  # Reduce spacing between columns
    "handletextpad": 0.1,  # Reduce spacing between handles and text
    "borderaxespad": 0.1,  # Reduce padding around the legend box
}

PHONE_DICT = {
    "iPhone": "smartphone\nbrand 2",
    "GooglePixel": "smartphone\nbrand 1",
    "Microscope": "microscope camera\nrgb",
    "Monochrome": "microscope camera\nmonochrome",
    "Generic": "generic\n",
    "Simulated": "simulated\n"
}

STRIPPLOT_PARAMS = {"linewidth": 0.5, "dodge": True, "s": 2}

BOXPLOT_PARAMS = {
    "boxprops": dict(facecolor="white"),
    "whis": (0, 100),
    "linewidth": 1,
    "showfliers": False,
}

XTICKLABEL_PARAMS = {
    "ha": "right",
    "rotation": 45,
    "rotation_mode": "anchor",
    "fontsize": AXIS_LABEL_SIZE,
}

TICKPARAMS_PARAMS = {"axis": "both", "labelsize": AXIS_LABEL_SIZE}

CENTERED_LEGEND_PARAMS = {
    "bbox_to_anchor": (1, 0.5),
    "loc": "center left",
    "fontsize": AXIS_LABEL_SIZE,
    "markerscale": 0.5,
}

HIST_CMAP = "colorblind"


SCATTER_KWARGS = {
    "s": 4,
    "edgecolor": "black",
    "linewidth": 0.3,
    "rasterized": True
}
