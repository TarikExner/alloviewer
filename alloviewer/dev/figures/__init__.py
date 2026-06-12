from typing import Any, Callable

from .figure_1 import figure_1_generation
from .figure_2 import figure_2_generation
from .figure_3 import figure_3_generation
from .figure_4 import figure_4_generation
from .figure_5 import figure_5_generation

from .figure_S1 import figure_S1_generation
from .figure_S2 import figure_S2_generation
from .figure_S3 import figure_S3_generation
from .figure_S4 import figure_S4_generation
from .figure_S5 import figure_S5_generation
from .figure_S6 import figure_S6_generation
from .figure_S7 import figure_S7_generation
from .figure_S8 import figure_S8_generation
from .figure_S9 import figure_S9_generation
from .figure_SIM import figure_SIM_generation

__all__ = [
    "figure_1_generation",
    "figure_2_generation",
    "figure_3_generation",
    "figure_4_generation",
    "figure_5_generation",

    "figure_S1_generation",
    "figure_S2_generation",
    "figure_S3_generation",
    "figure_S4_generation",
    "figure_S5_generation",
    "figure_S6_generation",
    "figure_S7_generation",
    "figure_S8_generation",
    "figure_S9_generation",

    "figure_SIM_generation",
]


DIRECTORIES = {
    "h5_path": "../scripts/image_datasets",
    "model_output_dir": "../scripts/models",

    "figure_output_dir": "./figures",
    "figure_data_dir": "./figure_data",
    "validation_results_dir": "../scripts/results",
    "ext_images_dir": "../scripts/ext_images",
    "sketch_dir": "./sketches/",
    "flow_data_dir": "../scripts/flow_data/"
}

def generate_all_figures():

    figure_1_generation(**DIRECTORIES)
    figure_2_generation(**DIRECTORIES)
    figure_3_generation(**DIRECTORIES)
    figure_4_generation(**DIRECTORIES)
    figure_5_generation(**DIRECTORIES)

    figure_S1_generation(**DIRECTORIES)
    figure_S2_generation(**DIRECTORIES)
    figure_S3_generation(**DIRECTORIES)
    figure_S4_generation(**DIRECTORIES)
    figure_S5_generation(**DIRECTORIES)
    figure_S6_generation(**DIRECTORIES)
    figure_S7_generation(**DIRECTORIES)
    figure_S8_generation(**DIRECTORIES)
    figure_S9_generation(**DIRECTORIES)

    figure_SIM_generation(**DIRECTORIES)

