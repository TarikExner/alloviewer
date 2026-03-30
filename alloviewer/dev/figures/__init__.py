from typing import Any, Callable

from .figure_S1 import figure_S1_generation
from .figure_S2 import figure_S2_generation
from .figure_S3 import figure_S3_generation
from .figure_S4 import figure_S4_generation
from .figure_S6 import figure_S6_generation
from .figure_S7 import figure_S7_generation
from .figure_SIM import figure_SIM_generation

__all__ = [
    "figure_S1_generation",
    "figure_S2_generation",
    "figure_S3_generation",
    "figure_S4_generation",
    "figure_S6_generation",
    "figure_S7_generation",
    "figure_SIM_generation",
]

DIRECTORIES = {
    "h5_path": "../scripts/image_datasets",
    "model_output_dir": "../scripts/models",

    "figure_output_dir": "./figures",
    "figure_data_dir": "./figure_data",
    "validation_results_dir": "../scripts/results",
    "ext_images_dir": "../scripts/ext_images"
}

def generate_all_figures(
    simulate_image_fn: Callable,
    sim_config: Any,
    camera: Any
):
    figure_S1_generation(**DIRECTORIES) #works
    figure_S2_generation(**DIRECTORIES) #works
    figure_S3_generation(**DIRECTORIES)
    figure_S4_generation(**DIRECTORIES)
    # figure_S6_generation(**DIRECTORIES)
    # figure_S7_generation(**DIRECTORIES)
    figure_SIM_generation(
        simulate_image_fn = simulate_image_fn,
        camera = camera,
        sim_config = sim_config,
        **DIRECTORIES
    )

