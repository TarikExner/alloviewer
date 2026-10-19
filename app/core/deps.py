from alloviewer.image_analysis.storage.repo import LayoutRepo
from .settings import settings

def get_repo() -> LayoutRepo:
    """
    Create the plate-layout repository using the configured store path.

    Returns
    -------
    LayoutRepo
        The plate-layout repository using the configured store path.
    """
    return LayoutRepo(root=settings.plate_layout_store)

