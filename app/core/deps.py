from alloviewer.image_analysis.storage.repo import LayoutRepo
from .settings import settings

def get_repo() -> LayoutRepo:
    return LayoutRepo(root=settings.plate_layout_store)

