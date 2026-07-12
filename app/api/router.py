from fastapi import APIRouter

from .routes.health import router as health_router
from .routes.upload import router as upload_router
from .routes.process_images import router as image_process_router
from .routes.plate_layouts import router as plate_layouts_router
from .routes.fcs_panel import router as fcs_panel_router
from .routes.process_fcxm import router as fcxm_process_router
from .routes.jobs import router as jobs_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(upload_router)
api_router.include_router(image_process_router)
api_router.include_router(plate_layouts_router)
api_router.include_router(fcs_panel_router)
api_router.include_router(fcxm_process_router)
api_router.include_router(jobs_router)

