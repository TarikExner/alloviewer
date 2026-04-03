# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.settings import settings
from .api.router import api_router
from . import thumbnails

def create_app() -> FastAPI:
    app = FastAPI(title="Plate Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(thumbnails.router)
    app.include_router(api_router)
    return app

app = create_app()

