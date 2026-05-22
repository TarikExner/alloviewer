import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    cors_origins: list[str]
    plate_layout_store: str

def _default_data_dir() -> Path:
    base_dir = Path(__file__).resolve().parents[1]  # points to backend/
    return (base_dir / "data").resolve()

settings = Settings(
    data_dir=Path(os.getenv("DATA_DIR", str(_default_data_dir()))).resolve(),
    cors_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://alloviewer.org"
    ],
    plate_layout_store=os.getenv("PLATE_LAYOUT_STORE", "/tmp/plate_layouts"),
)
