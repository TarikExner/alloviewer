import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    """
    Store the data directory and permitted cross-origin request origins.

    Attributes
    ----------
    data_dir : Path
        Base directory used to store job data.
    cors_origins : list[str]
        Origins allowed by the CORS middleware.
    """
    data_dir: Path
    cors_origins: list[str]

def _default_data_dir() -> Path:
    """
    Resolve the default data folder relative to the app package.

    Returns
    -------
    Path
        Computed value of ``(base_dir / 'data').resolve()``.
    """
    base_dir = Path(__file__).resolve().parents[1]  # points to backend/
    return (base_dir / "data").resolve()

settings = Settings(
    data_dir=Path(os.getenv("DATA_DIR", str(_default_data_dir()))).resolve(),
    cors_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",

        "https://alloviewer.org",
        "https://www.alloviewer.org",

        "https://alloviewer-woeh.vercel.app",
        "https://alloviewer-f4x0be0x0-tarikexners-projects.vercel.app",
    ],
)
