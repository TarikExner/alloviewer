from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/api/health")
def health():
    """
    Return a static service health response.

    Returns
    -------
    dict
        Static mapping with status set to ok.
    """
    return {"status": "ok"}

