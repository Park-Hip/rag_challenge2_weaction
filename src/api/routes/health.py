from httpx import AsyncClient
from fastapi.responses import JSONResponse

from fastapi import APIRouter
from src.core.logger import logger
from src.core.config import settings

router = APIRouter()

@router.get("/health")
async  def health_check():
    """
    """

    status_code = 200
    health_status = {"api": "online", "qdrant": "unknown"}

    try:
        async with AsyncClient(timeout=2) as client:
            r = await client.get(f"{settings.QDRANT_URL}/readyz")
            
            if r.status_code == 200:
                health_status['qdrant'] = "online"
            else:
                health_status['qdrant'] = "offline"
                status_code = 503 # 503 Service Unavailable
    except Exception as e:
        logger.error("Healthcheck: Qdrant unreachable", error=str(e))
        health_status["qdrant"] = "offline"
        status_code = 503

    return JSONResponse(content=health_status, status_code=status_code)



