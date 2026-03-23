from fastapi import APIRouter, HTTPException
from src.api.schemas.models import QueryRequest, QueryResponse
from src.generation.response_builder import ResponseBuilder
from src.core.logger import logger

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response_builder = ResponseBuilder()
        response = await response_builder.build_response(
            query=request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )

        return response
    
    except Exception as e:
        logger.error("Failed to process query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
