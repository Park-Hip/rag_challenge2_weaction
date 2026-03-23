from fastapi import APIRouter, HTTPException
from src.core.logger import logger
from src.core.config import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.splitter import Splitter
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer

router = APIRouter()

@router.post("/ingest")
async def trigger_ingestion():
    try:
        document_loader = DocumentLoader()
        splitter = Splitter()
        embedder = Embedder()
        indexer = Indexer()

        documents = document_loader.load_directory(settings.DATA_PATH)
        chunks = splitter.split_documents(documents)

        await embedder.embed_documents(chunks)
        indexer.index_documents(chunks)

        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        logger.error("Faile to inges documents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

