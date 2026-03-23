import asyncio
import time
import uuid

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.splitter import Splitter
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.core.logger import logger
from src.core.config import settings

async def main():
    run_id = str(uuid.uuid4())
    logger.info("Start running ingestion script", run_id=run_id)
    start = time.time()
    
    try:
        loader = DocumentLoader()
        data_path = settings.DATA_PATH
        docs = loader.load_directory(data_path)

        splitter = Splitter()
        chunks = splitter.split_documents(docs)

        embedder = Embedder()
        await embedder.embed_documents(chunks)

        indexer = Indexer()
        indexer.check_collection_existence()
        indexer.index_documents(chunks)

        run_time = time.time() - start
        logger.info("Successfully ran ingestion script", run_id=run_id, run_time=run_time)

    except Exception as e:
        logger.error("Failed to run ingestion script", error=str(e), run_id=run_id)
        raise e

if __name__ == "__main__":
    asyncio.run(main())
