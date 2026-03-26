import httpx
from httpx import AsyncClient
import json
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

from src.core.config import settings
from src.core.logger import logger

class Embedder:
    """
    Client for interacting with the Jina v3 Embedding API.
    Handles configuration parsing, batch slicing, and automatic HTTP retries.
    """

    def __init__(self):
        self.batch_size = settings.config_yaml.get('embedder').get("batch_size", 50)
        self.model = settings.config_yaml.get('embedder').get("model", "jina-embeddings-v3")
        self.dimensions = settings.config_yaml.get('embedder').get("dimensions", 1024)
        self.task = settings.config_yaml.get('embedder').get("task", "retrieval.query")
        self.normalize = settings.config_yaml.get('embedder').get("normalize", True)
        self.api_key = settings.JINA_API_KEY
        self.url = settings.EMBEDDING_URL

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def embed_batch(self, batch):
        """
        Sends a sliced batch of text chunks to the Jina API for vectorization.
        Args:
            batch (list[dict]): A sliced list of chunk dictionaries.
        Returns:
            list[dict]: The raw JSON payload returned by the Jina API containing vectors.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "task": self.task,
            "normalized": self.normalize,
            "input": [c["text"] for c in batch],
            "dimensions": self.dimensions
        }
        try:
            async with AsyncClient(timeout=60.0) as client:
                r = await client.post(self.url, headers=headers, json=data)
                r.raise_for_status()
                return r.json().get("data")

        except httpx.HTTPStatusError as e:
            logger.error(
                "Jina API Rejected Request", 
                status_code=e.response.status_code, 
                error_detail=e.response.text
            )
            raise e

        except Exception as e:
            logger.error("Failed to embed batch", error=str(e))
            raise e
    
    async def embed_documents(self, chunks: list[dict]):
        """
        Takes parsed document chunks, splits them into manageable API batches,
        calls the embedding API, and maps the resulting vectors back to the original chunks.
        Args:
            chunks (list[dict]): The full list of chunked documents from the Splitter.
        Returns:
            list[dict]: The mutated list of chunks, now containing an 'embedding' key.
        """

        chunk_len = len(chunks)
        for i in range(0, chunk_len, self.batch_size):
            try: 
                batch = chunks[i: i + self.batch_size]
                embeddings = await self.embed_batch(batch)

                for idx, chunk in enumerate(batch):
                    chunk["embedding"] = embeddings[idx]["embedding"]

            except Exception as e:
                logger.error("Faild to embed documents", error=str(e))
                raise e


