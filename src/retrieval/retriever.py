from typing import List
import json
import httpx
from httpx import AsyncClient
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from qdrant_client import QdrantClient
from langfuse import observe, get_client

from src.core.config import settings
from src.core.logger import logger

langfuse = get_client()

class Retriever():
    def __init__(self):
        """
        Connects the Qdrant vector database with the Jina Embedding API
        to perform semantic similarity searches on natural language queries.
        """

        # Embedder
        self.model = settings.config_yaml.get('embedder').get("model", "jina-embeddings-v3")
        self.dimensions = settings.config_yaml.get('embedder').get("dimensions", 1024)
        self.task = settings.config_yaml.get('embedder').get("task", "retrieval.query")
        self.normalize = settings.config_yaml.get('embedder').get("normalize", True)
        self.api_key = settings.JINA_API_KEY
        self.url = settings.EMBEDDING_URL

        # Vectorstore
        self.qdrant_url = settings.QDRANT_URL
        self.client  = QdrantClient(url=self.qdrant_url)
        self.collection_name = settings.config_yaml.get('qdrant').get("collection_name", "rag")
        self.limit = settings.config_yaml.get('qdrant').get("limit", 3)

    @observe(as_type="generation", name="jina_embedding")
    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Converts a single user query string into a dense vector array via Jina API.
        """


        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": self.model,
            "task": self.task,
            "normalized": self.normalize,
            "input": [text]
        }
        try:
            async with AsyncClient(timeout=60.0) as client:
                r = await client.post(self.url, headers=headers, json=data)
                r.raise_for_status()
                response_data = r.json()

                usage = response_data.get("usage", {})
                langfuse.update_current_generation(
                    model=self.model,
                    usage_details={
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                )

                return response_data.get("data")[0]["embedding"]

        except Exception as e:
            logger.error("Failed to get embed query", error=str(e))
            raise e

    @observe()
    async def retrieve_documents(self, query: str) -> list:
        """
        Embeds the user's question, searches Qdrant for the closest semantic matches,
        and returns a sorted list of the most relevant document chunks.

        Args:
            query (str): The raw question asked by the user.

        Returns:
            list: A list of Qdrant point objects containing scores and text payloads.
        """


        try:
            query_embedding = await self.embed_text(query)

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=self.limit
            )
            
            formatted_results = [
                {"score": r.score, "text": r.payload.get("text"), "source": r.payload.get("source")}
                for r in search_results.points
            ]
            langfuse.update_current_span(
                output=formatted_results
            )

            return search_results.points
        
        except Exception as e:
            logger.error("Failed to similarity search in Qdrant", error=str(e))
            raise e

        