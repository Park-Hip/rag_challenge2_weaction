from src.core.langfuse import langfuse, observe

from src.generation.llm_client import LLMClient
from src.core.logger import logger
from src.core.config import settings

class QueryProcessor:
    """
    Optional phase in the RAG pipeline.
    Uses the LLM to rewrite ambiguous user queries into clear, highly-optimized 
    search queries before they are sent to Qdrant.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.prompt = settings.prompts_yaml.get("rewrite")  

    @observe()
    async def optimize_query(self, query: str) -> str:
        """
        Takes the raw user query, passes it strictly to the LLM to strip conversational
        fluff, and returns a dense, keyword-rich string for the Embedder.
        """
        try:
            prompt = self.prompt.format(query=query)
            optimized_query = await self.llm.generate_response(prompt)
            
            logger.info("Query rewritten successfully", original=query, optimized=optimized_query)
            return optimized_query.strip()
            
        except Exception as e:
            logger.error("Failed to process and rewrite query", error=str(e))
            return query
