from langfuse import propagate_attributes
from src.core.langfuse import langfuse, observe

from src.generation.llm_client import LLMClient
from src.retrieval.retriever import Retriever
from src.retrieval.query_processor import QueryProcessor
from src.core.config import settings
from src.core.logger import logger

class ResponseBuilder():
    """
    """

    def __init__(self):
        self.retriever = Retriever()
        self.query_processor = QueryProcessor()
        self.llm = LLMClient()
        self.system_prompt = settings.prompts_yaml.get("qa")  

    @observe(name="build_response")
    async def build_response(self, query: str, user_id: str = None, session_id: str = None) -> dict:
        """
        """
        
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=["rag-qa", "production"]
        ):
            try:
                optimized_query = await self.query_processor.optimize_query(query)
                search_results = await self.retriever.retrieve_documents(optimized_query)

                if not documents:
                    trace_id = langfuse.get_current_trace_id()
                    langfuse.flush()
                    return {
                        "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong kho lưu trữ.", 
                        "sources": [],
                        "trace_id": trace_id
                    }

                context = "\n\n".join([r.payload.get("text") for r in search_results])
                final_prompt = self.system_prompt.format(context=context, query=optimized_query)

                response = await self.llm.generate_response(final_prompt)

                serialized_sources = []
                for r in search_results:
                    serialized_sources.append({
                        "id": str(r.id),
                        "score": float(r.score),
                        "text": r.payload.get("text", ""),
                        "source_file": r.payload.get("source", "Unknown Source")
                    })

                trace_id = langfuse.get_current_trace_id()
                langfuse.flush()

                return {
                    "answer": response, 
                    "sources": serialized_sources,
                    "trace_id": trace_id
                }

            except Exception as e:
                logger.error("Failed to build response", error=str(e))
                raise e