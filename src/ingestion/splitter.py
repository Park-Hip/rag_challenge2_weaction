from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from typing import Dict, List
from src.core.logger import logger

class Splitter:
    """
    Responsible for chunking massive Markdown documents into token-optimized slices.
    Relies on `RecursiveCharacterTextSplitter` and configurations defined in `settings.yaml`.
    """
    def __init__(self):
        self.chunk_size = settings.config_yaml.get("chunking").get("chunk_size", 512)
        self.chunk_overlap = settings.config_yaml.get("chunking").get("chunk_overlap", 50)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> List[str]:
        """
        Takes a raw string of text and slices it into overlapping chunks.
        Args:
            text (str): The raw text or markdown string to be chunked.
        Returns:
            List[str]: A list of isolated text chunks.
        """

        try:
            chunks = self.splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error("Failed to split text", error=str(e))
            raise e

    def split_documents(self, docs: List[Dict]):
        """
        Iterates over loaded document payloads, splits their content into chunks,
        and cleanly links every chunk back to its parent document via exact metadata tracking.
        Args:
            docs (List[Dict]): A list of dictionaries emitted by the DocumentLoader.
        Returns:
            List[Dict]: A flattened list of chunks, where each item contains the split 'text' 
                        and extended 'metadata' (including source and chunk_id).
        """
        try:
            chunk_list = []
            for doc in docs:
                chunks = self.split_text(doc.get("content"))

                for i, c in enumerate(chunks):
                    chunk_list.append({
                        "text": c,
                        "metadata": {
                            **doc['metadata'],
                            "chunk_id": i
                        }
                    })
            logger.info(
                "Successfully split documents into chunks",
                count=len(chunk_list),
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap,
            )
            return chunk_list
        except Exception as e:
            logger.error("Failed to split documents", error=str(e))
            raise e



