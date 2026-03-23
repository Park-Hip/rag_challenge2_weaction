import pymupdf4llm
from pathlib import Path

from src.core.logger import logger

class DocumentLoader:
    """
    Handles extracting raw text from document files using PyMuPDF4LLM.
    """

    def load_pdf(self, file_path: str):
        """
        Parses a single PDF file and extracts its content as Markdown.
        Args:
            file_path (str): The absolute or relative path to the PDF file.
        Returns:
            str: The extracted Markdown text payload.
        Raises:
            Exception: If the file is corrupted, unreadable, or not found.
        """

        try:
            md = pymupdf4llm.to_markdown(file_path)
            return md
        except Exception as e:
            logger.error("Failed to load file", error=str(e), file_path=file_path)
            raise e

    def load_directory(self, dir_path: str):
        """
        Recursively scans a directory for PDF files and extracts their content as Markdown.
        Args:
            dir_path (str): The directory path containing the raw PDF documents.
        Returns:
            list[dict]: A list of dictionaries taking the shape:
                        {"content": "...", "metadata": {"source": "..."}}
        """
        try:
            docs = []
            path = Path(dir_path)
            pdf_paths = list(path.rglob("*.pdf"))

            for file in pdf_paths:
                try:
                    md = self.load_pdf(str(file))
                except Exception as e:
                    continue

                docs.append({
                    "content": md,
                    "metadata": {"source": str(file)}
                })
            logger.info("Successfully loadded PDF files", dir_path=dir_path, count=len(pdf_paths))
            return docs
            
        except Exception as e:
            logger.warning("Failed to load PDF files from directory", dir_path=dir_path, error=str(e))
            raise e


