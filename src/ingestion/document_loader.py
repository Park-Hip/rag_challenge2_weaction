import pymupdf4llm
from pathlib import Path

from src.core.logger import logger

class DocumentLoader:
    def load_pdf(self, file_path: str):
        "Load a pdf file to markdown"
        try:
            md = pymupdf4llm.to_markdown(file_path)
            return md
        except Exception as e:
            logger.error("Failed to load PDF file", error=str(e), file_path=file_path)
            raise e

    def load_directories(self, dir_path: str):
        "Load multiple PDFs in a directories"
        try:
            docs = []
            path = Path(dir_path)
            pdf_paths = list(path.rglob("*.pdf"))

            for file in pdf_paths:
                try:
                    md = self.load_pdf(str(file))
                except Exception as e:
                    logger.error("Failed to load pdf file", error=str(e), file=str(file))
                    continue

                docs.append({
                    "content": md,
                    "metadata": {"source": str(file)}
                })
            logger.info("Successfully loadded PDF files", dir_path=dir_path, count=len(pdf_paths))
        except Exception as e:
            logger.warning("Failed to load PDF files from directory", dir_path=dir_path, error=str(e))
            raise e


