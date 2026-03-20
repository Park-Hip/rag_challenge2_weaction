from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.cpre.config import settings

class Splitter:
    def __init__(self):
        chunk_size = settings.config_yaml.get("chunking").get("chunk_size", 1000)
        chunk_overlap = settings.config_yaml.get("chunking").get("chunk_overlap", 200)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )
    def split
