from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any
from pathlib import Path
import yaml

class Settings(BaseSettings):
    EMBEDDING_URL: str = "https://api.jina.ai/v1/embeddings"
    QDRANT_URL: str = "http://localhost:6333"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434/v1"

    JINA_API_KEY: str
    GROQ_API_KEY: str

    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_HOST: str = "http://localhost:3000"

    DATA_PATH: str = "data/raw"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",extra="ignore")

    config_yaml: Dict[str, Any] = {}
    prompts_yaml: Dict[str, Any] = {}

def load_settings() -> Settings:
    settings = Settings()
    
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            settings.config_yaml = yaml.safe_load(f)
            
    prompts_path = Path("config/prompts.yaml")
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            settings.prompts_yaml = yaml.safe_load(f)
            
    return settings

settings = load_settings()


