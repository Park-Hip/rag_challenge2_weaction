from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Dict, Any
from pathlib import Path
import yaml

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",)

    config_yaml: Dict[str, Any] = {}

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


